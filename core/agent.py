from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
import tensorflow as tf
import numpy as np
from threading import Thread
from threading import Lock
import queue
import pickle
import copy
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

K.tensorflow_backend.set_session(session)


class Connect6:
    def __init__(self):
        self.board_size = (19, 19)
        self.model = None
        self.session = None
        self.graph = None

    def init(self):
        self.model = self.build_model()
        self.model._make_predict_function()
        self.session = K.get_session()
        self.graph = tf.get_default_graph()

    def build_model(self):
        inputs = layers.Input(shape=(self.board_size[0], self.board_size[1], 5))
        layer = layers.Conv2D(256, (1, 1), use_bias=False, padding='same', kernel_initializer='he_normal')(inputs)
        for i in range(19):
            conv_layer = layers.BatchNormalization()(layer)
            conv_layer = layers.PReLU()(conv_layer)
            conv_layer = layers.Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_initializer='he_normal')(conv_layer)
            conv_layer = layers.BatchNormalization()(conv_layer)
            conv_layer = layers.PReLU()(conv_layer)
            conv_layer = layers.Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(conv_layer)
            conv_layer = layers.BatchNormalization()(conv_layer)
            conv_layer = layers.PReLU()(conv_layer)
            conv_layer = layers.Conv2D(256, (1, 1), use_bias=False, padding='same', kernel_initializer='he_normal')(conv_layer)
            layer = layers.Add()([layer, conv_layer])
        layer = layers.BatchNormalization()(layer)
        layer = layers.PReLU()(layer)
        layer = layers.Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(layer)
        layer = layers.Flatten()(layer)
        outputs = layers.Activation('softmax')(layer)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(lr=3e-4), loss='categorical_crossentropy')
        return model

    def train(self, batch_data):
        state = []
        action = []
        reward = []
        target = []

        if len(batch_data) == 0:
            return

        for data in batch_data:
            state.append(data['state'])
            action.append(data['action'])
            reward.append(data['reward'])
            target.append(np.zeros(self.board_size[0] * self.board_size[1]))

        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        target = np.array(target)

        for idx in range(len(batch_data)):
            target[idx][action[idx][0] * self.board_size[1] + action[idx][1]] = reward[idx]

        self.model.fit(state, target, epochs=1, verbose=0)

    def save(self, path, weight_only=False):
        model_path = os.path.join(path, 'model')
        if not os.path.exists(path):
            os.mkdir(path)

        if weight_only:
            self.model.save_weights(model_path)
        else:
            self.model.save(model_path)

    def load(self, path, weight_only=False):
        model_path = os.path.join(path, 'model')

        if weight_only:
            self.model.load_weights(model_path)
        else:
            self.model = models.load_model(model_path)
            self.model._make_predict_function()
            self.session = K.get_session()
            self.graph = tf.get_default_graph()

    def get_action(self, environment, best):
        action = None
        if environment.turn == 1:
            action = (self.board_size[0] // 2, self.board_size[1] // 2)
        if action is None:
            action = self.search_vcdt_action(environment)
        if action is None:
            action = self.search_action(environment, 1, best)[0]
        return action

    def search_action(self, environment, count, best):
        player = environment.get_player()
        opponent = environment.get_opponent()
        player_threat = environment.get_threat_count(player)
        opponent_threat = environment.get_threat_count(opponent)

        mask = None
        if mask is None and player_threat > 0:
            mask = self.get_attack_mask(environment)
        if mask is None and opponent_threat > 0:
            mask = self.get_defense_mask(environment)
        if mask is None:
            mask = self.get_basic_mask(environment)

        state = environment.get_state()
        state = environment.pre_processing(state, player, environment.get_stone())
        state = np.array([state])

        with self.session.as_default():
            with self.graph.as_default():
                policy = self.model.predict(state)[0]

        action_list = []

        if best:
            choice_list = np.argsort(policy)
            for choice in reversed(choice_list):
                action = (choice // self.board_size[1], choice % self.board_size[1])
                if mask[action[0]][action[1]] > 0:
                    action_list.append(action)
                    if len(action_list) >= count:
                        return action_list
        else:
            nonzero = np.argwhere(policy > 1e-15).flatten()
            zero = np.argwhere(policy <= 1e-15).flatten()

            nonzero_policy = policy[nonzero]
            choice_list = np.random.choice(nonzero, len(nonzero), p=nonzero_policy, replace=False)
            for choice in choice_list:
                action = (choice // self.board_size[1], choice % self.board_size[1])
                if mask[action[0]][action[1]] > 0:
                    action_list.append(action)
                    if len(action_list) >= count:
                        return action_list

            choice_list = np.random.permutation(zero)
            for choice in choice_list:
                action = (choice // self.board_size[1], choice % self.board_size[1])
                if mask[action[0]][action[1]] > 0:
                    action_list.append(action)
                    if len(action_list) >= count:
                        return action_list

        return action_list

    def search_vcdt_action(self, environment):
        player = environment.get_player()
        search_list = queue.Queue()
        next_search_list = queue.Queue()
        cache = set()
        lock = Lock()
        thread_list = []
        thread_limit = 8

        search_list.put((environment, []))
        while not search_list.empty():
            while not search_list.empty():
                current_environment, current_path = search_list.get()
                current_player = current_environment.get_player()

                if current_environment.check_win():
                    return (current_path[0][0], current_path[0][1])

                args = (current_environment, current_path, next_search_list, cache, lock)
                if current_player == player:
                    thread = Thread(target=self.search_vcdt_attack, args=args)
                    thread.start()
                else:
                    thread = Thread(target=self.search_vcdt_defense, args=args)
                    thread.start()

                thread_list.append(thread)
                while len(thread_list) >= thread_limit:
                    thread_list = [thread for thread in thread_list if thread.isAlive()]

            for thread in thread_list:
                thread.join()

            while not next_search_list.empty():
                search_list.put(next_search_list.get())

        del(cache)
        return None

    def search_vcdt_attack(self, environment, path, search_list, cache, lock):
        player = environment.get_player()
        stone = environment.get_stone()
        action_list = self.search_action(environment, 3, True)

        with lock:
            for action in action_list:
                next_environment = copy.deepcopy(environment)
                next_environment.step(action)
                next_threat = next_environment.get_threat_count(player)

                next_path = copy.deepcopy(path)
                next_path.append(action + (player,))
                if pickle.dumps(sorted(next_path)) in cache:
                    continue

                if not next_environment.check_win() and next_environment.check_done():
                    continue
                if next_environment.check_win() or stone == 2 or (stone == 1 and next_threat >= 2):
                    search_list.put((next_environment, next_path))
                    cache.add(pickle.dumps(sorted(next_path)))

    def search_vcdt_defense(self, environment, path, search_list, cache, lock):
        player = environment.get_player()
        opponent = environment.get_opponent()
        stone = environment.get_stone()
        action = self.search_action(environment, 1, True)[0]

        with lock:
            next_environment = copy.deepcopy(environment)
            next_environment.step(action)
            next_threat = next_environment.get_threat_count(opponent)

            next_path = copy.deepcopy(path)
            next_path.append(action + (player,))
            if pickle.dumps(sorted(next_path)) in cache:
                return

            if not (next_environment.check_done() or (stone == 2 and next_threat <= 0)):
                search_list.put((next_environment, path))
                cache.add(pickle.dumps(sorted(next_path)))

    def get_attack_mask(self, environment):
        player = environment.get_player()
        state = environment.get_state()
        mask = np.zeros(environment.board_size)
        threat_map = environment.get_threat_map(player)

        for y in range(environment.board_size[0]):
            for x in range(environment.board_size[1]):
                if threat_map[y][x] > 0 and state[y][x] == environment.empty:
                    action = (y, x)
                    next_environment = copy.deepcopy(environment)
                    next_environment.step(action)

                    if next_environment.check_win():
                        mask[action[0]][action[1]] = 1
                        return mask
                    if next_environment.check_done():
                        continue
                    if environment.get_stone() == 1:
                        continue
                    if self.get_attack_mask(next_environment) is not None:
                        mask[action[0]][action[1]] = 1
                        return mask

        return None

    def get_defense_mask(self, environment):
        opponent = environment.get_opponent()
        state = environment.get_state()
        mask = np.zeros(environment.board_size)
        threat = environment.get_threat_count(opponent)
        threat_map = environment.get_threat_map(opponent)

        for y in range(environment.board_size[0]):
            for x in range(environment.board_size[1]):
                if threat_map[y][x] > 0 and state[y][x] == environment.empty:
                    action = (y, x)
                    next_environment = copy.deepcopy(environment)
                    next_environment.step(action)
                    next_threat = next_environment.get_threat_count(opponent)

                    if next_environment.check_done() or next_threat - threat < 0:
                        mask[action[0]][action[1]] = 1

        return mask

    def get_basic_mask(self, environment):
        state = environment.get_state()
        mask = np.zeros(environment.board_size)
        bound_map = environment.get_bound_map()

        for y in range(environment.board_size[0]):
            for x in range(environment.board_size[1]):
                if bound_map[y][x] > 0 and state[y][x] == environment.empty:
                    action = (y, x)
                    mask[action[0]][action[1]] = 1

        return mask
