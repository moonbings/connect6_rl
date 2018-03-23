import numpy as np
import time
import copy


class Connect6:
    def run(self, environment, black_agent, white_agent, black_best_move=False, white_best_move=False, render=False):
        if render:
            environment.print()

        win = 'draw'
        black_data = []
        white_data = []
        start_time = time.time()

        while True:
            player = environment.get_player()
            state = environment.get_state()
            state = environment.pre_processing(state, player, environment.get_stone())

            if player == environment.black:
                agent = black_agent
            else:
                agent = white_agent

            if player == environment.black:
                best_move = black_best_move
            else:
                best_move = white_best_move

            if agent is None:
                action = environment.input()
            else:
                action = agent.get_action(environment, best_move)

            environment.step(action)
            if render:
                environment.print()

            if environment.turn - 1 > 1:
                if player == environment.black:
                    black_data.append({
                        'state': np.array(state),
                        'action': np.array(action),
                        'turn': environment.turn - 1
                    })
                elif player == environment.white:
                    white_data.append({
                        'state': np.array(state),
                        'action': np.array(action),
                        'turn': environment.turn - 1
                    })

            if environment.check_done():
                if environment.check_win():
                    if player == environment.black:
                        win = 'black'
                    elif player == environment.white:
                        win = 'white'
                break

        end_time = time.time()

        data = []
        if win == 'black':
            for idx in range(len(black_data)):
                black_data[idx]['reward'] = 1
            data = black_data
        if win == 'white':
            for idx in range(len(white_data)):
                white_data[idx]['reward'] = 1
            data = white_data

        return {
            'turn': environment.turn - 1,
            'win': win,
            'time': end_time - start_time,
            'data': self.augment_data(data)
        }

    def augment_data(self, batch_data):
        augmented_batch_data = []

        for data in batch_data:
            augmented_data = copy.deepcopy(data)

            for i in range(4):
                augmented_batch_data.append(copy.deepcopy(augmented_data))
                augmented_data['state'] = self.rotate_state(augmented_data['state'])
                augmented_data['action'] = self.rotate_action(augmented_data['action'])

            augmented_data['state'] = self.flip_state(augmented_data['state'])
            augmented_data['action'] = self.flip_action(augmented_data['action'])
            for i in range(4):
                augmented_batch_data.append(copy.deepcopy(augmented_data))
                augmented_data['state'] = self.rotate_state(augmented_data['state'])
                augmented_data['action'] = self.rotate_action(augmented_data['action'])

        return augmented_batch_data

    def rotate_state(self, state):
        return np.rot90(state, k=1)

    def flip_state(self, state):
        return np.array(state)[::-1]

    def rotate_action(self, action):
        return 19 - 1 - action[1], action[0]

    def flip_action(self, action):
        return 19 - 1 - action[0], action[1]
