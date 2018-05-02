import core.agent
import core.environment
import core.simulation
import numpy as np
import pickle
import signal
import shutil
import json
import sys
import os


class Train:
    def __init__(self):
        with open('train.json', 'r') as file:
            self.metadata = json.load(file)

        self.environment = core.environment.Connect6()
        self.train_agent = core.agent.Connect6()
        self.sample_agent = core.agent.Connect6()
        self.simulation = core.simulation.Connect6()
        self.iteration = 0
        self.evaluation = np.zeros((1, 1))
        self.win_encoding = {'black': 0, 'white': 1, 'draw': 0.5}

        self.save_path = 'save'
        self.agent_path = os.path.join(self.save_path, 'agent')
        self.iteration_path = os.path.join(self.save_path, 'iteration')
        self.evaluation_path = os.path.join(self.save_path, 'evaluation')

        if not os.path.exists(self.save_path): os.mkdir(self.save_path)
        if not os.path.exists(self.agent_path): os.mkdir(self.agent_path)
        if not os.path.exists(self.iteration_path):
            self.train_agent.init()
            self.train_agent.save(os.path.join(self.agent_path, str(self.iteration)), weight_only=True)
            self.train_agent.save(self.save_path)
            self.save_iteration()

            self.environment.reset()
            result = self.simulation.run(self.environment, self.train_agent, self.train_agent, black_best_move=True, white_best_move=True, render=False)
            self.evaluation[0][0] = self.win_encoding[result['win']]
            self.save_evaluation()

    def signal_handler(self, signal, frame):
        self.exit = True

    def run(self):
        self.exit = False
        signal.signal(signal.SIGINT, self.signal_handler)

        self.load_iteration()
        self.load_evaluation()
        self.train_agent.load(self.save_path)
        self.sample_agent.init()

        while True:
            self.iteration += 1

            black_win = np.sum(np.where(self.evaluation != 1, 1, 0), axis=1)
            white_win = np.sum(np.where(self.evaluation != 0, 1, 0), axis=0)
            black_checkpoint_list = np.argsort(black_win) * self.metadata['checkpoint_period']
            white_checkpoint_list = np.argsort(white_win) * self.metadata['checkpoint_period']
            black_checkpoint_list = black_checkpoint_list[-self.metadata['sampling_range']:]
            white_checkpoint_list = white_checkpoint_list[-self.metadata['sampling_range']:]

            black_checkpoint = np.random.choice(black_checkpoint_list, 1)[0]
            white_checkpoint = np.random.choice(white_checkpoint_list, 1)[0]

            self.sample_agent.load(os.path.join(self.agent_path, str(white_checkpoint)), weight_only=True)
            self.environment.reset()
            black_result = self.simulation.run(self.environment, self.train_agent, self.sample_agent, black_best_move=False, white_best_move=False, render=False)
            self.print_simulation(self.iteration - 1, white_checkpoint, black_result)

            self.sample_agent.load(os.path.join(self.agent_path, str(black_checkpoint)), weight_only=True)
            self.environment.reset()
            white_result = self.simulation.run(self.environment, self.sample_agent, self.train_agent, black_best_move=False, white_best_move=False, render=False)
            self.print_simulation(black_checkpoint, self.iteration - 1, white_result)

            self.train_agent.train(black_result['data'] + white_result['data'])

            if self.iteration % self.metadata['checkpoint_period'] == 0:
                self.evaluation = np.pad(self.evaluation, 1, 'constant', constant_values=0)[1:, 1:]

                for i in range(len(self.evaluation) - 1):
                    sample_checkpoint = i * self.metadata['checkpoint_period']
                    self.sample_agent.load(os.path.join(self.agent_path, str(sample_checkpoint)), weight_only=True)

                    self.environment.reset()
                    result = self.simulation.run(self.environment, self.train_agent, self.sample_agent, black_best_move=True, white_best_move=True, render=False)
                    self.print_simulation(self.iteration, sample_checkpoint, result)
                    self.evaluation[-1][i] = self.win_encoding[result['win']]

                    self.environment.reset()
                    result = self.simulation.run(self.environment, self.sample_agent, self.train_agent, black_best_move=True, white_best_move=True, render=False)
                    self.print_simulation(sample_checkpoint, self.iteration, result)
                    self.evaluation[i][-1] = self.win_encoding[result['win']]

                self.environment.reset()
                result = self.simulation.run(self.environment, self.train_agent, self.train_agent, black_best_move=True, white_best_move=True, render=False)
                self.print_simulation(self.iteration, self.iteration, result)
                self.evaluation[-1][-1] = self.win_encoding[result['win']]

                self.train_agent.save(os.path.join(self.agent_path, str(self.iteration)), weight_only=True)
                self.save_evaluation()

            self.train_agent.save(self.save_path)
            self.save_iteration()

            if self.exit:
                sys.exit()

    def load_iteration(self):
        with open(self.iteration_path, 'r') as file:
            iteration = int(file.read())
        self.iteration = iteration

    def save_iteration(self):
        with open(self.iteration_path, 'w') as file:
            file.write(str(self.iteration))

    def load_evaluation(self):
        with open(self.evaluation_path, 'rb') as file:
            evaluation = pickle.load(file)
        self.evaluation = evaluation

    def save_evaluation(self):
        with open(self.evaluation_path, 'wb') as file:
            pickle.dump(self.evaluation, file)

    def print_simulation(self, black_checkpoint, white_checkpoint, result):
        sys.stdout.write('black: {:<6d} | '.format(black_checkpoint))
        sys.stdout.write('white: {:<6d} | '.format(white_checkpoint))
        sys.stdout.write('turn: {:<3d} | '.format(result['turn']))
        sys.stdout.write('win: {:<5} | '.format(result['win']))
        sys.stdout.write('time: {:<8.2f}\n'.format(result['time']))
        sys.stdout.flush()


train = Train()
train.run()
