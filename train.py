import core.agent
import core.environment
import core.simulation
import numpy as np
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
        self.checkpoint = 0

        self.save_path = 'save'
        self.agent_path = os.path.join(self.save_path, 'agent')
        self.checkpoint_agent_path = os.path.join(self.agent_path, 'checkpoint')
        self.sample_agent_path = os.path.join(self.agent_path, 'sample')
        self.checkpoint_path = os.path.join(self.save_path, 'checkpoint')

        if not os.path.exists(self.save_path): os.mkdir(self.save_path)
        if not os.path.exists(self.agent_path): os.mkdir(self.agent_path)
        if not os.path.exists(self.checkpoint_agent_path): os.mkdir(self.checkpoint_agent_path)
        if not os.path.exists(self.sample_agent_path): os.mkdir(self.sample_agent_path)
        if not os.path.exists(self.checkpoint_path):
            self.train_agent.init()
            self.train_agent.save(os.path.join(self.checkpoint_agent_path, str(self.checkpoint)), weight_only=True)
            self.train_agent.save(os.path.join(self.sample_agent_path, str(self.checkpoint)), weight_only=True)
            self.train_agent.save(self.agent_path)
            self.save_checkpoint()

    def signal_handler(self, signal, frame):
        self.exit = True

    def run(self):
        self.exit = False
        signal.signal(signal.SIGINT, self.signal_handler)

        self.load_checkpoint()
        self.train_agent.load(self.agent_path)
        self.sample_agent.init()

        while True:
            self.checkpoint += 1

            sample_checkpoint_list = sorted(list(map(int, os.listdir(self.sample_agent_path))), reverse=True)
            while len(sample_checkpoint_list) > 1 and sample_checkpoint_list[-1] < self.checkpoint * (1 - self.metadata['sampling_range']):
                shutil.rmtree(os.path.join(self.sample_agent_path, str(sample_checkpoint_list.pop())))

            sample_checkpoint = sample_checkpoint_list[np.random.choice(len(sample_checkpoint_list), 1)[0]]
            self.sample_agent.load(os.path.join(self.sample_agent_path, str(sample_checkpoint)), weight_only=True)

            self.environment.reset()
            black_result = self.simulation.run(self.environment, self.train_agent, self.sample_agent, black_best_move=False, white_best_move=False, render=False)
            self.print_simulation(self.checkpoint, self.checkpoint - 1, sample_checkpoint, black_result)

            self.environment.reset()
            white_result = self.simulation.run(self.environment, self.sample_agent, self.train_agent, black_best_move=False, white_best_move=False, render=False)
            self.print_simulation(self.checkpoint, sample_checkpoint, self.checkpoint - 1, white_result)

            self.train_agent.train(black_result['data'] + white_result['data'])

            if self.checkpoint % self.metadata['checkpoint_period'] == 0:
                self.train_agent.save(os.path.join(self.checkpoint_agent_path, str(self.checkpoint)), weight_only=True)
            if self.checkpoint % self.metadata['sampling_period'] == 0:
                self.train_agent.save(os.path.join(self.sample_agent_path, str(self.checkpoint)), weight_only=True)
            self.train_agent.save(self.agent_path)
            self.save_checkpoint()

            if self.exit:
                sys.exit()

    def load_checkpoint(self):
        with open(self.checkpoint_path, 'r') as file:
            checkpoint = int(file.read())
        self.checkpoint = checkpoint

    def save_checkpoint(self):
        with open(self.checkpoint_path, 'w') as file:
            file.write(str(self.checkpoint))

    def print_simulation(self, checkpoint, black_checkpoint, white_checkpoint, result):
        sys.stdout.write('checkpoint: {:<6d} | '.format(checkpoint))
        sys.stdout.write('black: {:<6d} | '.format(black_checkpoint))
        sys.stdout.write('white: {:<6d} | '.format(white_checkpoint))
        sys.stdout.write('turn: {:<3d} | '.format(result['turn']))
        sys.stdout.write('win: {:<5} | '.format(result['win']))
        sys.stdout.write('time: {:<8.2f}\n'.format(result['time']))
        sys.stdout.flush()


train = Train()
train.run()
