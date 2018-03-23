import core.agent
import core.environment
import core.simulation
import json
import sys
import os


class Test:
    def __init__(self):
        with open('test.json', 'r') as file:
            self.metadata = json.load(file)

        self.environment = core.environment.Connect6()
        self.black_agent = core.agent.Connect6()
        self.white_agent = core.agent.Connect6()
        self.simulation = core.simulation.Connect6()

        self.save_path = 'save'
        self.agent_path = os.path.join(self.save_path, 'agent', 'checkpoint')

    def run(self):
        self.black_agent.init()
        self.white_agent.init()

        checkpoint_list = sorted(list(map(int, os.listdir(self.agent_path))), reverse=True)

        black_checkpoint_list = checkpoint_list
        white_checkpoint_list = checkpoint_list
        if self.metadata['black_checkpoint'] is not None:
            black_checkpoint_list = sorted(self.metadata['black_checkpoint'], reverse=True)
        if self.metadata['white_checkpoint'] is not None:
            white_checkpoint_list = sorted(self.metadata['white_checkpoint'], reverse=True)

        for black_checkpoint in black_checkpoint_list:
            self.black_agent.load(os.path.join(self.agent_path, str(black_checkpoint)), weight_only=True)
            for white_checkpoint in white_checkpoint_list:
                self.white_agent.load(os.path.join(self.agent_path, str(white_checkpoint)), weight_only=True)

                self.environment.reset()
                result = self.simulation.run(self.environment, self.black_agent, self.white_agent, black_best_move=True, white_best_move=True, render=False)
                self.print_simulation(black_checkpoint, white_checkpoint, result)

    def print_simulation(self, black_checkpoint, white_checkpoint, result):
        sys.stdout.write('black: {:<6d} | '.format(black_checkpoint))
        sys.stdout.write('white: {:<6d} | '.format(white_checkpoint))
        sys.stdout.write('turn: {:<3d} | '.format(result['turn']))
        sys.stdout.write('win: {:<5} | '.format(result['win']))
        sys.stdout.write('time: {:<8.2f}\n'.format(result['time']))
        sys.stdout.flush()


test = Test()
test.run()
