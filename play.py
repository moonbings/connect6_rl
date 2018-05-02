import core.agent
import core.environment
import core.simulation
import json
import sys
import os


class Play:
    def __init__(self):
        with open('play.json', 'r') as file:
            self.metadata = json.load(file)

        self.environment = core.environment.Connect6()
        self.black_agent = None
        self.white_agent = None
        if self.metadata['black_checkpoint'] is not None:
            self.black_agent = core.agent.Connect6()
        if self.metadata['white_checkpoint'] is not None:
            self.white_agent = core.agent.Connect6()
        self.simulation = core.simulation.Connect6()

        self.save_path = 'save'
        self.agent_path = os.path.join(self.save_path, 'agent')

    def run(self):
        if self.metadata['black_checkpoint'] is not None:
            self.black_agent.init()
            self.black_agent.load(os.path.join(self.agent_path, str(self.metadata['black_checkpoint'])), weight_only=True)
        if self.metadata['white_checkpoint'] is not None:
            self.white_agent.init()
            self.white_agent.load(os.path.join(self.agent_path, str(self.metadata['white_checkpoint'])), weight_only=True)

        self.environment.reset()

        if self.metadata['board'] is not None:
            black_stone = []
            white_stone = []
            for y in range(self.environment.board_size[0]):
                for x in range(self.environment.board_size[1]):
                    if self.metadata['board'][y][x] == 'b':
                        black_stone.append((y, x))
                    elif self.metadata['board'][y][x] == 'w':
                        white_stone.append((y, x))

            while len(black_stone) != 0 or len(white_stone) != 0:
                player = self.environment.get_player()
                if player == self.environment.black:
                    if len(black_stone) == 0:
                        raise ValueError()
                    self.environment.step(black_stone.pop())
                elif player == self.environment.white:
                    if len(white_stone) == 0:
                        raise ValueError()
                    self.environment.step(white_stone.pop())
            if self.environment.check_done():
                raise ValueError()

        result = self.simulation.run(self.environment, self.black_agent, self.white_agent, black_best_move=True, white_best_move=True, render=True)
        self.print_simulation(result)

    def print_simulation(self, result):
        sys.stdout.write('turn: {:<3d} | '.format(result['turn']))
        sys.stdout.write('win: {:<5} | '.format(result['win']))
        sys.stdout.write('time: {:<8.2f}\n'.format(result['time']))
        sys.stdout.flush()


play = Play()
play.run()
