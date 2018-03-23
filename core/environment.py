# -*- coding: utf-8 -*-
import numpy as np
import curses
import time
import sys


class Connect6:
    def __init__(self):
        self.board_size = (19, 19)
        self.board = np.zeros(self.board_size)
        self.empty = 0
        self.black = 1
        self.white = 2
        self.connect = 6
        self.turn = 1
        self.win = False
        self.done = False

        self.stone_bound = 3
        self.stone_bound_map = np.zeros((self.board_size[0], self.board_size[1]))
        self.black_threat_count = 0
        self.black_threat_map = np.zeros((self.board_size[0], self.board_size[1]))
        self.white_threat_count = 0
        self.white_threat_map = np.zeros((self.board_size[0], self.board_size[1]))

        self.dx = [0, 1, 1, 1]
        self.dy = [1, 0, -1, 1]

    def reset(self):
        self.__init__()

    def print(self):
        screen = curses.initscr()
        screen.keypad(True)
        curses.curs_set(False)
        curses.mousemask(True)
        curses.noecho()

        screen.clear()
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if self.board[i][j] == self.empty:
                    screen.insch(i, j * 2, '┼')
                elif self.board[i][j] == self.black:
                    screen.insch(i, j * 2, '○')
                elif self.board[i][j] == self.white:
                    screen.insch(i, j * 2, '●')
        screen.refresh()
        time.sleep(0.5)

    def input(self):
        self.print()

        screen = curses.initscr()
        screen.keypad(True)
        curses.curs_set(False)
        curses.mousemask(True)
        curses.noecho()

        while True:
            event = screen.getch()
            if event == 27:
                sys.exit()
            if event == curses.KEY_MOUSE:
                _, mx, my, _, _ = curses.getmouse()
                pos = (my, mx // 2)
                if self.check_pos(pos) and self.board[pos[0]][pos[1]] == self.empty:
                    break

        return pos

    def step(self, pos):
        if self.check_pos(pos) and self.board[pos[0]][pos[1]] == self.empty:
            self.board[pos[0]][pos[1]] = self.get_player()
        else:
            raise IndexError()

        self.update_win(pos)
        self.update_bound(pos)
        self.update_threat(pos)
        self.turn += 1

    def pre_processing(self, state, player, stone):
        if player == self.black:
            state1 = (state == self.black).astype(int)[:, :, np.newaxis]
            state2 = (state == self.white).astype(int)[:, :, np.newaxis]
        else:
            state1 = (state == self.white).astype(int)[:, :, np.newaxis]
            state2 = (state == self.black).astype(int)[:, :, np.newaxis]
        if stone == 2:
            state3 = np.ones(self.board_size)[:, :, np.newaxis]
            state4 = np.zeros(self.board_size)[:, :, np.newaxis]
        else:
            state3 = np.zeros(self.board_size)[:, :, np.newaxis]
            state4 = np.ones(self.board_size)[:, :, np.newaxis]
        state5 = np.ones(self.board_size)[:, :, np.newaxis]

        prep_state = np.concatenate((state1, state2, state3, state4, state5), axis=-1)
        return prep_state

    def check_pos(self, pos):
        return pos[0] >= 0 and pos[1] >= 0 and pos[0] < self.board_size[0] and pos[1] < self.board_size[1]

    def check_win(self):
        return self.win

    def check_done(self):
        return self.win or self.turn >= self.board_size[0] * self.board_size[1]

    def get_player(self):
        if (self.turn // 2) % 2 == 0:
            return self.black
        else:
            return self.white

    def get_opponent(self):
        player = self.get_player()
        if player == self.black:
            return self.white
        elif player == self.white:
            return self.black

    def get_stone(self):
        if self.turn % 2 == 0:
            return 2
        else:
            return 1

    def get_state(self):
        return np.array(self.board)

    def get_bound_map(self):
        return np.array(self.stone_bound_map)

    def get_threat_count(self, player):
        if player == self.black:
            return self.black_threat_count
        elif player == self.white:
            return self.white_threat_count

    def get_threat_map(self, player):
        if player == self.black:
            return np.array(self.black_threat_map)
        elif player == self.white:
            return np.array(self.white_threat_map)

    def update_win(self, pos):
        player = self.get_player()
        board = self.get_state()

        for i in range(4):
            count = 0
            for j in range(-self.connect + 1, self.connect):
                target = [pos[0] + self.dy[i] * j, pos[1] + self.dx[i] * j]

                if self.check_pos(target):
                    if board[target[0]][target[1]] == player:
                        count += 1
                    else:
                        count = 0
                else:
                    count = 0

                if count >= self.connect:
                    self.win = True

    def update_bound(self, pos):
        for i in range(4):
            for j in range(-self.stone_bound, self.stone_bound + 1):
                target = [pos[0] + self.dy[i] * j, pos[1] + self.dx[i] * j]

                if self.check_pos(target):
                    self.stone_bound_map[target[0]][target[1]] = 1

    def update_threat(self, pos):
        player = self.get_player()
        board = self.get_state()

        black_threat_count_delta = 0
        black_threat_map_delta = np.zeros((board.shape[0], board.shape[1]))
        white_threat_count_delta = 0
        white_threat_map_delta = np.zeros((board.shape[0], board.shape[1]))

        board[pos[0]][pos[1]] = self.empty
        black_threat_count, black_threat_map = self.get_threat_line(board, pos, self.black)
        black_threat_count_delta -= black_threat_count
        black_threat_map_delta = np.subtract(black_threat_map_delta, black_threat_map)
        white_threat_count, white_threat_map = self.get_threat_line(board, pos, self.white)
        white_threat_count_delta -= white_threat_count
        white_threat_map_delta = np.subtract(white_threat_map_delta, white_threat_map)

        board[pos[0]][pos[1]] = player
        black_threat_count, black_threat_map = self.get_threat_line(board, pos, self.black)
        black_threat_count_delta += black_threat_count
        black_threat_map_delta = np.add(black_threat_map_delta, black_threat_map)
        white_threat_count, white_threat_map = self.get_threat_line(board, pos, self.white)
        white_threat_count_delta += white_threat_count
        white_threat_map_delta = np.add(white_threat_map_delta, white_threat_map)

        self.black_threat_count += black_threat_count_delta
        self.black_threat_map = np.add(self.black_threat_map, black_threat_map_delta)
        self.white_threat_count += white_threat_count_delta
        self.white_threat_map = np.add(self.white_threat_map, white_threat_map_delta)

    def get_threat_line(self, board, pos, player):
        threat_count = 0
        threat_map = np.zeros((self.board_size[0], self.board_size[1]))

        if player == self.black:
            op_player = self.white
        else:
            op_player = self.black

        for i in range(4):
            target = [pos[0], pos[1]]
            while self.check_pos(target):
                target[0] -= self.dy[i]
                target[1] -= self.dx[i]

            player_list = []
            empty_list = []
            while True:
                target[0] += self.dy[i]
                target[1] += self.dx[i]
                if not self.check_pos(target):
                    break

                while len(player_list) != 0:
                    distance = max([abs(target[0] - player_list[0][0]), abs(target[1] - player_list[0][1])])
                    if distance < self.connect:
                        break
                    player_list.pop(0)
                while len(empty_list) != 0:
                    distance = max([abs(target[0] - empty_list[0][0]), abs(target[1] - empty_list[0][1])])
                    if distance < self.connect:
                        break
                    empty_list.pop(0)

                if board[target[0]][target[1]] == self.empty:
                    empty_list.append([target[0], target[1]])
                elif board[target[0]][target[1]] == player:
                    player_list.append([target[0], target[1]])
                elif board[target[0]][target[1]] == op_player:
                    player_list = []
                    empty_list = []

                if len(player_list) >= self.connect - 2 and len(empty_list) == self.connect - len(player_list):
                    if len(empty_list) == 0:
                        continue

                    threat_count += 1
                    for j in range(self.connect):
                        threat_map[target[0] - self.dy[i] * j][target[1] - self.dx[i] * j] += 1

                    target = [empty_list[-1][0], empty_list[-1][1]]
                    player_list = []
                    empty_list = []

        return threat_count, threat_map
