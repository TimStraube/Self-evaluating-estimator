import random
import numpy


class Umwelt():
    def __init__(self):
        self.battleship = Schiffeversenken(9)
        self.world_state = self.battleship.restart()

    def observe(self):
        return self.world_state

    def act(self, action):
        if action[0] == 1:
            self.battleship.restart()
        else:
            self.world_state = self.battleship.step(
                self.world_state, action[1])


class Schiffeversenken:
    def __init__(self, size):
        # player 0 and 3 as indices for map
        self.rows = size
        self.columns = size
        self.size = size
        self.actions = (
            self.columns * self.rows
        )
        self.moves = 0
        self.player = 1

    def __repr__(self):
        return "battleship"

    def restart(self):
        self.repeat = False
        self.ships_possible = [[3, 2], [3, 2]]
        self.num_shipparts = sum(self.ships_possible[0])
        # initalization of all submaps
        state = numpy.zeros(
            (6, self.columns, self.rows), dtype=numpy.uint8
        )
        self.ships = [[], []]
        self.place_ships(state, self.player)
        self.place_ships(state, -self.player)
        return state

    def shipIndex(self, player):
        # f: {-1, 1} -> {0, 3}
        return 3 * int(player > 0)

    def hitIndex(self, player):
        # f: {-1, 1} -> {1, 4}
        return 3 * int(player > 0) + 1

    def knowledgeIndex(self, player):
        # f: {-1, 1} -> {2, 5}
        return 3 * int(player > 0) + 2

    def step(self, state, action):
        x = action // self.size
        y = action % self.size

        hit = state[self.hitIndex(self.player), x, y]
        ship = state[self.shipIndex(-self.player), x, y]

        self.repeat = False

        if (hit == False and ship == False):
            # hit water
            state[self.hitIndex(self.player), x, y] = 255
            self.player = -self.player
        elif (hit == False and ship == True):
            # hit ship
            state[self.hitIndex(self.player), x, y] = 255
            state[self.knowledgeIndex(self.player), x, y] = 255
            # for ii in range(len(self.ships[int(player > 0)])):
            #     self.ships[int(player > 0)][ii].remove([x, y])
            self.repeat = True
        else:
            self.player = -self.player
        return state

    def get_valid_moves(self, state, player):
        return (
            (state[self.hitIndex(player), :, :] == 0)
            .astype(numpy.uint8)
            .flatten()
        )

    def policy(self, policy, state):
        valid_moves = (
            (state[self.hitIndex(1), :, :] == 0)
            .astype(numpy.uint8)
            .flatten()
        )
        policy *= valid_moves
        policy /= numpy.sum(policy)
        return policy

    def check_win(self, state, action, player):
        state_hit = state[self.hitIndex(player)]
        state_ship = state[self.shipIndex(-player)]
        if (numpy.sum(state_ship * state_hit) ==
                self.num_shipparts):

            return True
        else:
            return False

    def terminated(self, state, action):
        if self.check_win(state, action, 1):
            return 1, True
        if self.check_win(state, action, -1):
            return 1, True
        return 0, False

    def change_perspective(self, state, player):
        # TODO test
        return_state = numpy.zeros(
            (6, self.columns, self.rows),
            dtype=numpy.uint8
        )
        if player == -1:
            state_copy = state[0:3]
            return_state[0:3] = state[3:6]
            return_state[3:6] = state_copy
            return return_state
        else:
            return state

    def get_encoded_state(self, state):
        obsA = (
            state[
                self.hitIndex(1):
                self.knowledgeIndex(1) + 1
            ] == 255
        ).astype(numpy.float32)
        obsB = (
            state[
                self.hitIndex(-1):
                self.knowledgeIndex(-1) + 1
            ] == 255
        ).astype(numpy.float32)
        observation = numpy.concatenate(
            (obsB, obsA),
            axis=0
        )
        # observation = (state == 255).astype(numpy.float32)
        return observation

    def place_ships(self, state, player):
        for ship in self.ships_possible[int(player > 0)]:
            random_direction = random.randint(0, 1)

            # random_direction = 0

            positions = numpy.array([])

            # loop for checking if a ship can be placed
            for i in range(self.size - ship + 1):
                prefix = numpy.zeros(
                    (1, i),
                    dtype=numpy.uint8
                )
                body = numpy.ones(
                    (1, ship),
                    dtype=numpy.uint8
                )
                postfix = numpy.zeros(
                    (1, self.size - ship - i),
                    dtype=numpy.uint8
                )
                ship_possible = numpy.concatenate(
                    (prefix, body, postfix),
                    axis=1
                )
                if random_direction:
                    ship_possible_squeezed = numpy.squeeze(
                        numpy.matmul(
                            ship_possible,
                            numpy.logical_not(
                                state[
                                    self.shipIndex(player), :,
                                    :
                                ]
                            )
                        ) == ship
                    )
                else:
                    transposed_shipmap = numpy.transpose(
                        state[
                            self.shipIndex(player),
                            :,
                            :
                        ]
                    )
                    ship_possible_squeezed = numpy.squeeze(
                        numpy.matmul(
                            ship_possible,
                            numpy.logical_not(
                                transposed_shipmap
                            )
                        ) == ship
                    )
                positions = numpy.append(
                    positions,
                    ship_possible_squeezed,
                    axis=0
                )

            positions = numpy.reshape(
                positions,
                (self.size - ship + 1, self.size)
            )
            possible_positions = numpy.where(positions == 1)

            length_possible_positions = possible_positions[0].size

            random_ship_position = random.randint(
                0,
                length_possible_positions - 1
            )

            x = possible_positions[0][random_ship_position]
            y = possible_positions[1][random_ship_position]

            if random_direction:
                p1 = [x, y]
                p2 = [x + ship - 1, y]
            else:
                p1 = [y, x]
                p2 = [y, x + ship - 1]

            ship_array = self.points_between(p1, p2)

            self.ships[int(player > 0)].append(ship_array)

            for point in ship_array:
                state[
                    self.shipIndex(player),
                    point[0],
                    point[1]
                ] = 255

    def points_between(self, p1, p2):
        points = []

        if p1[0] == p2[0]:
            y_values = list(
                range(
                    min(p1[1], p2[1]),
                    max(p1[1], p2[1]) + 1
                )
            )
            points = [[p1[0], y] for y in y_values]
        elif p1[1] == p2[1]:
            x_values = list(
                range(
                    min(p1[0], p2[0]),
                    max(p1[0], p2[0]) + 1
                )
            )
            points = [[x, p1[1]] for x in x_values]

        return points
