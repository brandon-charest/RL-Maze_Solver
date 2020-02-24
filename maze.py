import numpy as np
import random
import time
from threeviz.api import plot_3d, plot_line_seg


class Agent:
    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j

    @property
    def loc(self):
        return self.i, self.j

    def vmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i + direction, self.j)

    def hmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i, self.j + direction)

    def __repr__(self):
        return str(self.loc)


class QLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=1.0):
        self.q = np.zeros((num_states, num_actions))
        self.a = learning_rate
        self.g = discount_factor

    # https://en.wikipedia.org/wiki/Q-learning
    def update(self, st, at, rt, st1):
        self.q[st, at] = (1 - self.a) * self.q[st, at] + self.a * (rt + self.g * np.max(self.q[st1]))


WALL = -1
GOAL = 1
class Maze:
    def __init__(self, rows=4, cols=4):
        self.maze = np.zeros((rows, cols))
        self.agent = Agent(0, 0)

    def reset(self):
        self.agent.i = 0
        self.agent.j = 0

    def state_for_agent(self, a):
        nr, nc = self.maze.shape
        return a.i * nc + a.j

    def in_bounds(self, i, j):
        nr, nc = self.maze.shape
        return 0 <= i < nr and 0 <= j < nc

    def agent_in_bounds(self, a):
        return self.in_bounds(a.i, a.j)

    def agent_would_not_die(self, a):
        return not self.maze[a.i, a.j] == WALL

    def visualize(self):
        nr, nc = self.maze.shape
        z = -0.1
        a = self.agent
        plot_line_seg(0, 0, z, nr, 0, z, 'e1', size=0.2, color='red')
        plot_line_seg(0, 0, z, 0, nc, z, 'e2', size=0.2, color='red')
        plot_line_seg(0, nc, z, nr, nc, z, 'e3', size=0.2, color='red')
        plot_line_seg(nr, 0, z, nr, nc, z, 'e4', size=0.2, color='red')
        plot_3d(*get_midpoint_for_loc(a.i, a.j), z, 'agent', color='blue', size=1)
        plot_3d(*get_midpoint_for_loc(nr-1, nc-1), z, 'goal', color='green', size=1)

        x, y = np.where(self.maze == WALL)
        plot_3d(x + 0.5, y + 0.5, [z]*len(x), 'wall', color='red', size=1)

    def is_valid_new_agent(self, a):
        return self.agent_in_bounds(a) and self.agent_would_not_die(a)

    @property
    def all_actions(self):
        a = self.agent
        return [
            a.vmove(1),
            a.vmove(-1),
            a.hmove(1),
            a.hmove(-1)
        ]

    def compute_possible_moves(self):
        moves = self.all_actions
        return [(m, idx) for idx, m in enumerate(moves) if self.is_valid_new_agent(m)]

    def do_a_move(self, a):
        assert self.is_valid_new_agent(a), "Agent can't go there."
        self.agent = a
        return 10 if self.has_won() else -0.1

    def has_won(self):
        a = self.agent
        return self.maze[a.i, a.j] == GOAL


def get_midpoint_for_loc(i, j):
    return i + 0.5, j + 0.5


def make_test_maze(size=4):
    m = Maze(size, size)
    e = m.maze
    height, width = e.shape
    e[-1, -1] = GOAL
    for i in range(len(e)):
        for j in range(len(e[i])):
            if i in [0, height-1] and j in [width-1, 0]:
                continue
            if random.random() < 0.3:
                e[i, j] = WALL
    return m


def main():
    size = 8
    q = QLearning(size**2, 4)
    final_score = 0
    m = make_test_maze(size)
    m.visualize()
    cont = input()
    if cont.lower() == 'n':
        return
    agents = []
    for i in range(1000):
        m.reset()
        iterations = 0
        while not m.has_won():
            iterations += 1

            if random.random() > 0.5 or i < 200:
                moves = m.compute_possible_moves()
                random.shuffle(moves)
                move, move_idx = moves[0]
            else:
                moves = m.all_actions
                s = m.state_for_agent(m.agent)
                move_idx = np.argmax(q.q[s])
                move = moves[move_idx]

            at = move_idx
            st = m.state_for_agent(m.agent)

            agents.append(m.agent)

            score = m.do_a_move(move)
            final_score += score
            rt = score
            st1 = m.state_for_agent(m.agent)

            q.update(st, at, rt, st1)
        print(f'I={i}: Final score of {final_score} in {iterations} iterations.')

    print(q.q)
    m.reset()
    m.visualize()
    re_run = True
    while re_run:
        while not m.has_won():
            time.sleep(0.1)
            s = m.state_for_agent(m.agent)
            action_idx = np.argmax(q.q[s])
            m.do_a_move(m.all_actions[action_idx])
            m.visualize()
        run = input("Run again? y/n")
        if run.lower() == 'n':
            re_run = False
        else:
            m.reset()


if __name__ == '__main__':
    main()
