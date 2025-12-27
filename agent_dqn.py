import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer

class Agent:
    def __init__(self):
        self.n_games, self.epsilon, self.gamma = 0, 0, 0.9
        self.memory = deque(maxlen=100_000)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # 14 input, 512 hidden
        self.model = Linear_QNet(14, 512, 3).to(self.device)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        p_l1, p_r1, p_u1, p_d1 = Point(head.x-20, head.y), Point(head.x+20, head.y), Point(head.x, head.y-20), Point(head.x, head.y+20)
        p_l2, p_r2, p_u2, p_d2 = Point(head.x-40, head.y), Point(head.x+40, head.y), Point(head.x, head.y-40), Point(head.x, head.y+40)
        dir_l, dir_r, dir_u, dir_d = game.direction == 'LEFT', game.direction == 'RIGHT', game.direction == 'UP', game.direction == 'DOWN'

        state = [
            # Danger 1 step
            (dir_r and game.is_collision(p_r1)) or (dir_l and game.is_collision(p_l1)) or (dir_u and game.is_collision(p_u1)) or (dir_d and game.is_collision(p_d1)),
            (dir_u and game.is_collision(p_r1)) or (dir_d and game.is_collision(p_l1)) or (dir_l and game.is_collision(p_u1)) or (dir_r and game.is_collision(p_d1)),
            (dir_d and game.is_collision(p_r1)) or (dir_u and game.is_collision(p_l1)) or (dir_r and game.is_collision(p_u1)) or (dir_l and game.is_collision(p_d1)),
            # Danger 2 steps
            (dir_r and game.is_collision(p_r2)) or (dir_l and game.is_collision(p_l2)) or (dir_u and game.is_collision(p_u2)) or (dir_d and game.is_collision(p_d2)),
            (dir_u and game.is_collision(p_r2)) or (dir_d and game.is_collision(p_l2)) or (dir_l and game.is_collision(p_u2)) or (dir_r and game.is_collision(p_d2)),
            (dir_d and game.is_collision(p_r2)) or (dir_u and game.is_collision(p_l2)) or (dir_r and game.is_collision(p_u2)) or (dir_l and game.is_collision(p_d2)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x, game.food.x > game.head.x, game.food.y < game.head.y, game.food.y > game.head.y 
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        mini_sample = random.sample(self.memory, 1000) if len(self.memory) > 1000 else self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, is_test=False):
        self.epsilon = max(0, 150 - self.n_games) if not is_test else 0
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            final_move[random.randint(0, 2)] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            final_move[torch.argmax(self.model(state0)).item()] = 1
        return final_move