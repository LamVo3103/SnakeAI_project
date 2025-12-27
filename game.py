import pygame
import random
import numpy as np
from collections import namedtuple

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20
SPEED = 500

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()
        self.record = 0

    def reset(self):
        self.direction = 'RIGHT'
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y), Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake: self._place_food()

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Reward Shaping: Thưởng nhỏ khi tiến gần mồi
        old_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
        self._move(action)
        self.snake.insert(0, self.head)
        new_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        reward = 0.1 if new_dist < old_dist else -0.2
        game_over = False

        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            return -20, game_over, self.score # tăng hình phạt khi chết

        if self.head == self.food:
            self.score += 1
            reward = 15 # Thưởng cao khi ăn mồi
            self._place_food()
        else:
            self.snake.pop()
        
        self.update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _move(self, action):
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]): new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]): new_dir = clock_wise[(idx + 1) % 4]
        else: new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == 'RIGHT': x += BLOCK_SIZE
        elif self.direction == 'LEFT': x -= BLOCK_SIZE
        elif self.direction == 'DOWN': y += BLOCK_SIZE
        elif self.direction == 'UP': y -= BLOCK_SIZE
        self.head = Point(x, y)

    def update_ui(self):
        self.display.fill((0,0,0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 0, 255), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0, 100, 255), pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # hiển thị score và record
        font = pygame.font.SysFont('arial', 20, bold=True)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        record_text = font.render(f"Record: {self.record}", True, (255, 255, 0)) # màu kỉ lục vàng
        
        self.display.blit(score_text, [10, 10])
        self.display.blit(record_text, [10, 35]) 
        
        pygame.display.flip()