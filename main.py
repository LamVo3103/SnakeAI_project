import pygame
import time
from game import SnakeGameAI, BLOCK_SIZE
from agent import get_path_astar

def main():
    game = SnakeGameAI()
    
    astar_record = 0

    running = True
    while running:
        # thoát
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # AI tìm đường đi bằng A*
        path = get_path_astar(game)

        if path:
            # lấy bước tiếp theo
            next_step = path[0]

            # cập nhật vị trí rắn
            game.head = next_step
            game.snake.insert(0, game.head)

            # kiểm tra có ăn mồi hay chưa
            if game.head == game.food:
                game.score += 1
                # cập nhật kỷ lục ngay khi điểm số tăng
                if game.score > astar_record:
                    astar_record = game.score
                    game.record = astar_record
                game._place_food()
            else:
                game.snake.pop()
        else:
            # AI bị bí đường
            print(f"AI bị bí đường! Điểm đạt được: {game.score}")
            time.sleep(1)
            game.reset()
            game.record = astar_record

        #cập nhập giao diện
        game.update_ui()

        # thua cuộc
        if game.is_collision():
            print(f"Game Over! Điểm số: {game.score}")
            # Cập nhật kỷ lục trước khi reset điểm về 0
            if game.score > astar_record:
                astar_record = game.score
            
            time.sleep(1)
            game.reset()
            game.record = astar_record

        game.clock.tick(20) # Tốc độ hiển thị A*

    pygame.quit()

if __name__ == "__main__":
    main()