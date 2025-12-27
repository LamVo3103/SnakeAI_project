import heapq
from collections import namedtuple

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20


def heuristic(a, b):
    # Khoảng cách Manhattan
    return abs(a.x - b.x) + abs(a.y - b.y)


def get_path_astar(game):
    start = game.head
    goal = game.food

    # priority_queue lưu: (priority, path)
    # priority = độ dài đường đi hiện tại + dự đoán đến đích (heuristic)
    pq = [(0 + heuristic(start, goal), [start])]
    visited = {start: 0}  # Lưu điểm và chi phí g(n) thấp nhất từng đạt được

    while pq:
        (priority, path) = heapq.heappop(pq)
        current = path[-1]

        if current == goal:
            return path[1:]  # trả về các bước đi tiếp theo (bỏ điểm xuất phát)

        for dx, dy in [(0, -BLOCK_SIZE), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0)]:
            next_pt = Point(current.x + dx, current.y + dy)

            # chi phí mới để đến điểm này
            new_cost = len(path)

            if not game.is_collision(next_pt):
                if next_pt not in visited or new_cost < visited[next_pt]:
                    visited[next_pt] = new_cost
                    new_priority = new_cost + heuristic(next_pt, goal)
                    new_path = list(path)
                    new_path.append(next_pt)
                    heapq.heappush(pq, (new_priority, new_path))
    return None