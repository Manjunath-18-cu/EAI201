import heapq
import math

# ---------- Heuristic Functions ----------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# ---------- Pathfinding Algorithms ----------
def greedy_best_first(start, goal, grid, heuristic):
    rows, cols = len(grid), len(grid[0])
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), start))
    came_from = {}
    visited = set()

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            return reconstruct_path(came_from, start, goal)
        if current in visited:
            continue
        visited.add(current)

        for nx, ny in neighbors(current, rows, cols):
            if grid[nx][ny] == 1 or (nx, ny) in visited:
                continue
            came_from[(nx, ny)] = current
            heapq.heappush(open_list, (heuristic((nx, ny), goal), (nx, ny)))

    return None


def a_star(start, goal, grid, heuristic):
    rows, cols = len(grid), len(grid[0])
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_list:
        _, cost, current = heapq.heappop(open_list)
        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for nx, ny in neighbors(current, rows, cols):
            if grid[nx][ny] == 1:
                continue
            new_cost = g_score[current] + 1
            if (nx, ny) not in g_score or new_cost < g_score[(nx, ny)]:
                g_score[(nx, ny)] = new_cost
                priority = new_cost + heuristic((nx, ny), goal)
                came_from[(nx, ny)] = current
                heapq.heappush(open_list, (priority, new_cost, (nx, ny)))

    return None


# ---------- Utilities ----------
def neighbors(pos, rows, cols):
    x, y = pos
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            yield (nx, ny)

def reconstruct_path(came_from, start, goal):
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = came_from.get(cur)
        if cur is None:
            return None
    path.append(start)
    path.reverse()
    return path


# ---------- Main ----------
original_grid = [
    ['S', 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 'G'],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0]
]

grid = []
start = goal = None
for i in range(len(original_grid)):
    row = []
    for j in range(len(original_grid[0])):
        cell = original_grid[i][j]
        if cell == 'S':
            start = (i, j)
            row.append(0)
        elif cell == 'G':
            goal = (i, j)
            row.append(0)
        else:
            row.append(int(cell))
    grid.append(row)

# --- Run Algorithms ---
gbfs_path = greedy_best_first(start, goal, grid, manhattan)
astar_man = a_star(start, goal, grid, manhattan)
astar_euc = a_star(start, goal, grid, euclidean)

# --- Output Final Paths Only ---
print("Greedy BFS (Manhattan):", gbfs_path)
print("A* (Manhattan):", astar_man)
print("A* (Euclidean):", astar_euc)
