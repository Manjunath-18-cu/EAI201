from collections import deque
import heapq

graph = {
    "A": [("B", 2), ("C", 3)],
    "B": [("A", 2), ("D", 4), ("E", 1)],
    "C": [("A", 3), ("F", 5), ("G", 2)],
    "D": [("B", 4), ("H", 3)],
    "E": [("B", 1), ("H", 6), ("I", 4)],
    "F": [("C", 5), ("J", 2)],
    "G": [("C", 2), ("I", 3)],
    "H": [("D", 3), ("E", 6), ("J", 7)],
    "I": [("E", 4), ("G", 3), ("J", 1)],
    "J": [("F", 2), ("H", 7), ("I", 1)]
}

def dfs(start, goal):
    stack = [(start, [start], 0)]
    visited = set()
    while stack:
        node, path, cost = stack.pop()
        if node == goal: print("Path:", path, "Cost:", cost); return
        if node not in visited:
            visited.add(node)
            for neigh, c in graph[node]:
                if neigh not in visited:
                    stack.append((neigh, path+[neigh], cost+c))
    print("No path")

def bfs(start, goal):
    queue = deque([(start, [start], 0)])
    visited = set()
    while queue:
        node, path, cost = queue.popleft()
        if node == goal: print("Path:", path, "Cost:", cost); return
        if node not in visited:
            visited.add(node)
            for neigh, c in graph[node]:
                if neigh not in visited:
                    queue.append((neigh, path+[neigh], cost+c))
    print("No path")

def ucs(start, goal):
    pq=[(0,start,[start])]
    visited=set()
    while pq:
        cost,node,path=heapq.heappop(pq)
        if node==goal: print("Path:",path,"Cost:",cost);return
        if node not in visited:
            visited.add(node)
            for neigh,c in graph[node]:
                if neigh not in visited:
                    heapq.heappush(pq,(cost+c,neigh,path+[neigh]))
    print("No path")

start=input("Start: ").upper()
goal=input("Goal: ").upper()
algo=input("Algorithm (DFS/BFS/UCS): ").upper()

if algo=="DFS": dfs(start,goal)
elif algo=="BFS": bfs(start,goal)
elif algo=="UCS": ucs(start,goal)
