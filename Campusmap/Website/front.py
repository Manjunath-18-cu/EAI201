from flask import Flask, render_template, request, jsonify
import heapq
from collections import deque

app = Flask(__name__)

# Graph edges (updated to match the coordinate names)
edges = [
    ("main gate", "id gate", 100),
    ("id gate", "AC1", 80),
    ("AC1", "lawn", 60),
    ("AC1", "cafe", 70),
    ("AC1", "lib", 90),
    ("lawn", "cafe", 40),
    ("lawn", "lib", 50),
    ("lawn", "AC2", 75),
    ("AC2", "food court", 65),
    ("food court", "hostel", 85),
    ("hostel", "sports", 100),
]

# Build graph
graph = {}
for u, v, w in edges:
    graph.setdefault(u, []).append((v, w))
    graph.setdefault(v, []).append((u, w))  # undirected

# Locations for dropdowns (capitalized for better display)
locations = sorted([loc.title() for loc in graph.keys()])

# Coordinates for each location (fixed to match graph node names)
location_coordinates = {
    "main gate": {"lat": 13.2200, "lng": 77.7539},
    "id gate": {"lat": 13.2214, "lng": 77.7549},
    "ac1": {"lat": 13.2218, "lng": 77.7550},
    "lawn": {"lat": 13.222775, "lng": 77.755576},
    "cafe": {"lat": 13.22242, "lng": 77.755158},
    "lib": {"lat": 13.221971, "lng": 77.75558},
    "ac2": {"lat": 13.22336, "lng": 77.755963},
    "food court": {"lat": 13.224758, "lng": 77.75725},
    "hostel": {"lat": 13.224195, "lng": 77.758613},
    "sports": {"lat": 13.22887, "lng": 77.7572}
}

# Helper function to convert to lowercase for internal processing
def get_lowercase_name(name):
    return name.lower()

# Search algorithms
def bfs(start, goal):
    start_lower = get_lowercase_name(start)
    goal_lower = get_lowercase_name(goal)
    
    queue = deque([[start_lower]])
    visited = set()
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal_lower:
            return [loc.title() for loc in path]  # Return capitalized path
        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph.get(node, []):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None

def dfs(start, goal):
    start_lower = get_lowercase_name(start)
    goal_lower = get_lowercase_name(goal)
    
    stack = [[start_lower]]
    visited = set()
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal_lower:
            return [loc.title() for loc in path]  # Return capitalized path
        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph.get(node, []):
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)
    return None

def ucs(start, goal):
    start_lower = get_lowercase_name(start)
    goal_lower = get_lowercase_name(goal)
    
    pq = [(0, [start_lower])]
    visited = set()
    while pq:
        cost, path = heapq.heappop(pq)
        node = path[-1]
        if node == goal_lower:
            return [loc.title() for loc in path], cost  # Return capitalized path
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph.get(node, []):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(pq, (cost + weight, new_path))
    return None, float('inf')

# Routes
@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/navigate', methods=['GET', 'POST'])
def navigate():
    if request.method == 'POST':
        start_location = request.form['start']
        goal_location = request.form['goal']
        algorithm = request.form['algorithm']

        if algorithm == "bfs":
            path = bfs(start_location, goal_location)
            cost = None
        elif algorithm == "dfs":
            path = dfs(start_location, goal_location)
            cost = None
        elif algorithm == "ucs":
            path, cost = ucs(start_location, goal_location)
        else:
            path, cost = None, None

        return render_template(
            'index.html',
            path=path,
            cost=cost,
            locations=locations,
            selected_start=start_location,
            selected_goal=goal_location,
            selected_algorithm=algorithm
        )

    return render_template('index.html', locations=locations)

# API endpoint for AJAX requests
@app.route('/find_path', methods=['POST'])
def find_path_api():
    data = request.get_json()
    start = data.get('start')
    goal = data.get('goal')
    algo = data.get('algorithm')
    
    if not start or not goal or not algo:
        return jsonify({'error': 'Missing parameters'}), 400
    
    if algo == 'bfs':
        path = bfs(start, goal)
        cost = None
    elif algo == 'dfs':
        path = dfs(start, goal)
        cost = None
    elif algo == 'ucs':
        path, cost = ucs(start, goal)
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400
    
    # Convert path to coordinates for map display
    path_coordinates = []
    if path:
        for location in path:
            loc_lower = get_lowercase_name(location)
            if loc_lower in location_coordinates:
                path_coordinates.append([location_coordinates[loc_lower]['lat'], location_coordinates[loc_lower]['lng']])
    
    if path:
        return jsonify({
            'path': path,
            'path_coordinates': path_coordinates,
            'cost': cost,
            'success': True
        })
    else:
        return jsonify({
            'path': [],
            'path_coordinates': [],
            'cost': None,
            'success': False,
            'error': 'No path found'
        })

# API endpoint to get all location coordinates
@app.route('/get_locations', methods=['GET'])
def get_locations():
    return jsonify(location_coordinates)

if __name__ == '__main__':
    app.run(debug=True)
