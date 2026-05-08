import heapq
import copy

trivial = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 0]]

veryEasy = [[1, 2, 3],
            [4, 5, 6],
            [7, 0, 8]]

easy = [[1, 2, 3],
        [4, 5, 6],
        [0, 7, 8]]

medium = [[1, 2, 3],
          [5, 0, 6],
          [4, 7, 8]]

hard = [[8, 7, 1],
        [6, 0, 2],
        [5, 4, 3]]

eight_goal_state = [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 0]]

ALGORITHM_NAMES = {
    1: "Uniform Cost Search",
    2: "A* with Misplaced Tile Heuristic",
    3: "A* with Manhattan Distance Heuristic",
}


class Node:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h

    @property
    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f < other.f


def misplaced_tile_heuristic(current_state, goal_state):
    h_cost = 0
    for i in range(len(current_state)):
        for j in range(len(current_state[i])):
            if current_state[i][j] != goal_state[i][j] and current_state[i][j] != 0:
                h_cost += 1
    return h_cost


def manhattan_distance_heuristic(current_state, goal_state):
    h_cost = 0
    for i in range(len(current_state)):
        for j in range(len(current_state[i])):
            if current_state[i][j] == 0:
                continue
            goal_row = (current_state[i][j] - 1) // 3
            goal_col = (current_state[i][j] - 1) % 3
            h_cost += abs(i - goal_row) + abs(j - goal_col)
    return h_cost


def get_heuristic(algorithm_choice):
    if algorithm_choice == 1:
        return lambda state, goal: 0         
    elif algorithm_choice == 2:
        return misplaced_tile_heuristic
    elif algorithm_choice == 3:
        return manhattan_distance_heuristic
    else:
        raise ValueError(f"Unknown algorithm choice: {algorithm_choice}")


def find_blank(state):
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == 0:
                return i, j
    raise ValueError("No blank tile found in state.")


def get_neighbors(state):
    row, col = find_blank(state)
    n = len(state)
    moves = [
        ("Up",    row - 1, col),
        ("Down",  row + 1, col),
        ("Left",  row,     col - 1),
        ("Right", row,     col + 1),
    ]

    neighbors = []
    for action, new_row, new_col in moves:
        if 0 <= new_row < n and 0 <= new_col < n:
            new_state = copy.deepcopy(state)
            new_state[row][col], new_state[new_row][new_col] = (
                new_state[new_row][new_col],
                new_state[row][col],
            )
            neighbors.append((action, new_state))
    return neighbors


def traceback(node):
    path = []
    current = node
    while current is not None:
        path.append(current)
        current = current.parent
    path.reverse()

    print("\n=== Solution Path ===")
    for step, n in enumerate(path):
        if step == 0:
            print(f"Initial State:")
        else:
            print(f"Step {step} — Move blank {n.action}:  g={n.g}, h={n.h}, f={n.f}")
        for row in n.state:
            print(" ", row)
        print()


def general_search(initial_state, goal_state, algorithm_choice):
    heuristic = get_heuristic(algorithm_choice)

    h0 = heuristic(initial_state, goal_state)
    start_node = Node(state=initial_state, g=0, h=h0)

    priority_queue = []
    heapq.heappush(priority_queue, start_node)

    visited_states = set()
    nodes_expanded = 0
    max_queue_size = 1

    while priority_queue:
        max_queue_size = max(max_queue_size, len(priority_queue))
        current_node = heapq.heappop(priority_queue)

        if current_node.state == goal_state:
            traceback(current_node)
            print(f"Goal reached!")
            print(f"  Solution depth  : {current_node.g}")
            print(f"  Nodes expanded  : {nodes_expanded}")
            print(f"  Max queue size  : {max_queue_size}")
            return current_node

        state_tuple = tuple(tuple(row) for row in current_node.state)
        if state_tuple in visited_states:
            continue
        visited_states.add(state_tuple)
        nodes_expanded += 1

        for action, new_state in get_neighbors(current_node.state):
            child_state_tuple = tuple(tuple(row) for row in new_state)
            if child_state_tuple in visited_states:
                continue

            new_g = current_node.g + 1
            new_h = heuristic(new_state, goal_state)
            child_node = Node(
                state=new_state,
                parent=current_node,
                action=action,
                g=new_g,
                h=new_h,
            )
            heapq.heappush(priority_queue, child_node)

    print("Failure: Queue is empty and goal was not found.")
    return None


# ── CLI / DRIVER ────────────────────────────────────────────────────────────

def read_puzzle():
    print("Enter your puzzle row by row, using space-separated integers.")
    print("Use 0 to represent the blank tile.")
    state = []
    for i in range(3):
        while True:
            raw = input(f"  Row {i + 1}: ").strip().split()
            if len(raw) == 3 and all(r.isdigit() for r in raw):
                state.append([int(r) for r in raw])
                break
            print("  Invalid input — enter exactly 3 integers separated by spaces.")
    return state


def main():
    print("=" * 50)
    print("8-Puzzle Solver")
    print("=" * 50)

    print("\nSelect puzzle:")
    print("  1. Use a default puzzle")
    print("  2. Enter your own puzzle")
    puzzle_choice = input("Choice: ").strip()

    if puzzle_choice == "1":
        defaults = {
            "1": ("Trivial",  trivial),
            "2": ("Very Easy", veryEasy),
            "3": ("Easy",     easy),
            "4": ("Medium",   medium),
            "5": ("Hard", hard)
        }
        print("\nDefault puzzles:")
        for k, (name, _) in defaults.items():
            print(f"  {k}. {name}")
        sel = input("Choice: ").strip()
        if sel not in defaults:
            print("Invalid selection — using Trivial.")
            sel = "1"
        label, initial_state = defaults[sel]
        print(f"\nUsing '{label}' puzzle:")
    else:
        initial_state = read_puzzle()
        print("\nYour puzzle:")

    for row in initial_state:
        print(" ", row)

    print("\nSelect algorithm:")
    for k, name in ALGORITHM_NAMES.items():
        print(f"  {k}. {name}")
    while True:
        alg = input("Choice: ").strip()
        if alg in {"1", "2", "3"}:
            algorithm_choice = int(alg)
            break
        print("  Please enter 1, 2, or 3.")

    print(f"\nRunning {ALGORITHM_NAMES[algorithm_choice]}\n")
    general_search(initial_state, eight_goal_state, algorithm_choice)


if __name__ == "__main__":
    main()