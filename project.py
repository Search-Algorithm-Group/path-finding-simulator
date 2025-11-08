import pygame
import sys
import time
from queue import PriorityQueue


class PathfindingVisualizer:
    """
    A class to visually demonstrate pathfinding algorithms
    using Pygame. Supports A*, Dijkstra, and Greedy Best-First Search.
    """

    # Define color mappings for various grid elements
    COLORS = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'green': (0, 255, 0),
        'red': (255, 0, 0),
        'lightblue': (173, 216, 230),
        'blue': (0, 0, 255),
        'gray': (128, 128, 128)
    }

    def __init__(self):
        # Grid settings
        self.rows = 15
        self.cols = 20
        self.cell_size = 40
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size

        # Initialize Pygame window and clock
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pathfinding Visualizer")
        self.clock = pygame.time.Clock()

        # Grid state: 0 = empty, 1 = wall, 2 = start, 3 = end
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.start = (1, 1)
        self.end = (self.rows - 2, self.cols - 2)

        # Animation and search state
        self.explored_nodes = []
        self.path = []
        self.animation_step = 0
        self.animation_speed = 50  # milliseconds between steps

        self.create_maze()

    def create_maze(self):
        """
        Initialize the grid with walls to form a simple maze layout.
        """
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        # Horizontal walls
        for i in range(2, self.cols - 2):
            self.grid[2][i] = 1
            self.grid[3][i] = 1
            self.grid[8][i] = 1
            self.grid[12][i] = 1

        # Vertical walls
        for i in range(2, self.rows - 2):
            self.grid[i][5] = 1
            self.grid[i][10] = 1
            self.grid[i][15] = 1

        # Additional obstacles
        for i in range(3, 7):
            self.grid[6][i] = 1
            self.grid[10][i + 8] = 1

        # Set start and end points
        self.grid[self.start[0]][self.start[1]] = 2
        self.grid[self.end[0]][self.end[1]] = 3

    def get_neighbors(self, row, col):
        """
        Returns valid non-wall neighbor cells for a given grid cell.
        """
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (
                0 <= new_row < self.rows and
                0 <= new_col < self.cols and
                self.grid[new_row][new_col] != 1
            ):
                neighbors.append((new_row, new_col))
        return neighbors

    def manhattan_distance(self, a, b):
        """
        Returns the Manhattan distance between two points.
        """
        return abs(b[0] - a[0]) + abs(b[1] - a[1])

    def find_path(self, algorithm):
        """
        Executes the selected pathfinding algorithm and returns the path and explored nodes.
        """
        start, end = self.start, self.end
        explored = []
        frontier = PriorityQueue()

        # Initialize the priority queue based on the algorithm
        if algorithm == "astar":
            frontier.put((0, start, 0))  # (priority, node, g_cost)
        elif algorithm == "dijkstra":
            frontier.put((0, start, 0))
        else:  # greedy
            frontier.put((self.manhattan_distance(start, end), start, 0))

        came_from = {start: None}
        cost_so_far = {start: 0}
        visited = set()

        while not frontier.empty():
            _, current, current_g = frontier.get()

            if current in visited:
                continue
            visited.add(current)

            if self.grid[current[0]][current[1]] == 1:
                continue  # Skip wall cells

            explored.append(current)

            if current == end:
                break

            for next_pos in self.get_neighbors(*current):
                if self.grid[next_pos[0]][next_pos[1]] == 1:
                    continue

                new_cost = cost_so_far[current] + 1

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    h_cost = self.manhattan_distance(next_pos, end)

                    # Calculate priority based on algorithm
                    if algorithm == "astar":
                        priority = new_cost + h_cost
                    elif algorithm == "dijkstra":
                        priority = new_cost
                    else:
                        priority = h_cost

                    frontier.put((priority, next_pos, new_cost))
                    came_from[next_pos] = current

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()

        # Ensure path is valid and doesn't cross walls
        if path and path[0] == start and not any(self.grid[r][c] == 1 for r, c in path):
            return path, explored

        return None, explored

    def reset_animation(self):
        """
        Reset the current animation state.
        """
        self.explored_nodes = []
        self.path = []
        self.animation_step = 0

    def run_algorithm(self, algorithm):
        """
        Trigger the selected algorithm and prepare its animation.
        """
        self.reset_animation()
        path, explored = self.find_path(algorithm)
        if path:
            self.explored_nodes = explored
            self.path = path
            self.animation_step = 0

    def draw_grid(self):
        """
        Render the entire grid including walls, explored nodes, and path.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                x = j * self.cell_size
                y = i * self.cell_size

                if self.grid[i][j] == 1:
                    color = self.COLORS['black']
                elif self.grid[i][j] == 2:
                    color = self.COLORS['green']
                elif self.grid[i][j] == 3:
                    color = self.COLORS['red']
                else:
                    color = self.COLORS['white']

                pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.COLORS['gray'], (x, y, self.cell_size, self.cell_size), 1)

        # Draw explored nodes (avoid overwriting walls or start/end)
        for i in range(min(self.animation_step, len(self.explored_nodes))):
            row, col = self.explored_nodes[i]
            if (row, col) != self.start and (row, col) != self.end and self.grid[row][col] != 1:
                x = col * self.cell_size
                y = row * self.cell_size
                pygame.draw.rect(self.screen, self.COLORS['lightblue'], (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.COLORS['gray'], (x, y, self.cell_size, self.cell_size), 1)

        # Draw final path after exploration
        if self.animation_step >= len(self.explored_nodes):
            for i in range(len(self.path) - 1):
                start_pos = self.path[i]
                end_pos = self.path[i + 1]
                start_x = start_pos[1] * self.cell_size + self.cell_size // 2
                start_y = start_pos[0] * self.cell_size + self.cell_size // 2
                end_x = end_pos[1] * self.cell_size + self.cell_size // 2
                end_y = end_pos[0] * self.cell_size + self.cell_size // 2

                pygame.draw.line(self.screen, self.COLORS['blue'], (start_x, start_y), (end_x, end_y), 3)

    def run(self):
        """
        Main application loop to handle events, updates, and drawing.
        """
        last_update = pygame.time.get_ticks()

        while True:
            current_time = pygame.time.get_ticks()

            # Handle user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.run_algorithm("astar")
                    elif event.key == pygame.K_d:
                        self.run_algorithm("dijkstra")
                    elif event.key == pygame.K_g:
                        self.run_algorithm("greedy")
                    elif event.key == pygame.K_r:
                        self.create_maze()
                        self.reset_animation()

            # Advance animation step
            if (
                current_time - last_update > self.animation_speed and
                self.animation_step < len(self.explored_nodes) + 1
            ):
                self.animation_step += 1
                last_update = current_time

            # Draw everything
            self.screen.fill(self.COLORS['white'])
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(60)


# Entry point
if __name__ == "__main__":
    visualizer = PathfindingVisualizer()
    visualizer.run()
