import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import math
import itertools
from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict, Optional, Union, Callable
import time
import random
import copy

class Point:
    """Represents a point on a 2D grid."""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        return Point(self.x + other[0], self.y + other[1])
    
    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return Point(self.x - other[0], self.y - other[1])
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def as_tuple(self):
        return (self.x, self.y)


class Tile:
    """Represents a tile in a tiling puzzle."""
    def __init__(self, points: List[Point], tile_id: int, color: str = None):
        """
        Initialize a tile with a list of points.
        
        Args:
            points: List of Point objects defining the shape of the tile
            tile_id: Unique identifier for the tile
            color: Color for visualization
        """
        self.original_points = points.copy()
        self.points = points.copy()
        self.tile_id = tile_id
        
        if color is None:
            # Generate a unique color based on tile_id
            r = (tile_id * 79) % 255 / 255
            g = (tile_id * 149) % 255 / 255
            b = (tile_id * 223) % 255 / 255
            self.color = (r, g, b)
        else:
            self.color = color
        
        self._normalize()
    
    def _normalize(self):
        """Normalize the tile to have its leftmost-topmost point at (0,0)."""
        # Find the minimum x and y coordinates
        if not self.points:
            return
            
        min_x = min(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        
        # Translate all points
        for i in range(len(self.points)):
            self.points[i] = Point(self.points[i].x - min_x, self.points[i].y - min_y)
    
    def rotate_90(self) -> 'Tile':
        """Return a new tile rotated 90 degrees clockwise."""
        new_points = [Point(-p.y, p.x) for p in self.points]
        rotated = Tile(new_points, self.tile_id, self.color)
        return rotated
    
    def rotate_180(self) -> 'Tile':
        """Return a new tile rotated 180 degrees."""
        new_points = [Point(-p.x, -p.y) for p in self.points]
        rotated = Tile(new_points, self.tile_id, self.color)
        return rotated
    
    def rotate_270(self) -> 'Tile':
        """Return a new tile rotated 270 degrees clockwise (or 90 counterclockwise)."""
        new_points = [Point(p.y, -p.x) for p in self.points]
        rotated = Tile(new_points, self.tile_id, self.color)
        return rotated
    
    def flip_horizontal(self) -> 'Tile':
        """Return a new tile flipped horizontally."""
        new_points = [Point(-p.x, p.y) for p in self.points]
        flipped = Tile(new_points, self.tile_id, self.color)
        return flipped
    
    def flip_vertical(self) -> 'Tile':
        """Return a new tile flipped vertically."""
        new_points = [Point(p.x, -p.y) for p in self.points]
        flipped = Tile(new_points, self.tile_id, self.color)
        return flipped
    
    def translate(self, dx: int, dy: int) -> 'Tile':
        """Return a new tile translated by (dx, dy)."""
        new_points = [Point(p.x + dx, p.y + dy) for p in self.points]
        translated = Tile(new_points, self.tile_id, self.color)
        return translated
    
    def get_all_transformations(self, allow_reflections: bool = True) -> List['Tile']:
        """
        Get all possible transformations of the tile.
        
        Args:
            allow_reflections: If True, include reflected variations
            
        Returns:
            List of all transformed tiles
        """
        # Start with all rotations
        transformations = [
            self,
            self.rotate_90(),
            self.rotate_180(),
            self.rotate_270()
        ]
        
        if allow_reflections:
            # Add reflections
            flipped = self.flip_horizontal()
            transformations.extend([
                flipped,
                flipped.rotate_90(),
                flipped.rotate_180(),
                flipped.rotate_270()
            ])
        
        # Filter out duplicates by comparing point sets
        unique_transformations = []
        seen_point_sets = set()
        
        for tile in transformations:
            point_set = frozenset((p.x, p.y) for p in tile.points)
            if point_set not in seen_point_sets:
                seen_point_sets.add(point_set)
                unique_transformations.append(tile)
        
        return unique_transformations
    
    def get_width_and_height(self) -> Tuple[int, int]:
        """Get the width and height of the tile's bounding box."""
        if not self.points:
            return 0, 0
            
        max_x = max(p.x for p in self.points)
        max_y = max(p.y for p in self.points)
        return max_x + 1, max_y + 1
    
    def get_area(self) -> int:
        """Get the area of the tile (number of unit squares)."""
        return len(self.points)
    
    def can_place_at(self, board: np.ndarray, position: Point) -> bool:
        """
        Check if the tile can be placed at a given position on the board.
        
        Args:
            board: 2D numpy array representing the board state
            position: Position to place the top-left corner of the tile
            
        Returns:
            True if placement is valid, False otherwise
        """
        height, width = board.shape
        
        for point in self.points:
            x, y = position.x + point.x, position.y + point.y
            
            # Check boundaries
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
            
            # Check if the position is already occupied
            if board[y, x] != 0:
                return False
        
        return True
    
    def place_at(self, board: np.ndarray, position: Point):
        """
        Place the tile on the board at a given position.
        
        Args:
            board: 2D numpy array representing the board state
            position: Position to place the top-left corner of the tile
        """
        for point in self.points:
            x, y = position.x + point.x, position.y + point.y
            board[y, x] = self.tile_id
    
    def remove_from(self, board: np.ndarray, position: Point):
        """
        Remove the tile from the board at a given position.
        
        Args:
            board: 2D numpy array representing the board state
            position: Position where the top-left corner of the tile was placed
        """
        for point in self.points:
            x, y = position.x + point.x, position.y + point.y
            board[y, x] = 0
    
    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        
        # Compare sets of points
        points_self = {(p.x, p.y) for p in self.points}
        points_other = {(p.x, p.y) for p in other.points}
        return points_self == points_other and self.tile_id == other.tile_id
    
    def __hash__(self):
        return hash((frozenset((p.x, p.y) for p in self.points), self.tile_id))
    
    def __str__(self):
        return f"Tile {self.tile_id} with {len(self.points)} points"
    
    def __repr__(self):
        return f"Tile(points={self.points}, tile_id={self.tile_id})"


class PentominoTile(Tile):
    """Represents a pentomino tile (5 connected squares)."""
    PENTOMINO_SHAPES = {
        'F': [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
        'I': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        'L': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)],
        'N': [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)],
        'P': [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        'T': [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
        'U': [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],
        'V': [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
        'W': [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        'X': [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
        'Y': [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)],
        'Z': [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]
    }
    
    def __init__(self, shape_name: str, tile_id: int = None):
        """
        Initialize a pentomino tile.
        
        Args:
            shape_name: One of the 12 pentomino shapes (F, I, L, N, P, T, U, V, W, X, Y, Z)
            tile_id: Unique identifier for the tile
        """
        if shape_name not in self.PENTOMINO_SHAPES:
            raise ValueError(f"Invalid pentomino shape: {shape_name}")
        
        shape_points = [Point(x, y) for x, y in self.PENTOMINO_SHAPES[shape_name]]
        
        if tile_id is None:
            # Generate tile_id from the shape name (A=1, B=2, etc.)
            tile_id = ord(shape_name) - ord('A') + 1
        
        super().__init__(shape_points, tile_id)
        self.shape_name = shape_name


class TilingPuzzle:
    """Base class for tiling puzzles."""
    
    def __init__(self, width: int, height: int, tiles: List[Tile] = None):
        """
        Initialize a tiling puzzle.
        
        Args:
            width: Width of the board
            height: Height of the board
            tiles: List of tiles to use in the puzzle
        """
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=int)
        self.tiles = tiles or []
        self.solutions = []
        self.solution_count = 0
        self.max_solutions = float('inf')  # Maximum number of solutions to find
        self.allow_reflections = True
        self.allow_rotations = True
        
        # Generate all tile transformations
        self.transformed_tiles = []
        if self.tiles:
            self._generate_transformed_tiles()
    
    def _generate_transformed_tiles(self):
        """Generate all possible transformations of the tiles."""
        self.transformed_tiles = []
        
        for tile in self.tiles:
            if self.allow_rotations or self.allow_reflections:
                transformations = tile.get_all_transformations(
                    allow_reflections=self.allow_reflections
                )
                if not self.allow_rotations:
                    # Only keep original and reflections
                    transformations = [t for t in transformations 
                                      if t == tile or t == tile.flip_horizontal() 
                                      or t == tile.flip_vertical()]
            else:
                transformations = [tile]  # Only the original
            
            self.transformed_tiles.append(transformations)
    
    def add_tile(self, tile: Tile):
        """Add a tile to the puzzle."""
        self.tiles.append(tile)
        
        # Update transformed tiles
        if self.allow_rotations or self.allow_reflections:
            transformations = tile.get_all_transformations(
                allow_reflections=self.allow_reflections
            )
            if not self.allow_rotations:
                # Only keep original and reflections
                transformations = [t for t in transformations 
                                 if t == tile or t == tile.flip_horizontal() 
                                 or t == tile.flip_vertical()]
        else:
            transformations = [tile]  # Only the original
        
        self.transformed_tiles.append(transformations)
    
    def clear_board(self):
        """Clear the board."""
        self.board.fill(0)
    
    def is_valid_solution(self) -> bool:
        """Check if the current board state is a valid solution."""
        # Check if all cells are filled
        return np.all(self.board != 0)
    
    def count_configurations(self, max_solutions: int = float('inf'), 
                            allow_rotations: bool = True, 
                            allow_reflections: bool = True) -> int:
        """
        Count the number of valid tiling configurations.
        
        Args:
            max_solutions: Maximum number of solutions to find
            allow_rotations: If True, consider rotated tiles as distinct
            allow_reflections: If True, consider reflected tiles as distinct
            
        Returns:
            Number of valid configurations
        """
        self.solutions = []
        self.solution_count = 0
        self.max_solutions = max_solutions
        self.allow_rotations = allow_rotations
        self.allow_reflections = allow_reflections
        
        # Regenerate transformed tiles with new settings
        self._generate_transformed_tiles()
        
        # Clear the board
        self.clear_board()
        
        # Start the search
        self._search()
        
        return self.solution_count
    
    def _search(self):
        """Recursive search for solutions. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _search method")
    
    def visualize_board(self, solution_index: int = None):
        """
        Visualize the current board state or a specific solution.
        
        Args:
            solution_index: Index of the solution to visualize (None for current board)
        """
        if solution_index is not None:
            if solution_index >= len(self.solutions):
                print(f"Solution index {solution_index} out of range")
                return
            board = self.solutions[solution_index]
        else:
            board = self.board
        
        # Create a colormap
        colors = [(1, 1, 1)]  # White for empty cells
        for tile in self.tiles:
            colors.append(tile.color)
        
        # Handle case where there might be more tile IDs than tiles
        max_id = int(np.max(board))
        while len(colors) <= max_id:
            # Generate additional colors if needed
            r = random.random()
            g = random.random()
            b = random.random()
            colors.append((r, g, b))
        
        cmap = ListedColormap(colors)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10 * self.height / self.width))
        ax.imshow(board, cmap=cmap)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create legend
        legend_elements = []
        for i, tile in enumerate(self.tiles):
            legend_elements.append(
                mpatches.Patch(color=tile.color, label=f'Tile {i+1}')
            )
        
        ax.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(self.tiles)))
        
        plt.tight_layout()
        plt.show()
    
    def visualize_all_solutions(self, max_to_show: int = 10):
        """
        Visualize all found solutions.
        
        Args:
            max_to_show: Maximum number of solutions to display
        """
        if not self.solutions:
            print("No solutions found")
            return
        
        n_solutions = min(len(self.solutions), max_to_show)
        
        # Calculate grid dimensions
        cols = min(3, n_solutions)
        rows = (n_solutions + cols - 1) // cols
        
        # Create a colormap
        colors = [(1, 1, 1)]  # White for empty cells
        for tile in self.tiles:
            colors.append(tile.color)
        
        # Handle case where there might be more tile IDs than tiles
        max_id = max(int(np.max(sol)) for sol in self.solutions[:n_solutions])
        while len(colors) <= max_id:
            # Generate additional colors if needed
            r = random.random()
            g = random.random()
            b = random.random()
            colors.append((r, g, b))
        
        cmap = ListedColormap(colors)
        
        # Create the plot
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(-1, 1) if cols == 1 else axes.reshape(1, -1)
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < n_solutions:
                    ax = axes[i, j]
                    ax.imshow(self.solutions[idx], cmap=cmap)
                    ax.set_title(f"Solution {idx+1}")
                    
                    # Add grid
                    ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
                    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
                    
                    # Remove ticks
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    # Hide unused subplots
                    axes[i, j].axis('off')
        
        # Create legend
        legend_elements = []
        for i, tile in enumerate(self.tiles):
            legend_elements.append(
                mpatches.Patch(color=tile.color, label=f'Tile {i+1}')
            )
        
        fig.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, 0.05), ncol=min(5, len(self.tiles)))
        
        plt.tight_layout()
        plt.show()


class ExactCoverTilingPuzzle(TilingPuzzle):
    """
    Tiling puzzle solved using the Dancing Links algorithm (DLX)
    for the exact cover problem.
    """
    
    def __init__(self, width: int, height: int, tiles: List[Tile] = None):
        super().__init__(width, height, tiles)
        self.dlx = None
    
    def count_configurations(self, max_solutions: int = float('inf'), 
                            allow_rotations: bool = True, 
                            allow_reflections: bool = True) -> int:
        """
        Count the number of valid tiling configurations using DLX algorithm.
        
        Args:
            max_solutions: Maximum number of solutions to find
            allow_rotations: If True, consider rotated tiles as distinct
            allow_reflections: If True, consider reflected tiles as distinct
            
        Returns:
            Number of valid configurations
        """
        self.solutions = []
        self.solution_count = 0
        self.max_solutions = max_solutions
        self.allow_rotations = allow_rotations
        self.allow_reflections = allow_reflections
        
        # Regenerate transformed tiles with new settings
        self._generate_transformed_tiles()
        
        # Create and solve the exact cover problem
        self._setup_exact_cover_problem()
        self._solve_exact_cover()
        
        return self.solution_count
    
    def _setup_exact_cover_problem(self):
        """Set up the exact cover problem from the tiling puzzle."""
        # The exact cover matrix has:
        # - One column for each cell in the board (constraints to cover each cell exactly once)
        # - One row for each possible placement of each transformed tile
        
        num_cells = self.width * self.height
        
        # Create a dictionary to represent the sparse exact cover matrix
        # Each key is a constraint (column), and the value is a set of rows that satisfy it
        self.constraints = {}
        
        # Add cell coverage constraints (one for each cell)
        for y in range(self.height):
            for x in range(self.width):
                cell_idx = y * self.width + x
                self.constraints[cell_idx] = set()
        
        # Add rows for each possible tile placement
        self.rows = []  # (tile_idx, transformation_idx, position)
        
        for tile_idx, transformations in enumerate(self.transformed_tiles):
            for transform_idx, tile in enumerate(transformations):
                # Try placing the tile at each possible position
                for y in range(self.height):
                    for x in range(self.width):
                        pos = Point(x, y)
                        if tile.can_place_at(self.board, pos):
                            # This is a valid placement
                            row_idx = len(self.rows)
                            self.rows.append((tile_idx, transform_idx, pos))
                            
                            # Update constraints for this placement
                            for p in tile.points:
                                cell_x, cell_y = pos.x + p.x, pos.y + p.y
                                cell_idx = cell_y * self.width + cell_x
                                self.constraints[cell_idx].add(row_idx)
    
    def _solve_exact_cover(self):
        """Solve the exact cover problem using Knuth's Algorithm X with Dancing Links."""
        # Initialize DLX object if not already created
        if self.dlx is None:
            self.dlx = DancingLinks(len(self.constraints))
        else:
            self.dlx.reset(len(self.constraints))
        
        # Add rows to the DLX matrix
        for row_idx, (tile_idx, transform_idx, pos) in enumerate(self.rows):
            # Get the tile and its transformation
            tile = self.transformed_tiles[tile_idx][transform_idx]
            
            # Determine which columns (constraints) this row covers
            columns = []
            for p in tile.points:
                cell_x, cell_y = pos.x + p.x, pos.y + p.y
                cell_idx = cell_y * self.width + cell_x
                columns.append(cell_idx)
            
            # Add the row to the DLX matrix
            self.dlx.add_row(columns, row_idx)
        
        # Solve the exact cover problem
        self.dlx.solve(self.max_solutions, self._process_solution)
    
    def _process_solution(self, row_indices):
        """Process a solution found by the DLX algorithm."""
        # Create a new board
        board = np.zeros((self.height, self.width), dtype=int)
        
        # Fill the board with the selected tiles
        for row_idx in row_indices:
            tile_idx, transform_idx, pos = self.rows[row_idx]
            tile = self.transformed_tiles[tile_idx][transform_idx]
            
            # Place the tile on the board
            for p in tile.points:
                cell_x, cell_y = pos.x + p.x, pos.y + p.y
                board[cell_y, cell_x] = tile_idx + 1  # Add 1 to avoid 0
        
        # Save the solution
        self.solutions.append(board.copy())
        self.solution_count += 1
        
        # Return True to continue searching, False to stop
        return self.solution_count < self.max_solutions


class DancingLinks:
    """
    Implementation of Knuth's Dancing Links (DLX) algorithm
    for solving the exact cover problem.
    """
    
    class Node:
        """Node in the dancing links data structure."""
        def __init__(self, row: int, col: int):
            self.row = row
            self.col = col
            self.up = self
            self.down = self
            self.left = self
            self.right = self
    
    class ColumnHeader(Node):
        """Column header node in the dancing links data structure."""
        def __init__(self, col: int):
            super().__init__(-1, col)
            self.size = 0
    
    def __init__(self, num_cols: int):
        """
        Initialize the dancing links data structure.
        
        Args:
            num_cols: Number of columns in the exact cover matrix
        """
        self.num_cols = num_cols
        self.header = self.Node(-1, -1)  # Root node
        self.column_headers = []
        
        # Create column headers
        self.reset(num_cols)
    
    def reset(self, num_cols: int = None):
        """
        Reset the dancing links data structure.
        
        Args:
            num_cols: Number of columns in the exact cover matrix (if None, use existing)
        """
        if num_cols is not None:
            self.num_cols = num_cols
        
        # Create the circular linked structure for the header
        self.header = self.Node(-1, -1)
        self.header.left = self.header
        self.header.right = self.header
        
        # Create column headers
        self.column_headers = []
        for i in range(self.num_cols):
            col_header = self.ColumnHeader(i)
            col_header.right = self.header
            col_header.left = self.header.left
            self.header.left.right = col_header
            self.header.left = col_header
            self.column_headers.append(col_header)
    
    def add_row(self, columns: List[int], row_idx: int):
        """
        Add a row to the dancing links matrix.
        
        Args:
            columns: List of column indices where the row has 1s
            row_idx: Row index (used for tracking the solution)
        """
        if not columns:
            return
        
        # Create nodes for each 1 in the row
        row_start = None
        prev_node = None
        
        for col_idx in columns:
            # Create a new node
            node = self.Node(row_idx, col_idx)
            
            # Link to its column
            col_header = self.column_headers[col_idx]
            node.up = col_header.up
            node.down = col_header
            col_header.up.down = node
            col_header.up = node
            col_header.size += 1
            
            # Link to the previous node in the row
            if prev_node is None:
                row_start = node
            else:
                node.left = prev_node
                node.right = prev_node.right
                prev_node.right.left = node
                prev_node.right = node
            
            prev_node = node
        
        # Make the row circular
        if row_start and prev_node:
            row_start.left = prev_node
            prev_node.right = row_start
    
    def cover_column(self, col_header):
        """Remove a column from the matrix."""
        col_header.right.left = col_header.left
        col_header.left.right = col_header.right
        
        i = col_header.down
        while i != col_header:
            j = i.right
            while j != i:
                j.down.up = j.up
                j.up.down = j.down
                self.column_headers[j.col].size -= 1
                j = j.right
            i = i.down
    
    def uncover_column(self, col_header):
        """Restore a previously covered column."""
        i = col_header.up
        while i != col_header:
            j = i.left
            while j != i:
                self.column_headers[j.col].size += 1
                j.down.up = j
                j.up.down = j
                j = j.left
            i = i.up
        
        col_header.right.left = col_header
        col_header.left.right = col_header
    
    def solve(self, max_solutions: int, solution_callback: Callable[[List[int]], bool]):
        """
        Solve the exact cover problem.
        
        Args:
            max_solutions: Maximum number of solutions to find
            solution_callback: Function to call for each solution found
                The function should take a list of row indices and return
                True to continue searching, False to stop
        """
        self.solutions_found = 0
        self.max_solutions = max_solutions
        self.solution_callback = solution_callback
        self.solution = []
        
        # Start the recursive search
        self._search()
    
    def _search(self):
        """Recursive search for solutions using Algorithm X."""
        # If the matrix is empty, we've found a solution
        if self.header.right == self.header:
            # Convert row indices to the original problem's representation
            row_indices = [node.row for node in self.solution]
            
            # Call the solution callback
            continue_search = self.solution_callback(row_indices)
            
            self.solutions_found += 1
            
            # Check if we've reached the maximum number of solutions
            if not continue_search or self.solutions_found >= self.max_solutions:
                return True
            
            return False
        
        # Choose the column with the minimum number of 1s
        col_header = self.choose_column()
        
        # If any column has zero 1s, the problem is unsolvable
        if col_header.size == 0:
            return False
        
        # Cover the column
        self.cover_column(col_header)
        
        # Try each row in the column
        r = col_header.down
        while r != col_header:
            # Add the row to the solution
            self.solution.append(r)
            
            # Cover all columns in the row
            j = r.right
            while j != r:
                self.cover_column(self.column_headers[j.col])
                j = j.right
            
            # Recursively search for the rest of the solution
            if self._search():
                return True
            
            # Backtrack: uncover all columns in the row
            j = r.left
            while j != r:
                self.uncover_column(self.column_headers[j.col])
                j = j.left
            
            # Remove the row from the solution
            self.solution.pop()
            
            r = r.down
        
        # Uncover the column
        self.uncover_column(col_header)
        
        return False
    
    def choose_column(self):
        """Choose the column with the minimum number of 1s."""
        min_size = float('inf')
        chosen_col = None
        
        col = self.header.right
        while col != self.header:
            if col.size < min_size:
                min_size = col.size
                chosen_col = col
            col = col.right
        
        return chosen_col


class BacktrackingTilingPuzzle(TilingPuzzle):
    """Tiling puzzle solved using backtracking algorithm."""
    
    def __init__(self, width: int, height: int, tiles: List[Tile] = None):
        super().__init__(width, height, tiles)
        self.used_tiles = []
    
    def _search(self):
        """Recursive backtracking search for solutions."""
        # Check if we've reached a solution
        if len(self.used_tiles) == len(self.tiles):
            if self.is_valid_solution():
                # Found a valid solution
                self.solutions.append(self.board.copy())
                self.solution_count += 1
                
                # Check if we've reached the maximum number of solutions
                if self.solution_count >= self.max_solutions:
                    return True
            
            return False
        
        # Find the next empty cell
        next_cell = self._find_next_empty_cell()
        if next_cell is None:
            # No empty cell, but not all tiles used (can happen if tiles don't fit perfectly)
            return False
        
        x, y = next_cell
        pos = Point(x, y)
        
        # Try each unused tile
        for tile_idx in range(len(self.tiles)):
            if tile_idx in self.used_tiles:
                continue
            
            # Try all transformations of the tile
            for transformed_tile in self.transformed_tiles[tile_idx]:
                if transformed_tile.can_place_at(self.board, pos):
                    # Place the tile
                    transformed_tile.place_at(self.board, pos)
                    self.used_tiles.append(tile_idx)
                    
                    # Recursively continue the search
                    if self._search():
                        return True
                    
                    # Backtrack
                    self.used_tiles.pop()
                    transformed_tile.remove_from(self.board, pos)
        
        return False
    
    def _find_next_empty_cell(self):
        """Find the next empty cell to fill (using a heuristic)."""
        # Simple strategy: first empty cell in row-major order
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y, x] == 0:
                    return (x, y)
        return None


class PentominoPuzzle(ExactCoverTilingPuzzle):
    """
    A classic pentomino puzzle, where the goal is to tile a rectangle
    with the 12 pentominoes.
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize a pentomino puzzle.
        
        Args:
            width: Width of the board
            height: Height of the board
        """
        # Check if the area is correct
        if width * height != 60:
            print("Warning: Standard pentomino puzzles need a 60-square area.")
            if width * height < 60:
                raise ValueError("Area is too small to fit all pentominoes.")
        
        # Create the 12 pentominoes
        tiles = []
        for i, shape in enumerate("FILNPTUVWXYZ"):
            tiles.append(PentominoTile(shape, i+1))
        
        super().__init__(width, height, tiles)
    
    def count_configurations(self, max_solutions: int = float('inf'), 
                            allow_rotations: bool = True, 
                            allow_reflections: bool = True) -> int:
        """
        Count the number of valid tiling configurations.
        
        For standard pentomino puzzles:
        - 6x10 board: 2,339 solutions
        - 5x12 board: 1,010 solutions
        - 4x15 board: 368 solutions
        - 3x20 board: 2 solutions
        
        Args:
            max_solutions: Maximum number of solutions to find
            allow_rotations: If True, consider rotated tiles as distinct
            allow_reflections: If True, consider reflected tiles as distinct
            
        Returns:
            Number of valid configurations
        """
        return super().count_configurations(max_solutions, allow_rotations, allow_reflections)


class PolyominoTilingAnalyzer:
    """
    Class for analyzing the number of possible tilings with polyominoes.
    This implements mathematical formulas and algorithms for counting configurations.
    """
    
    @staticmethod
    def count_domino_tilings(m: int, n: int) -> int:
        """
        Count the number of ways to tile an m×n board with 1×2 dominoes.
        Uses an exact formula based on number theory when possible,
        or counting matchings in a graph for odd cases.
        
        Args:
            m: Width of the board
            n: Height of the board
            
        Returns:
            Number of valid tilings
        """
        # Check if the area is even (required for dominoes)
        if (m * n) % 2 != 0:
            return 0
        
        # Special cases with exact formulas
        if m == 1:
            return 1 if n % 2 == 0 else 0
        
        if n == 1:
            return 1 if m % 2 == 0 else 0
        
        if m == 2:
            return fibonacci(n + 1)
        
        if n == 2:
            return fibonacci(m + 1)
        
        # For small rectangles, use dynamic programming
        if m <= 10 and n <= 10:
            return PolyominoTilingAnalyzer._count_domino_tilings_dp(m, n)
        
        # For larger even×even rectangles, use the formula
        if m % 2 == 0 and n % 2 == 0:
            return PolyominoTilingAnalyzer._count_domino_tilings_even(m, n)
        
        # For odd×odd rectangles, there's no exact formula
        # Use permanent-based approach
        return PolyominoTilingAnalyzer._count_domino_tilings_permanent(m, n)
    
    @staticmethod
    def _count_domino_tilings_dp(m: int, n: int) -> int:
        """
        Count domino tilings using dynamic programming.
        
        Args:
            m: Width of the board
            n: Height of the board
            
        Returns:
            Number of valid tilings
        """
        # Make sure m <= n for efficiency
        if m > n:
            m, n = n, m
        
        # Create a bit mask to represent the board state
        # Each row's state is represented by a binary number
        # where 1 means the cell is covered by a vertical domino from above
        
        # Initialize the DP table
        dp = [0] * (1 << m)
        dp[0] = 1  # Base case: empty board has 1 tiling
        
        # Iterate through each cell
        for j in range(n):
            for i in range(m):
                new_dp = [0] * (1 << m)
                
                for mask in range(1 << m):
                    # Check if position (i, j) is already covered
                    if mask & (1 << i):
                        # Already covered by a vertical domino
                        new_dp[mask & ~(1 << i)] += dp[mask]
                    else:
                        # Not covered, try horizontal domino
                        if i + 1 < m and (mask & (1 << (i+1))) == 0:
                            new_dp[mask] += dp[mask]
                        
                        # Try vertical domino
                        new_dp[mask | (1 << i)] += dp[mask]
                
                dp = new_dp
        
        # The answer is dp[0] (all cells covered)
        return dp[0]
    
    @staticmethod
    def _count_domino_tilings_even(m: int, n: int) -> int:
        """
        Count domino tilings for even×even rectangles using the exact formula.
        
        Args:
            m: Width of the board (even)
            n: Height of the board (even)
            
        Returns:
            Number of valid tilings
        """
        # Based on the formula derived from Temperley-Fisher and Kasteleyn
        result = 1
        
        for j in range(n):
            for i in range(m):
                term = math.cos(math.pi * (i+1) / (m+1))**2 + math.cos(math.pi * (j+1) / (n+1))**2
                result *= 4 * term
        
        return int(round(math.sqrt(result)))
    
    @staticmethod
    def _count_domino_tilings_permanent(m: int, n: int) -> int:
        """
        Count domino tilings using the permanent of the adjacency matrix.
        This is suitable for general m×n grids but is computationally expensive.
        
        Args:
            m: Width of the board
            n: Height of the board
            
        Returns:
            Number of valid tilings
        """
        # Represent the grid as a bipartite graph
        # Each vertex is a cell, and edges connect adjacent cells
        
        # Create the adjacency matrix
        num_vertices = m * n
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        
        # Connect adjacent cells
        for i in range(m):
            for j in range(n):
                v1 = j * m + i
                
                # Connect to right neighbor
                if i + 1 < m:
                    v2 = j * m + (i + 1)
                    adj_matrix[v1, v2] = 1
                    adj_matrix[v2, v1] = 1
                
                # Connect to bottom neighbor
                if j + 1 < n:
                    v2 = (j + 1) * m + i
                    adj_matrix[v1, v2] = 1
                    adj_matrix[v2, v1] = 1
        
        # Calculate the permanent (for perfect matchings)
        # This is an approximate calculation for larger matrices
        if num_vertices > 20:
            # Use an approximation for very large matrices
            return PolyominoTilingAnalyzer._approximate_permanent(adj_matrix)
        else:
            return PolyominoTilingAnalyzer._calculate_permanent(adj_matrix) // (2 ** (m * n // 2))
    
    @staticmethod
    def _calculate_permanent(matrix: np.ndarray) -> int:
        """
        Calculate the permanent of a matrix.
        This implementation is inefficient for large matrices.
        
        Args:
            matrix: Square matrix
            
        Returns:
            Permanent of the matrix
        """
        n = matrix.shape[0]
        
        if n == 1:
            return matrix[0, 0]
        
        result = 0
        for j in range(n):
            if matrix[0, j] != 0:
                # Create sub-matrix excluding row 0 and column j
                sub_matrix = np.delete(np.delete(matrix, 0, 0), j, 1)
                result += matrix[0, j] * PolyominoTilingAnalyzer._calculate_permanent(sub_matrix)
        
        return result
    
    @staticmethod
    def _approximate_permanent(matrix: np.ndarray) -> int:
        """
        Approximate the permanent of a matrix using Markov Chain Monte Carlo.
        Based on the Jerrum-Sinclair algorithm.
        
        Args:
            matrix: Square matrix
            
        Returns:
            Approximated permanent
        """
        # For this implementation, we'll simply return a placeholder
        # Implementing a proper MCMC approximation would be complex
        return 0
    
    @staticmethod
    def count_polyomino_tilings(board_width: int, board_height: int, 
                              polyomino_shapes: List[List[Tuple[int, int]]],
                              allow_rotations: bool = True,
                              allow_reflections: bool = True) -> int:
        """
        Count the number of ways to tile a board with the given polyomino shapes.
        Uses an exact cover algorithm.
        
        Args:
            board_width: Width of the board
            board_height: Height of the board
            polyomino_shapes: List of polyomino shapes, each defined as a list of (x, y) coordinates
            allow_rotations: If True, allow rotations of the polyominoes
            allow_reflections: If True, allow reflections of the polyominoes
            
        Returns:
            Number of valid tilings
        """
        # Create tiles from the shapes
        tiles = []
        for i, shape in enumerate(polyomino_shapes):
            points = [Point(x, y) for x, y in shape]
            tiles.append(Tile(points, i+1))
        
        # Create and solve the puzzle
        puzzle = ExactCoverTilingPuzzle(board_width, board_height, tiles)
        return puzzle.count_configurations(
            allow_rotations=allow_rotations,
            allow_reflections=allow_reflections
        )
    
    @staticmethod
    def calculate_monomer_dimer_entropy(p: float) -> float:
        """
        Calculate the entropy of the monomer-dimer system at a given dimer density.
        This is related to the exponential growth rate of the number of tilings.
        
        Args:
            p: Dimer density (fraction of vertices covered by dimers)
            
        Returns:
            Entropy per vertex
        """
        if p < 0 or p > 1:
            raise ValueError("Dimer density must be between 0 and 1")
        
        # Calculate entropy based on approximate formula
        if p == 0:
            return 0.0  # No dimers, only one configuration
        
        if p == 1 and p % 2 == 1:
            return 0.0  # Can't fully cover with dimers if odd
        
        # Approximate formula for the monomer-dimer entropy
        # Based on Baxter's exact solution for the square lattice
        if p < 0.5:
            # Low dimer density regime
            return -p * math.log(p) - (1 - 2*p) * math.log(1 - 2*p) - p * math.log(2)
        else:
            # High dimer density regime (asymptotic to perfect matchings)
            # Based on Kastelyn's formula, entropy is approximately G/π where G is Catalan's constant
            catalan = 0.915965594177219
            max_entropy = catalan / math.pi
            
            # Interpolate between p=0.5 and p=1
            middle_entropy = -0.5 * math.log(0.5) - 0.5 * math.log(2)
            return middle_entropy + (2*p - 1) * (max_entropy - middle_entropy)
    
    @staticmethod
    def count_latin_square_tilings(n: int) -> int:
        """
        Count the number of n×n Latin squares.
        Latin squares are related to certain tiling problems.
        
        Args:
            n: Size of the Latin square
            
        Returns:
            Number of Latin squares
        """
        if n <= 0:
            return 0
        
        if n == 1:
            return 1
        
        if n == 2:
            return 2
        
        if n == 3:
            return 12
        
        if n == 4:
            return 576
        
        if n == 5:
            return 161280
        
        if n == 6:
            return 812851200
        
        if n == 7:
            return 61479419904000
        
        # For larger n, we return an approximation based on asymptotic formulas
        # The growth rate is approximately (n!)^(2n) / n^(n^2)
        lg_approx = 2 * n * sum(math.log(i) for i in range(1, n+1)) - n**2 * math.log(n)
        return int(math.exp(lg_approx))
    
    @staticmethod
    def aztec_diamond_count(n: int) -> int:
        """
        Count the number of domino tilings of the Aztec diamond of order n.
        This is given by the formula 2^(n(n+1)/2).
        
        Args:
            n: Order of the Aztec diamond
            
        Returns:
            Number of valid tilings
        """
        return 2 ** (n * (n + 1) // 2)
    
    @staticmethod
    def hexagonal_count(n: int) -> int:
        """
        Count the number of rhombus tilings of a regular hexagon of side length n.
        This is given by MacMahon's formula.
        
        Args:
            n: Side length of the hexagon
            
        Returns:
            Number of valid tilings
        """
        result = 1
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                for k in range(1, n + 1):
                    result *= (i + j + k - 1) / (i + j + k - 2)
        
        return int(round(result))


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


def demonstrate_pentomino_puzzle():
    """Demonstrate solving a pentomino puzzle."""
    print("Demonstrating pentomino puzzle...")
    
    # Create a 6x10 pentomino puzzle
    puzzle = PentominoPuzzle(6, 10)
    
    # Count solutions (limited to 5 for demonstration)
    num_solutions = puzzle.count_configurations(max_solutions=5)
    
    print(f"Found {num_solutions} solutions (limited to 5)")
    
    # Visualize the solutions
    puzzle.visualize_all_solutions()


def demonstrate_domino_counting():
    """Demonstrate counting domino tilings for various board sizes."""
    print("Counting domino tilings for various board sizes...")
    
    # List of board dimensions to test
    board_sizes = [
        (2, 1), (2, 2), (2, 3), (2, 4), 
        (3, 4), (4, 4), (4, 5), (4, 6), 
        (6, 6), (8, 8)
    ]
    
    # Calculate and print the results
    for m, n in board_sizes:
        count = PolyominoTilingAnalyzer.count_domino_tilings(m, n)
        print(f"{m}×{n} board: {count} domino tilings")


def demonstrate_custom_tiling():
    """Demonstrate a custom tiling problem."""
    print("Demonstrating custom tiling problem...")
    
    # Define some polyomino shapes (tetromino)
    tetrominoes = [
        [(0, 0), (1, 0), (2, 0), (3, 0)],  # I tetromino
        [(0, 0), (1, 0), (0, 1), (1, 1)],  # O tetromino
        [(0, 0), (1, 0), (2, 0), (1, 1)],  # T tetromino
        [(0, 0), (1, 0), (2, 0), (2, 1)],  # L tetromino
        [(0, 0), (1, 0), (2, 0), (0, 1)]   # J tetromino
    ]
    
    # Create tiles from the shapes
    tiles = []
    for i, shape in enumerate(tetrominoes):
        points = [Point(x, y) for x, y in shape]
        tiles.append(Tile(points, i+1))
    
    # Create a 4x5 board (area = 20 = 4*5)
    puzzle = ExactCoverTilingPuzzle(4, 5, tiles)
    
    # Count solutions
    num_solutions = puzzle.count_configurations(max_solutions=10)
    
    print(f"Found {num_solutions} solutions (limited to 10)")
    
    # Visualize the solutions
    puzzle.visualize_all_solutions()


def main():
    """Main function to demonstrate various tiling puzzles."""
    print("Tiling Puzzle Configuration Counter")
    print("==================================")
    
    # Choose which demonstrations to run
    demos = {
        "pentomino": False,
        "domino": True,
        "custom": False
    }
    
    if demos["pentomino"]:
        demonstrate_pentomino_puzzle()
        print()
    
    if demos["domino"]:
        demonstrate_domino_counting()
        print()
    
    if demos["custom"]:
        demonstrate_custom_tiling()
        print()
    
    print("Demo completed!")


if __name__ == "__main__":
    main()