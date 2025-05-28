import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import sympy as sp
from scipy.optimize import minimize
from IPython.display import display, Math

class CubeDuplication:
    """
    A class to explore the ancient problem of duplicating the cube.
    
    The Delian problem asks: Given a cube, construct a new cube with exactly
    twice the volume using only straightedge and compass.
    """
    
    def __init__(self):
        """Initialize with the exact and approximate values of cube root of 2."""
        # The problem reduces to finding the cube root of 2
        self.exact_solution = 2**(1/3)
        self.approximate_solution = self.approximate_cube_root_of_2(precision=20)
        self.classical_solutions = {}
        
    @staticmethod
    def approximate_cube_root_of_2(precision=10):
        """Compute the cube root of 2 to a specified precision using various methods."""
        # Method 1: Using numpy's built-in cube root
        numpy_result = np.cbrt(2)
        
        # Method 2: Using binary search
        def binary_search_cbrt(n, precision):
            low, high = 1.0, 2.0  # Cube root of 2 is between 1 and 2
            
            for _ in range(precision):
                mid = (low + high) / 2
                if mid**3 < n:
                    low = mid
                else:
                    high = mid
                    
            return (low + high) / 2
        
        binary_result = binary_search_cbrt(2, precision*10)
        
        # Method 3: Using Newton's method
        def newton_cbrt(n, precision):
            x = 1.0  # Initial guess
            epsilon = 10**(-precision)
            
            while True:
                x_new = x - (x**3 - n) / (3 * x**2)
                if abs(x_new - x) < epsilon:
                    return x_new
                x = x_new
                
        newton_result = newton_cbrt(2, precision)
        
        # Return all results for comparison
        return {
            'numpy': numpy_result,
            'binary_search': binary_result,
            'newton': newton_result,
            'exact': 2**(1/3)
        }
        
    def compare_approximations(self):
        """Compare different approximation methods and their errors."""
        results = self.approximate_solution
        exact = results['exact']
        
        print("Approximations of the cube root of 2:")
        print("-" * 40)
        print(f"Exact value: {exact:.20f}")
        print("-" * 40)
        
        for method, value in results.items():
            if method != 'exact':
                error = abs(value - exact)
                rel_error = error / exact
                print(f"{method.capitalize():13}: {value:.20f}")
                print(f"{'Absolute error':13}: {error:.20e}")
                print(f"{'Relative error':13}: {rel_error:.20e}")
                print("-" * 40)
                
    def plot_cubes(self):
        """Plot the original cube and the doubled cube for visual comparison."""
        fig = plt.figure(figsize=(12, 6))
        
        # First subplot for the original cube
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_cube(ax1, 1, 'Original Cube (Volume = 1)')
        
        # Second subplot for the doubled cube
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_cube(ax2, self.exact_solution, 'Doubled Cube (Volume = 2)')
        
        plt.tight_layout()
        return fig
    
    def _plot_cube(self, ax, side_length, title):
        """Helper method to plot a cube with a given side length."""
        # Define the 8 vertices of the cube
        vertices = np.array([
            [0, 0, 0],
            [side_length, 0, 0],
            [side_length, side_length, 0],
            [0, side_length, 0],
            [0, 0, side_length],
            [side_length, 0, side_length],
            [side_length, side_length, side_length],
            [0, side_length, side_length]
        ])
        
        # Define the 6 faces using indices of vertices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # Left face
        ]
        
        # Create the 3D polygons
        cube = Poly3DCollection(faces, alpha=0.25, linewidths=1, edgecolor='k')
        
        # Add cube to the plot
        ax.add_collection3d(cube)
        
        # Plot the edges
        for i, j in [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]:
            ax.plot3D(*zip(vertices[i], vertices[j]), color='blue')
        
        # Plot the vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r', s=50)
        
        # Label key vertices
        ax.text(0, 0, 0, "O", fontsize=12, color='black')
        ax.text(side_length, 0, 0, "A", fontsize=12, color='black')
        ax.text(side_length, side_length, 0, "B", fontsize=12, color='black')
        ax.text(0, 0, side_length, "C", fontsize=12, color='black')
        
        # Set plot limits and labels
        margin = 0.2
        ax.set_xlim(-margin, side_length + margin)
        ax.set_ylim(-margin, side_length + margin)
        ax.set_zlim(-margin, side_length + margin)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add dimension labels
        midpoint = side_length / 2
        ax.text(midpoint, -0.1, -0.1, f"Length = {side_length:.6f}", color='blue')
        
        # Calculate volume
        volume = side_length**3
        ax.set_title(f"{title}\nVolume = {volume:.6f}")
        
        return ax
        
    def archytas_solution(self):
        """
        Implement Archytas' solution to the cube duplication problem.
        His solution involves the intersection of a cylinder, torus, and cone.
        """
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Original unit cube dimensions
        original_side = 1.0
        
        # Desired cube root of 2
        cbrt2 = self.exact_solution
        
        # Draw the original and new cubes
        cubes = []
        colors = ['blue', 'red']
        alphas = [0.1, 0.05]
        for i, side in enumerate([original_side, cbrt2]):
            # Create the cube vertices
            vertices = np.array([
                [0, 0, 0],
                [side, 0, 0],
                [side, side, 0],
                [0, side, 0],
                [0, 0, side],
                [side, 0, side],
                [side, side, side],
                [0, side, side]
            ])
            
            # Create the cube faces
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[3], vertices[7], vertices[4]]
            ]
            
            # Create cube and add to plot
            cube = Poly3DCollection(faces, alpha=alphas[i], linewidths=1, edgecolor=colors[i])
            cubes.append(cube)
            ax.add_collection3d(cube)
            
        # Archytas' construction
        # The construction involves the intersection of:
        # 1. A cylinder based on the circle in the xy-plane
        # 2. A torus obtained by rotating this circle
        # 3. A cone with specific parameters
        
        # Parameters for Archytas' construction
        r = 1.0  # Radius of the circle and cylinder
        
        # Generate the cylinder (based on the circle in xy-plane)
        theta = np.linspace(0, 2*np.pi, 100)
        z = np.linspace(-1, 2, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_cyl = r * np.cos(theta_grid)
        y_cyl = r * np.sin(theta_grid)
        z_cyl = z_grid
        
        # Plot the cylinder (partially transparent)
        ax.plot_surface(x_cyl, y_cyl, z_cyl, color='green', alpha=0.1)
        
        # Generate the semicircle that will be rotated to form the torus
        phi = np.linspace(0, np.pi, 100)
        x_circle = r * np.cos(phi)
        z_circle = r * np.sin(phi)
        
        # Plot the generating semicircle
        ax.plot(x_circle, np.zeros_like(x_circle), z_circle, 'g-', linewidth=2)
        
        # Generate points on the torus (simplified)
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2*np.pi, 40)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        
        # Torus parametric equations (simplified for illustration)
        x_torus = r * np.cos(phi_grid) * np.cos(theta_grid)
        y_torus = r * np.cos(phi_grid) * np.sin(theta_grid)
        z_torus = r * np.sin(phi_grid)
        
        # Plot key points on the torus (not the entire surface for clarity)
        ax.scatter(x_torus.flatten()[::40], y_torus.flatten()[::40], z_torus.flatten()[::40], 
                   color='green', alpha=0.3, s=10)
        
        # Plot the cone (line from origin to a key circle on the torus)
        theta_key = np.linspace(0, 2*np.pi, 100)
        x_cone_base = r * np.cos(theta_key)
        y_cone_base = r * np.sin(theta_key)
        z_cone_base = np.zeros_like(theta_key)
        
        # Draw a simplified representation of the cone
        for i in range(0, len(theta_key), 10):
            ax.plot([0, x_cone_base[i]], [0, y_cone_base[i]], [0, z_cone_base[i]], 'y-', alpha=0.3)
            
        # Mark the solution point - the intersection of all three surfaces
        # This occurs at coordinates related to the cube root of 2
        ax.scatter([cbrt2], [0], [0], color='red', s=100, marker='*')
        ax.text(cbrt2, 0, 0, f"  Solution: ∛2 ≈ {cbrt2:.6f}", color='red', fontsize=12)
        
        # Set plot limits
        ax.set_xlim(-0.2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-0.5, 1.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Archytas' Solution to the Cube Duplication Problem")
        
        plt.tight_layout()
        self.classical_solutions['archytas'] = fig
        return fig
    
    def menaechmus_solution(self):
        """
        Implement Menaechmus' solution using intersecting conic sections.
        His solution involves the intersection of a parabola and a hyperbola.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Original unit cube
        original_side = 1.0
        
        # Desired cube root of 2
        cbrt2 = self.exact_solution
        
        # Draw the original and new cubes in 2D (as squares)
        squares = []
        for side in [original_side, cbrt2]:
            square = plt.Rectangle((0, 0), side, side, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(square)
            squares.append(square)
        
        # Menaechmus' construction
        # The construction involves the intersection of:
        # 1. A parabola y = x²
        # 2. A hyperbola xy = 2
        
        # Generate the parabola y = x²
        x_parabola = np.linspace(0, 2, 1000)
        y_parabola = x_parabola**2
        
        # Generate the hyperbola xy = 2
        x_hyperbola = np.linspace(0.1, 3, 1000)
        y_hyperbola = 2 / x_hyperbola
        
        # Plot the curves
        ax.plot(x_parabola, y_parabola, 'r-', linewidth=2, label='Parabola: y = x²')
        ax.plot(x_hyperbola, y_hyperbola, 'g-', linewidth=2, label='Hyperbola: xy = 2')
        
        # Mark the intersection point - this is the solution
        # At the intersection: x³ = 2, so x = ∛2
        ax.scatter([cbrt2], [cbrt2**2], color='purple', s=100, marker='*')
        ax.text(cbrt2, cbrt2**2, f"  Solution: ({cbrt2:.4f}, {cbrt2**2:.4f})", 
                color='purple', fontsize=12)
        
        # Add labels for the squares
        ax.text(original_side/2, -0.1, "Original Cube\n(side = 1)", 
                ha='center', va='top', color='blue')
        ax.text(cbrt2/2, -0.3, f"Doubled Cube\n(side = ∛2 ≈ {cbrt2:.4f})", 
                ha='center', va='top', color='red')
        
        # Set plot limits and labels
        margin = 0.5
        ax.set_xlim(-margin, 3)
        ax.set_ylim(-margin, 3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title("Menaechmus' Solution to the Cube Duplication Problem")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        self.classical_solutions['menaechmus'] = fig
        return fig
    
    def eratosthenes_solution(self):
        """
        Implement Eratosthenes' solution using the method of proportions and a 
        mechanical device known as the mesolabe.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Original unit cube
        original_side = 1.0
        
        # Desired cube root of 2
        cbrt2 = self.exact_solution
        
        # Draw the original and new cubes in 2D (as squares)
        for side in [original_side, cbrt2]:
            square = plt.Rectangle((0, 0), side, side, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(square)
        
        # Eratosthenes' solution involves finding two mean proportionals
        # If a/x = x/y = y/2a, then y is the side length of a cube with twice the volume
        
        # Set up the proportion: 1/x = x/y = y/2
        # The first mean proportional is x = ∛2
        # The second mean proportional is y = ∛4 = 2^(2/3)
        x = cbrt2
        y = 2**(2/3)
        
        # Draw the line representing the proportion
        ax.plot([0, 1, x, y, 2], [0, 0, 0, 0, 0], 'ko-', markersize=8)
        
        # Add labels for the points
        ax.text(0, 0.1, "O", fontsize=12)
        ax.text(1, 0.1, "A (1)", fontsize=12)
        ax.text(x, 0.1, f"X (∛2 ≈ {x:.4f})", fontsize=12)
        ax.text(y, 0.1, f"Y (∛4 ≈ {y:.4f})", fontsize=12)
        ax.text(2, 0.1, "B (2)", fontsize=12)
        
        # Illustrate the proportion
        prop_height = 1.5
        
        # Draw projecting lines
        ax.plot([1, x], [0, prop_height], 'r--')
        ax.plot([x, y], [0, prop_height], 'g--')
        ax.plot([y, 2], [0, prop_height], 'b--')
        
        # Draw the proportion lines
        ax.plot([0, 2], [prop_height, prop_height], 'k-', linewidth=1)
        
        # Add labels for the squares
        ax.text(original_side/2, -0.2, "Original Cube\n(side = 1)", 
                ha='center', va='top', color='blue')
        ax.text(cbrt2/2, -0.4, f"Doubled Cube\n(side = ∛2 ≈ {cbrt2:.4f})", 
                ha='center', va='top', color='red')
        
        # Add a text explanation of the proportion
        explanation = (
            "Eratosthenes' solution finds two mean proportionals:\n"
            "If 1/x = x/y = y/2, then x = ∛2 and y = ∛4\n"
            "The cube with side length x has twice the volume of the unit cube."
        )
        ax.text(1, 2, explanation, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Set plot limits and labels
        margin = 0.5
        ax.set_xlim(-margin, 2.5)
        ax.set_ylim(-0.5, 3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title("Eratosthenes' Solution to the Cube Duplication Problem")
        ax.grid(True)
        
        plt.tight_layout()
        self.classical_solutions['eratosthenes'] = fig
        return fig
    
    def demonstrate_impossibility(self):
        """
        Demonstrate why duplicating the cube is impossible with straightedge and compass.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Original unit cube
        original_side = 1.0
        
        # Desired cube root of 2
        cbrt2 = self.exact_solution
        
        # Draw the number line
        ax.plot([-0.5, 3], [0, 0], 'k-', linewidth=1)
        
        # Mark key points
        ax.scatter([0, 1, cbrt2, 2], [0, 0, 0, 0], c=['black', 'blue', 'red', 'blue'], s=[50, 50, 100, 50])
        
        # Add labels
        ax.text(0, -0.1, "0", ha='center', va='top', fontsize=12)
        ax.text(1, -0.1, "1", ha='center', va='top', fontsize=12)
        ax.text(cbrt2, -0.1, f"∛2 ≈ {cbrt2:.6f}", ha='center', va='top', fontsize=12, color='red')
        ax.text(2, -0.1, "2", ha='center', va='top', fontsize=12)
        
        # Add explanation
        explanation = (
            "THE IMPOSSIBILITY OF STRAIGHTEDGE AND COMPASS CONSTRUCTION\n\n"
            "A straightedge and compass construction can only create:\n"
            "1. Points from the intersection of lines and circles\n"
            "2. Lines from two points\n"
            "3. Circles from a center and a point\n\n"
            "Algebraically, this means we can only construct numbers by:\n"
            "- Addition, subtraction, multiplication, division\n"
            "- Taking square roots\n\n"
            "The cube root of 2 (∛2) cannot be expressed using only these operations,\n"
            "as proven by Pierre Wantzel in 1837 using Galois theory.\n\n"
            "∛2 is the root of the polynomial x³ - 2 = 0, which is irreducible over ℚ\n"
            "and has degree 3, not a power of 2."
        )
        ax.text(1.5, 1.0, explanation, fontsize=12, bbox=dict(facecolor='white', alpha=0.9),
                ha='center', va='center')
        
        # Set plot limits and labels
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-0.5, 2.5)
        
        ax.set_title("Why Duplicating the Cube is Impossible with Straightedge and Compass")
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def neusis_construction(self):
        """
        Implement a neusis construction (marked ruler) solution to the cube duplication problem.
        This method was described by Nicomedes.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Original unit cube
        original_side = 1.0
        
        # Desired cube root of 2
        cbrt2 = self.exact_solution
        
        # Draw the coordinate system
        ax.plot([-1, 3], [0, 0], 'k-', linewidth=1)  # x-axis
        ax.plot([0, 0], [-1, 3], 'k-', linewidth=1)  # y-axis
        
        # Draw a square with two units of area
        rect = plt.Rectangle((0, 0), 1, 2, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        
        # Neusis construction is a marked ruler method
        # We need to find two mean proportionals between 1 and 2
        
        # Draw the construction
        # First, draw the circle with radius 1.5 centered at (0, 1)
        circle = plt.Circle((0, 1), 1.5, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(circle)
        
        # Draw the point at (1, 0)
        ax.scatter([1], [0], color='red', s=80)
        ax.text(1.1, 0, "A", fontsize=12)
        
        # Draw the point at (0, 2)
        ax.scatter([0], [2], color='red', s=80)
        ax.text(0, 2.1, "B", fontsize=12)
        
        # The neusis involves finding a line through the origin that intersects
        # both the circle and the lines x=1 and y=2 in specific ways
        
        # For illustration, draw the solution line
        # This line passes through (0, 0), (cbrt2, cbrt2**2), and (2**(2/3), 2)
        x_line = np.linspace(0, 2, 1000)
        y_line = np.interp([0, cbrt2, 2**(2/3)], [0, cbrt2, 2**(2/3)], 
                            [0, cbrt2**2, 2], x_line)
        ax.plot(x_line, y_line, 'r-', linewidth=2)
        
        # Mark the key points on this line
        ax.scatter([0, cbrt2, 2**(2/3)], [0, cbrt2**2, 2], color='purple', s=[50, 80, 80])
        
        # Add labels for these points
        ax.text(-0.1, -0.1, "O", fontsize=12)
        ax.text(cbrt2+0.1, cbrt2**2, f"({cbrt2:.4f}, {cbrt2**2:.4f})", fontsize=12)
        ax.text(2**(2/3)+0.1, 2, f"({2**(2/3):.4f}, 2)", fontsize=12)
        
        # Explanation
        explanation = (
            "NEUSIS CONSTRUCTION\n\n"
            "This construction uses a marked ruler (neusis) to solve the problem.\n\n"
            "1. Place a circle with radius 1.5 centered at (0, 1)\n"
            "2. Mark points A(1, 0) and B(0, 2)\n"
            "3. The line through O intersects:\n"
            f"   - The x=1 line at ({cbrt2:.4f}, {cbrt2**2:.4f})\n"
            f"   - The y=2 line at ({2**(2/3):.4f}, 2)\n\n"
            f"The x-coordinate {cbrt2:.6f} gives the side length\n"
            "of a cube with twice the volume."
        )
        ax.text(2, 1, explanation, fontsize=12, bbox=dict(facecolor='white', alpha=0.9))
        
        # Set plot limits and labels
        margin = 0.5
        ax.set_xlim(-margin, 3)
        ax.set_ylim(-margin, 3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title("Neusis Construction for the Cube Duplication Problem")
        ax.grid(True)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def origami_solution(self):
        """
        Implement an origami (paper folding) solution to the cube duplication problem.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Original unit cube
        original_side = 1.0
        
        # Desired cube root of 2
        cbrt2 = self.exact_solution
        
        # Draw the coordinate system
        ax.plot([-1, 3], [0, 0], 'k-', linewidth=1)  # x-axis
        ax.plot([0, 0], [-1, 3], 'k-', linewidth=1)  # y-axis
        
        # Draw a unit square representing the paper
        square = plt.Rectangle((0, 0), 2, 2, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(square)
        
        # Draw key points for the origami construction
        ax.scatter([0, 1, 0, 2, cbrt2], [0, 0, 1, 0, cbrt2**2], 
                  color=['black', 'blue', 'blue', 'blue', 'red'], s=[50, 50, 50, 50, 100])
        
        # Label the points
        ax.text(-0.1, -0.1, "O", fontsize=12)
        ax.text(1, -0.2, "A", fontsize=12)
        ax.text(-0.2, 1, "B", fontsize=12)
        ax.text(2, -0.2, "C", fontsize=12)
        ax.text(cbrt2+0.1, cbrt2**2, f"P ({cbrt2:.4f}, {cbrt2**2:.4f})", 
                fontsize=12, color='red')
        
        # Draw the folding lines
        
        # The key fold creates a crease through points O and P
        # where the point P satisfies special conditions
        ax.plot([0, cbrt2], [0, cbrt2**2], 'r--', linewidth=2)
        
        # Draw illustrative fold lines
        fold_x = np.linspace(0, 2, 100)
        fold_y1 = np.zeros_like(fold_x)  # Initial position of bottom edge
        fold_y2 = np.ones_like(fold_x)   # Initial position of horizontal crease
        
        # Draw the initial position of these lines
        ax.plot(fold_x, fold_y1, 'b-', linewidth=1, alpha=0.5)
        ax.plot(fold_x, fold_y2, 'b-', linewidth=1, alpha=0.5)
        
        # Now draw approximate positions after folding
        # These are simplified for illustration
        fold_y1_after = 0.5 * cbrt2**2 * fold_x / cbrt2
        fold_y2_after = 1 - 0.3 * (fold_x - 1)**2
        
        ax.plot(fold_x, fold_y1_after, 'b-', linewidth=1, alpha=0.7)
        ax.plot(fold_x, fold_y2_after, 'b-', linewidth=1, alpha=0.7)
        
        # Add arrows to indicate the folding action
        ax.arrow(1, 0, 0, 0.3, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        ax.arrow(0.5, 1, 0, -0.2, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        
        # Explanation
        explanation = (
            "ORIGAMI SOLUTION\n\n"
            "Paper folding allows for solving cubic equations. For cube duplication:\n\n"
            "1. Start with a square paper OBAC\n"
            "2. Mark points O(0,0), A(1,0), B(0,1), and C(2,0)\n"
            "3. Fold the paper to create a crease where:\n"
            "   - Point A lands on line OB\n"
            "   - Point C lands on line BC\n\n"
            f"The x-coordinate of point P is exactly ∛2 ≈ {cbrt2:.6f},\n"
            "which is the side length of the doubled cube."
        )
        ax.text(1.2, 1.5, explanation, fontsize=12, bbox=dict(facecolor='white', alpha=0.9))
        
        # Set plot limits and labels
        margin = 0.5
        ax.set_xlim(-margin, 3)
        ax.set_ylim(-margin, 3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title("Origami Solution to the Cube Duplication Problem")
        ax.grid(True)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def interactive_demonstration(self):
        """
        Create an interactive demonstration of the cube duplication problem.
        """
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.3)
        
        # Original cube with side length 1
        original_side = 1.0
        
        # Add slider for the side length of the second cube
        ax_slider = plt.axes([0.2, 0.15, 0.65, 0.03])
        side_slider = Slider(
            ax=ax_slider,
            label='Side Length of Second Cube',
            valmin=1.0,
            valmax=2.0,
            valinit=1.25,
            valstep=0.01
        )
        
        # Function to update the plot
        def update(val):
            ax.clear()
            
            # Get current value of slider
            side_length = side_slider.val
            
            # Calculate volume ratio
            volume_ratio = side_length**3
            
            # Draw squares representing the cubes (2D visualization)
            original_square = plt.Rectangle((0, 0), original_side, original_side, 
                                          fill=False, edgecolor='blue', linewidth=2)
            new_square = plt.Rectangle((2, 0), side_length, side_length, 
                                      fill=False, edgecolor='red', linewidth=2)
            
            ax.add_patch(original_square)
            ax.add_patch(new_square)
            
            # Add labels
            ax.text(0.5, -0.2, "Original Cube\nVolume = 1", ha='center', fontsize=10)
            ax.text(2 + side_length/2, -0.2, 
                    f"New Cube\nSide Length = {side_length:.4f}\nVolume = {volume_ratio:.4f}", 
                    ha='center', fontsize=10)
            
            # Mark the target (doubled volume)
            ax.axhline(y=2, color='green', linestyle='--', alpha=0.5)
            ax.text(4.5, 2, "Target Volume = 2", color='green', va='center')
            
            # Mark the current volume
            ax.axhline(y=volume_ratio, color='red', linestyle='--', alpha=0.5)
            
            # Mark the exact solution
            exact_side = self.exact_solution
            ax.axvline(x=2 + exact_side, color='purple', linestyle='--', alpha=0.5)
            ax.text(2 + exact_side, 3.2, f"Exact Solution\nSide Length = ∛2 ≈ {exact_side:.6f}", 
                    color='purple', ha='center')
            
            # Draw the volume curve
            x = np.linspace(1, 2, 100)
            y = x**3
            ax.plot(2 + x, y, 'k-', alpha=0.8)
            
            # Set plot limits and labels
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 3.5)
            
            ax.set_xlabel('Side Length')
            ax.set_ylabel('Volume')
            ax.set_title("Interactive Cube Duplication Demonstration")
            ax.grid(True)
            
            # Add explanation
            if abs(volume_ratio - 2) < 0.01:
                message = "EXCELLENT! You've found the solution.\n"
                message += f"Side length ≈ {side_length:.6f} gives volume ≈ {volume_ratio:.6f}"
                message += "\nExact solution is side length = ∛2 ≈ 1.259921..."
                ax.text(2.5, 1, message, fontsize=12, 
                       bbox=dict(facecolor='green', alpha=0.3))
            else:
                error = abs(volume_ratio - 2)
                message = f"Current error: {error:.6f}\n"
                if volume_ratio < 2:
                    message += "Try increasing the side length."
                else:
                    message += "Try decreasing the side length."
                ax.text(2.5, 1, message, fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.7))
            
            fig.canvas.draw_idle()
        
        # Initial update
        update(side_slider.val)
        
        # Register the update function with the slider
        side_slider.on_changed(update)
        
        # Add reset button
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
        
        def reset(event):
            side_slider.reset()
        
        reset_button.on_clicked(reset)
        
        # Add "exact solution" button
        solution_ax = plt.axes([0.6, 0.025, 0.15, 0.04])
        solution_button = Button(solution_ax, 'Exact Solution', hovercolor='0.975')
        
        def set_exact_solution(event):
            side_slider.set_val(self.exact_solution)
        
        solution_button.on_clicked(set_exact_solution)
        
        return fig, side_slider, reset_button, solution_button
    
    def animate_cube_growth(self):
        """
        Animate the growth of a cube from unit volume to double volume.
        """
        # Create the figure and 3D axes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Original unit cube
        original_side = 1.0
        
        # Final side length (cube root of 2)
        cbrt2 = self.exact_solution
        
        # Initialize the cube with side length 1
        cube = self._plot_cube(ax, original_side, "Cube Volume Evolution")
        
        # Animation update function
        def update(frame):
            # Clear the axes
            ax.clear()
            
            # Calculate current side length (interpolate from 1 to cube root of 2)
            t = frame / 100  # Normalized time from 0 to 1
            current_side = (1-t) * original_side + t * cbrt2
            
            # Calculate current volume
            current_volume = current_side**3
            
            # Plot the cube
            self._plot_cube(ax, current_side, 
                           f"Cube Evolution\nSide Length = {current_side:.6f}\nVolume = {current_volume:.6f}")
            
            # Constant aspect ratio
            margin = 0.2
            ax.set_xlim(-margin, cbrt2 + margin)
            ax.set_ylim(-margin, cbrt2 + margin)
            ax.set_zlim(-margin, cbrt2 + margin)
            
            return ax,
        
        # Create the animation
        anim = animation.FuncAnimation(fig, update, frames=101, interval=50, blit=False)
        
        return anim
    
    def historical_context(self):
        """
        Provide historical context about the cube duplication problem.
        """
        # Create a figure for displaying the historical information
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Remove axis
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "THE DELIAN PROBLEM: HISTORICAL CONTEXT", 
                fontsize=18, weight='bold', ha='center', va='top')
        
        # Historical narrative
        history_text = """
        THE LEGEND OF DELOS
        
        According to legend, around 430 BCE, the citizens of Delos consulted the oracle at Delphi 
        during a plague. They were instructed to double the size of Apollo's cubic altar. The Delians 
        naively doubled each side of the altar, creating a cube with eight times the volume. The plague 
        continued, as they had not properly interpreted the oracle's command.
        
        MATHEMATICAL CHALLENGE
        
        The problem requires constructing a cube with exactly twice the volume of a given cube. 
        If the original cube has a side length of 1, the new cube must have a side length of ∛2 
        (the cube root of 2, approximately 1.25992...). 
        
        This became one of the three famous geometric problems of antiquity, alongside:
        • Squaring the circle (constructing a square with the same area as a given circle)
        • Trisecting an angle (dividing any angle into three equal parts)
        
        All three problems are impossible to solve using only straightedge and compass.
        
        MAJOR HISTORICAL APPROACHES
        
        Many great mathematicians attempted to solve this problem:
        
        1. Hippocrates of Chios (5th century BCE) - Reduced the problem to finding two mean proportionals
        
        2. Archytas of Tarentum (428-347 BCE) - Found a solution using the intersection of a cylinder, 
           torus, and cone in three dimensions
        
        3. Menaechmus (380-320 BCE) - Discovered conic sections (parabola, hyperbola, ellipse) and 
           used them to find a solution
        
        4. Eratosthenes (276-195 BCE) - Invented a mechanical device called the mesolabe to find 
           mean proportionals
        
        5. Nicomedes (280-210 BCE) - Created the conchoid curve and used it in his neusis construction
        
        MODERN RESOLUTION
        
        In 1837, Pierre Wantzel proved the impossibility of duplicating the cube with straightedge 
        and compass alone. This marked the resolution of a problem that had been open for over 
        two millennia.
        
        The proof relies on the fact that the cube root of 2 is not constructible because it 
        cannot be expressed using only square roots, which is what straightedge and compass 
        constructions are limited to algebraically.
        
        LEGACY
        
        The cube duplication problem has had a profound impact on mathematics:
        
        • It led to the discovery of conic sections and new curves
        • It prompted the development of solid geometry
        • It influenced the algebraic theory of equations
        • It illustrates important concepts in Galois theory and constructibility
        
        The problem exemplifies how mathematical investigations, even when they lead to impossibility 
        results, can drive significant advances in mathematical knowledge.
        """
        
        # Add the text
        ax.text(0.5, 0.5, history_text, fontsize=12, ha='center', va='center',
               bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=1'))
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create an instance of the CubeDuplication class
    cube_problem = CubeDuplication()
    
    # Compare different approximation methods
    cube_problem.compare_approximations()
    
    # Plot the original and doubled cubes
    cube_fig = cube_problem.plot_cubes()
    
    # Show Archytas' solution
    archytas_fig = cube_problem.archytas_solution()
    
    # Show Menaechmus' solution
    menaechmus_fig = cube_problem.menaechmus_solution()
    
    # Show Eratosthenes' solution
    eratosthenes_fig = cube_problem.eratosthenes_solution()
    
    # Demonstrate why it's impossible with straightedge and compass
    impossibility_fig = cube_problem.demonstrate_impossibility()
    
    # Show a neusis construction solution
    neusis_fig = cube_problem.neusis_construction()
    
    # Show an origami solution
    origami_fig = cube_problem.origami_solution()
    
    # Create an interactive demonstration
    interactive_fig, slider, reset_button, solution_button = cube_problem.interactive_demonstration()
    
    # Create an animation of the cube growth
    anim = cube_problem.animate_cube_growth()
    
    # Display historical context
    history_fig = cube_problem.historical_context()
    
    plt.show()