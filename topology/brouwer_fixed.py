import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fixed_point, minimize, root
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from scipy.spatial import Delaunay
from tqdm import tqdm
import itertools

class BrouwerFixedPointTheorem:
    """
    A class implementing the Brouwer Fixed Point Theorem.
    
    The Brouwer Fixed Point Theorem states that any continuous function mapping
    a compact convex set to itself has at least one fixed point.
    
    This class provides tools to visualize, compute, and verify fixed points
    for functions in different dimensions.
    """
    
    def __init__(self, dimension=2):
        """
        Initialize the BrouwerFixedPointTheorem class.
        
        Parameters:
        ----------
        dimension : int
            The dimension of the domain (1, 2, or 3)
        """
        self.dimension = dimension
        self.num_points = 1000  # Default number of points for visualization
    
    def find_fixed_point(self, func, domain=None, method='optimization'):
        """
        Find a fixed point of a continuous function.
        
        Parameters:
        ----------
        func : callable
            The continuous function for which to find a fixed point
        domain : tuple or list, optional
            The domain boundaries [(x_min, x_max), (y_min, y_max), ...]
        method : str, optional
            Method to use ('optimization', 'iteration', or 'root_finding')
            
        Returns:
        -------
        ndarray
            The fixed point(s) of the function
        """
        # Default domain based on dimension
        if domain is None:
            if self.dimension == 1:
                domain = [(-1, 1)]
            elif self.dimension == 2:
                domain = [(-1, 1), (-1, 1)]
            elif self.dimension == 3:
                domain = [(-1, 1), (-1, 1), (-1, 1)]
            else:
                domain = [(-1, 1)] * self.dimension
        
        # Initial guess: center of the domain
        initial_guess = np.array([(d[0] + d[1]) / 2 for d in domain])
        
        if method == 'optimization':
            # Define an objective function that measures distance between f(x) and x
            def objective(x):
                # First ensure the point is in the domain (project if necessary)
                x_proj = self.project_to_domain(x, domain)
                fx = func(x_proj)
                # Ensure f(x) is in the domain (though a valid Brouwer function should do this)
                fx_proj = self.project_to_domain(fx, domain)
                # Measure squared distance between x and f(x)
                return np.sum((x_proj - fx_proj) ** 2)
            
            # Use optimization to find a minimum of the objective
            result = minimize(objective, initial_guess, method='L-BFGS-B',
                            bounds=domain)
            
            if result.success:
                fixed_point = result.x
                # Verify it's actually a fixed point
                fx = func(fixed_point)
                if np.allclose(fixed_point, fx, atol=1e-6):
                    return fixed_point
                else:
                    print("Warning: Optimization didn't converge to a true fixed point.")
                    return fixed_point
            else:
                print("Optimization failed, trying root finding...")
                method = 'root_finding'
                
        if method == 'root_finding':
            # Define a function whose root is a fixed point
            def root_func(x):
                x_proj = self.project_to_domain(x, domain)
                return x_proj - func(x_proj)
            
            # Use root finding
            result = root(root_func, initial_guess)
            
            if result.success:
                fixed_point = result.x
                # Project back to domain if needed
                fixed_point = self.project_to_domain(fixed_point, domain)
                return fixed_point
            else:
                print("Root finding failed, trying iteration...")
                method = 'iteration'
                
        if method == 'iteration':
            # Use fixed-point iteration
            def iteration_func(x):
                x_proj = self.project_to_domain(x, domain)
                fx = func(x_proj)
                # Ensure output is in the domain
                return self.project_to_domain(fx, domain)
            
            try:
                # This might not converge for non-contractive maps
                fixed_point = fixed_point(iteration_func, initial_guess, xtol=1e-10, maxiter=1000)
                return fixed_point
            except:
                # Try with a simple custom iteration scheme
                x = initial_guess.copy()
                for _ in range(10000):
                    new_x = iteration_func(x)
                    if np.allclose(new_x, x, atol=1e-10):
                        return new_x
                    x = 0.5 * x + 0.5 * new_x  # Damping to help convergence
                
                # If we get here, we couldn't find a fixed point numerically
                print("Warning: Iteration didn't converge to a fixed point")
                return x
    
    def project_to_domain(self, point, domain):
        """
        Project a point back into the domain if it's outside.
        
        Parameters:
        ----------
        point : ndarray
            The point to project
        domain : list of tuples
            The domain boundaries [(x_min, x_max), (y_min, y_max), ...]
            
        Returns:
        -------
        ndarray
            The projected point
        """
        point = np.array(point)
        projected = np.copy(point)
        
        # For each dimension, clamp to the domain
        for i, (lower, upper) in enumerate(domain):
            if i < len(point):
                projected[i] = min(max(point[i], lower), upper)
        
        # For special cases (disks, balls), project to the shape
        if len(domain) == 2 and all(d == (-1, 1) for d in domain):
            # If domain is a standard 2D square, check if we need to project to a disk
            if self.dimension == 2 and np.sum(point**2) > 1:
                # Project to unit disk
                norm = np.sqrt(np.sum(point**2))
                projected = point / norm
                
        if len(domain) == 3 and all(d == (-1, 1) for d in domain):
            # If domain is a standard 3D cube, check if we need to project to a ball
            if self.dimension == 3 and np.sum(point**2) > 1:
                # Project to unit ball
                norm = np.sqrt(np.sum(point**2))
                projected = point / norm
        
        return projected
    
    def verify_fixed_point(self, func, point, tolerance=1e-6):
        """
        Verify if a point is a fixed point of a function.
        
        Parameters:
        ----------
        func : callable
            The function to check
        point : ndarray
            The potential fixed point
        tolerance : float, optional
            The numerical tolerance for equality
            
        Returns:
        -------
        bool
            True if the point is a fixed point, False otherwise
        """
        fx = func(point)
        return np.allclose(point, fx, atol=tolerance)
    
    def visualize_fixed_points_1d(self, funcs, domain=(-1, 1), num_points=1000):
        """
        Visualize fixed points of functions on a 1D interval.
        
        Parameters:
        ----------
        funcs : list of callables
            The functions to visualize
        domain : tuple, optional
            The domain interval (a, b)
        num_points : int, optional
            Number of points for visualization
            
        Returns:
        -------
        None
        """
        # Create figure
        fig, axs = plt.subplots(len(funcs), 1, figsize=(10, 3*len(funcs)))
        if len(funcs) == 1:
            axs = [axs]  # Make it iterable
        
        # Create x values
        x = np.linspace(domain[0], domain[1], num_points)
        
        for i, (ax, func) in enumerate(zip(axs, funcs)):
            # Compute function values
            y = np.array([func(xi) for xi in x])
            
            # Plot the function
            ax.plot(x, y, 'b-', label='f(x)')
            
            # Plot the identity line
            ax.plot(x, x, 'r--', label='y = x')
            
            # Find intersections (fixed points)
            fixed_points = []
            for j in range(len(x) - 1):
                if (y[j] - x[j]) * (y[j+1] - x[j+1]) <= 0:
                    # Linear interpolation to get a better estimate
                    t = (x[j] - y[j]) / ((y[j+1] - x[j+1]) - (y[j] - x[j]))
                    fixed_point = x[j] + t * (x[j+1] - x[j])
                    fixed_points.append(fixed_point)
            
            # Also use numerical methods to find fixed points
            try:
                numerical_fp = self.find_fixed_point(func, [domain], method='root_finding')
                if isinstance(numerical_fp, np.ndarray) and numerical_fp.size > 0:
                    numerical_fp = numerical_fp[0]  # Extract scalar from array
                    fixed_points.append(numerical_fp)
            except:
                pass
            
            # Plot fixed points
            for fp in fixed_points:
                if domain[0] <= fp <= domain[1] and domain[0] <= func(fp) <= domain[1]:
                    ax.plot([fp], [func(fp)], 'go', markersize=8, label='Fixed Point')
            
            # Set labels and title
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'Function {i+1}')
            
            # Add a legend (only once)
            if i == 0:
                ax.legend()
            
            # Set limits
            ax.set_xlim(domain)
            y_min, y_max = min(np.min(y), domain[0]), max(np.max(y), domain[1])
            margin = 0.1 * (y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_fixed_points_2d(self, funcs, domain=None, is_disk=True, grid_size=20):
        """
        Visualize fixed points of functions on a 2D domain.
        
        Parameters:
        ----------
        funcs : list of callables
            The functions to visualize
        domain : list of tuples, optional
            The domain boundaries [(x_min, x_max), (y_min, y_max)]
        is_disk : bool, optional
            Whether the domain is a disk
        grid_size : int, optional
            Size of the visualization grid
            
        Returns:
        -------
        None
        """
        # Default domain
        if domain is None:
            domain = [(-1, 1), (-1, 1)]
        
        # Create figure
        n_cols = min(len(funcs), 3)
        n_rows = (len(funcs) + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        
        # Make axs indexable for any number of functions
        if len(funcs) == 1:
            axs = np.array([axs])
        axs = np.atleast_2d(axs)
        
        # Create grid
        x = np.linspace(domain[0][0], domain[0][1], grid_size)
        y = np.linspace(domain[1][0], domain[1][1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        # For each function
        for i, func in enumerate(funcs):
            row, col = i // n_cols, i % n_cols
            ax = axs[row, col]
            
            # Draw the domain
            if is_disk:
                circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
                ax.add_artist(circle)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
            else:
                ax.set_xlim(domain[0])
                ax.set_ylim(domain[1])
            
            # Create a grid of points
            points = []
            for xi in range(grid_size):
                for yi in range(grid_size):
                    point = np.array([X[yi, xi], Y[yi, xi]])
                    
                    # Skip points outside the disk if needed
                    if is_disk and np.sum(point**2) > 1:
                        continue
                    
                    points.append(point)
            
            # Apply function to each point
            vectors = []
            for point in points:
                f_point = func(point)
                
                # If f_point is outside the domain, project it back
                if is_disk and np.sum(f_point**2) > 1:
                    norm = np.sqrt(np.sum(f_point**2))
                    f_point = f_point / norm
                
                # Vector from point to f(point)
                vector = f_point - point
                vectors.append(vector)
            
            # Plot the vector field
            points = np.array(points)
            vectors = np.array(vectors)
            
            # Normalize vectors for display (optional)
            vector_norms = np.sqrt(np.sum(vectors**2, axis=1))
            max_norm = max(vector_norms) if vector_norms.size > 0 else 1
            normalized_vectors = vectors / max_norm
            
            # Plot vectors
            ax.quiver(points[:, 0], points[:, 1], 
                     normalized_vectors[:, 0], normalized_vectors[:, 1],
                     vector_norms, cmap='viridis', 
                     angles='xy', scale_units='xy', scale=2)
            
            # Find fixed points numerically
            try:
                fixed_point = self.find_fixed_point(func, domain, method='optimization')
                
                # Verify it's within the domain
                if is_disk and np.sum(fixed_point**2) > 1:
                    fixed_point = fixed_point / np.sqrt(np.sum(fixed_point**2))
                
                # Check if it's a true fixed point
                if self.verify_fixed_point(func, fixed_point):
                    ax.plot(fixed_point[0], fixed_point[1], 'ro', markersize=10, label='Fixed Point')
                    
                    # Add a small circle around the fixed point
                    circle = plt.Circle((fixed_point[0], fixed_point[1]), 0.05, 
                                       fill=True, color='red', alpha=0.3)
                    ax.add_artist(circle)
            except:
                # If numerical methods fail, don't plot a fixed point
                pass
            
            # Set labels and title
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Vector Field for Function {i+1}')
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.3)
            
        # Hide any unused subplots
        for i in range(len(funcs), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_fixed_points_3d(self, funcs, domain=None, is_ball=True, grid_size=10):
        """
        Visualize fixed points of functions on a 3D domain.
        
        Parameters:
        ----------
        funcs : list of callables
            The functions to visualize
        domain : list of tuples, optional
            The domain boundaries [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        is_ball : bool, optional
            Whether the domain is a ball
        grid_size : int, optional
            Size of the visualization grid
            
        Returns:
        -------
        None
        """
        # Default domain
        if domain is None:
            domain = [(-1, 1), (-1, 1), (-1, 1)]
        
        # Create figure
        n_cols = min(len(funcs), 2)
        n_rows = (len(funcs) + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(7*n_cols, 7*n_rows))
        
        # Create grid
        x = np.linspace(domain[0][0], domain[0][1], grid_size)
        y = np.linspace(domain[1][0], domain[1][1], grid_size)
        z = np.linspace(domain[2][0], domain[2][1], grid_size)
        
        # For each function
        for i, func in enumerate(funcs):
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
            
            # Create random points in the domain
            num_points = min(300, grid_size**3)  # Limit for performance
            points = []
            
            if is_ball:
                # Generate points inside the unit ball
                while len(points) < num_points:
                    point = np.random.uniform(-1, 1, 3)
                    if np.sum(point**2) <= 1:
                        points.append(point)
            else:
                # Generate points in the cube
                for _ in range(num_points):
                    point = np.array([
                        np.random.uniform(domain[0][0], domain[0][1]),
                        np.random.uniform(domain[1][0], domain[1][1]),
                        np.random.uniform(domain[2][0], domain[2][1])
                    ])
                    points.append(point)
            
            points = np.array(points)
            
            # Apply function to each point
            vectors = []
            for point in points:
                f_point = func(point)
                
                # If f_point is outside the domain, project it back
                if is_ball and np.sum(f_point**2) > 1:
                    norm = np.sqrt(np.sum(f_point**2))
                    f_point = f_point / norm
                
                vector = f_point - point
                vectors.append(vector)
            
            vectors = np.array(vectors)
            
            # Normalize vectors for display
            vector_norms = np.sqrt(np.sum(vectors**2, axis=1))
            max_norm = max(vector_norms) if vector_norms.size > 0 else 1
            normalized_vectors = vectors / max_norm
            
            # Plot vectors
            ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                     normalized_vectors[:, 0], normalized_vectors[:, 1], normalized_vectors[:, 2],
                     length=0.2, normalize=True, color='b', alpha=0.6)
            
            # Draw a transparent ball if needed
            if is_ball:
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 30)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='gray', alpha=0.1)
            
            # Find fixed points numerically
            try:
                fixed_point = self.find_fixed_point(func, domain, method='optimization')
                
                # Verify it's within the domain
                if is_ball and np.sum(fixed_point**2) > 1:
                    fixed_point = fixed_point / np.sqrt(np.sum(fixed_point**2))
                
                # Check if it's a true fixed point
                if self.verify_fixed_point(func, fixed_point):
                    ax.scatter([fixed_point[0]], [fixed_point[1]], [fixed_point[2]], 
                              color='red', s=100, label='Fixed Point')
            except:
                # If numerical methods fail, don't plot a fixed point
                pass
            
            # Set labels and title
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'Vector Field for Function {i+1}')
            
            # Set limits
            if is_ball:
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_zlim(-1.1, 1.1)
            else:
                ax.set_xlim(domain[0])
                ax.set_ylim(domain[1])
                ax.set_zlim(domain[2])
        
        plt.tight_layout()
        plt.show()
    
    def implement_sperner_lemma_2d(self, grid_size=10, func=None):
        """
        Implement Sperner's lemma in 2D, which is used to prove Brouwer's fixed point theorem.
        
        Parameters:
        ----------
        grid_size : int, optional
            Size of the triangulation grid
        func : callable, optional
            Function to use for labeling (if None, a default function is used)
            
        Returns:
        -------
        None
        """
        # Create a triangulation of a square [0,1]×[0,1]
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        points = np.array([(xi, yi) for xi in x for yi in y])
        
        # Create Delaunay triangulation
        tri = Delaunay(points)
        
        # Default function if none provided
        if func is None:
            # A simple function that rotates points
            func = lambda p: np.array([
                0.5 + 0.3 * (p[0] - 0.5) * np.cos(np.pi/3) - 0.3 * (p[1] - 0.5) * np.sin(np.pi/3),
                0.5 + 0.3 * (p[0] - 0.5) * np.sin(np.pi/3) + 0.3 * (p[1] - 0.5) * np.cos(np.pi/3)
            ])
        
        # Assign a label to each point based on Sperner's conditions
        labels = np.zeros(len(points), dtype=int)
        
        for i, point in enumerate(points):
            # Label vertices of the square
            if np.isclose(point[0], 0) and np.isclose(point[1], 0):
                labels[i] = 0
            elif np.isclose(point[0], 1) and np.isclose(point[1], 0):
                labels[i] = 1
            elif np.isclose(point[0], 1) and np.isclose(point[1], 1):
                labels[i] = 2
            elif np.isclose(point[0], 0) and np.isclose(point[1], 1):
                labels[i] = 0
            # Label boundary edges
            elif np.isclose(point[0], 0):
                labels[i] = 0
            elif np.isclose(point[1], 0):
                labels[i] = np.random.choice([0, 1])
            elif np.isclose(point[0], 1):
                labels[i] = np.random.choice([1, 2])
            elif np.isclose(point[1], 1):
                labels[i] = np.random.choice([0, 2])
            else:
                # Interior points - use the function to determine label
                f_point = func(point)
                
                # Determine label based on which vertex the point is closest to
                distances = [
                    np.sum((f_point - np.array([0, 0]))**2),
                    np.sum((f_point - np.array([1, 0]))**2),
                    np.sum((f_point - np.array([1, 1]))**2)
                ]
                labels[i] = np.argmin(distances)
        
        # Find simplices with all three labels (Sperner simplices)
        sperner_simplices = []
        for simplex in tri.simplices:
            if len(set(labels[simplex])) == 3:
                sperner_simplices.append(simplex)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot triangulation
        ax.triplot(points[:, 0], points[:, 1], tri.simplices, 'k-', alpha=0.2)
        
        # Plot points colored by label
        colors = ['blue', 'red', 'green']
        for i, point in enumerate(points):
            ax.plot(point[0], point[1], 'o', color=colors[labels[i]], markersize=8, alpha=0.7)
        
        # Highlight Sperner simplices
        for simplex in sperner_simplices:
            vertices = points[simplex]
            ax.add_patch(Polygon(vertices, color='yellow', alpha=0.3))
        
        # Try to find a fixed point
        if func is not None:
            fixed_point = self.find_fixed_point(func, [(0, 1), (0, 1)], method='optimization')
            if self.verify_fixed_point(func, fixed_point):
                ax.plot(fixed_point[0], fixed_point[1], 'ro', markersize=12, label='Fixed Point')
                # Draw a line from the point to f(point)
                f_point = func(fixed_point)
                ax.plot([fixed_point[0], f_point[0]], [fixed_point[1], f_point[1]], 'k-', linewidth=2)
        
        # Add legend and labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Sperner\'s Lemma: {len(sperner_simplices)} Sperner simplices found')
        
        # Create a custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Label 0'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Label 1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Label 2'),
            Polygon([(0, 0), (1, 0), (0, 1)], color='yellow', alpha=0.3, label='Sperner Simplex')
        ]
        if func is not None:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                         markersize=15, label='Fixed Point'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Print information
        print(f"Found {len(sperner_simplices)} Sperner simplices.")
        print("According to Sperner's lemma, the number of Sperner simplices must be odd.")
        print("This is a key step in one proof of the Brouwer fixed point theorem.")
    
    def demonstrate_no_retraction(self, domain_type='disk', num_rays=16):
        """
        Demonstrate that there is no retraction from a disk/ball to its boundary,
        which is another way to prove Brouwer's theorem.
        
        Parameters:
        ----------
        domain_type : str, optional
            'disk' or 'square'
        num_rays : int, optional
            Number of rays to draw
            
        Returns:
        -------
        None
        """
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        
        # First plot: Show a continuous vector field that has no zeros on the disk
        ax = axs[0]
        
        # Create a grid
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(x, y)
        
        # Define a vector field with no zeros on the disk
        U = -Y
        V = X
        
        # Normalize the vector field
        norm = np.sqrt(U**2 + V**2)
        U = U / norm
        V = V / norm
        
        # Draw the domain boundary
        if domain_type == 'disk':
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
            ax.add_artist(circle)
        else:
            square = plt.Rectangle((-1, -1), 2, 2, fill=False, color='black', linewidth=2)
            ax.add_artist(square)
        
        # Plot the vector field
        if domain_type == 'disk':
            # Only plot vectors inside the disk
            mask = X**2 + Y**2 <= 1
            ax.quiver(X[mask], Y[mask], U[mask], V[mask], color='blue', alpha=0.8)
        else:
            # Plot all vectors in the square
            ax.quiver(X, Y, U, V, color='blue', alpha=0.8)
        
        # Add title and labels
        ax.set_title('Vector Field with No Zeros')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        # Second plot: Show why there's no retraction
        ax = axs[1]
        
        # Draw domain boundary
        if domain_type == 'disk':
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
            ax.add_artist(circle)
        else:
            square = plt.Rectangle((-1, -1), 2, 2, fill=False, color='black', linewidth=2)
            ax.add_artist(square)
        
        # Draw rays from center to boundary
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            if domain_type == 'disk':
                endpoint = np.array([np.cos(angle), np.sin(angle)])
                ax.plot([0, endpoint[0]], [0, endpoint[1]], 'b-', alpha=0.3)
                
                # Add arrows showing how a retraction would have to map points
                ax.arrow(0.7*endpoint[0], 0.7*endpoint[1], 
                        0.2*endpoint[0], 0.2*endpoint[1], 
                        head_width=0.05, head_length=0.05, fc='green', ec='green', alpha=0.7)
            else:
                # For square, determine where ray intersects boundary
                if abs(np.cos(angle)) > abs(np.sin(angle)):
                    # Hits left or right edge
                    x_end = 1 * np.sign(np.cos(angle))
                    y_end = np.tan(angle) * x_end
                else:
                    # Hits top or bottom edge
                    y_end = 1 * np.sign(np.sin(angle))
                    x_end = y_end / np.tan(angle) if np.tan(angle) != 0 else 0
                
                endpoint = np.array([x_end, y_end])
                ax.plot([0, endpoint[0]], [0, endpoint[1]], 'b-', alpha=0.3)
                
                # Add arrows
                mid_point = 0.7 * endpoint
                ax.arrow(mid_point[0], mid_point[1], 
                        0.2*endpoint[0], 0.2*endpoint[1], 
                        head_width=0.05, head_length=0.05, fc='green', ec='green', alpha=0.7)
        
        # Draw an example "attempted retraction" that must have a fixed point
        if domain_type == 'disk':
            # For disk, show a radial retraction
            for r in [0.25, 0.5, 0.75]:
                circle = plt.Circle((0, 0), r, fill=False, color='red', alpha=0.3, linestyle='--')
                ax.add_artist(circle)
        else:
            # For square, show a similar retraction
            for r in [0.25, 0.5, 0.75]:
                square = plt.Rectangle((-r, -r), 2*r, 2*r, fill=False, 
                                      color='red', alpha=0.3, linestyle='--')
                ax.add_artist(square)
        
        # Add title and labels
        ax.set_title('No Retraction Principle')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        # Add an annotation explaining the concept
        ax.text(0, -1.3, "A retraction must map each boundary point to itself.\n"
               "This creates a 'degree 1' map on the boundary.\n"
               "But a continuous function from disk to boundary must have degree 0.\n"
               "This contradiction proves no retraction exists.",
               ha='center', va='top', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        print("Brouwer's Fixed Point Theorem can be proven by showing that there is no retraction")
        print("from a disk to its boundary. If there was such a retraction r, then the composition")
        print("I∘r (identity composed with retraction) would have a fixed point by Brouwer's theorem.")
        print("But this is impossible by construction, creating a contradiction.")
    
    def demonstrate_winding_number(self, num_points=100):
        """
        Demonstrate the winding number argument for the Brouwer fixed point theorem.
        
        Parameters:
        ----------
        num_points : int, optional
            Number of points along the boundary to use
            
        Returns:
        -------
        None
        """
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        
        # First plot: function with a fixed point
        ax = axs[0]
        
        # Define a function with a fixed point (rotation + contraction)
        def f1(p):
            x, y = p
            # Rotate by 30 degrees and contract by 30%
            theta = np.pi/6
            r = 0.7
            new_x = r * (x*np.cos(theta) - y*np.sin(theta))
            new_y = r * (x*np.sin(theta) + y*np.cos(theta))
            return np.array([new_x, new_y])
        
        # Draw the unit circle
        theta = np.linspace(0, 2*np.pi, num_points)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, 'k-', linewidth=2, label='Unit Circle')
        
        # Map the circle through f1 and draw the image
        boundary_points = np.array([np.cos(theta), np.sin(theta)]).T
        mapped_points = np.array([f1(p) for p in boundary_points])
        ax.plot(mapped_points[:, 0], mapped_points[:, 1], 'r-', linewidth=2, label='Image of Circle')
        
        # Draw some radial lines to visualize the mapping
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            p = np.array([np.cos(angle), np.sin(angle)])
            fp = f1(p)
            ax.plot([0, p[0]], [0, p[1]], 'b-', alpha=0.3)
            ax.plot([0, fp[0]], [0, fp[1]], 'r-', alpha=0.3)
            ax.arrow(p[0], p[1], fp[0]-p[0], fp[1]-p[1], head_width=0.05, 
                    head_length=0.05, fc='black', ec='black', alpha=0.5)
        
        # Find and plot the fixed point
        fixed_point = self.find_fixed_point(f1, [(-1, 1), (-1, 1)], method='optimization')
        if self.verify_fixed_point(f1, fixed_point):
            ax.plot(fixed_point[0], fixed_point[1], 'go', markersize=10, label='Fixed Point')
        
        # Add title and labels
        ax.set_title('Function with a Fixed Point')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.legend()
        
        # Second plot: Attempt to create a function without a fixed point
        ax = axs[1]
        
        # Define a "putative" function without a fixed point (shifting everything)
        def f2(p):
            x, y = p
            return np.array([x + 0.2, y + 0.1])
        
        # Draw the unit circle
        ax.plot(x, y, 'k-', linewidth=2, label='Unit Circle')
        
        # Map the circle through f2
        mapped_points = np.array([f2(p) for p in boundary_points])
        
        # Compute the vector field p - f(p)
        vector_field = boundary_points - mapped_points
        
        # Normalize for display
        norms = np.sqrt(np.sum(vector_field**2, axis=1))
        normalized_field = vector_field / np.max(norms)
        
        # Draw the displacement vectors
        ax.quiver(boundary_points[:, 0], boundary_points[:, 1], 
                 normalized_field[:, 0], normalized_field[:, 1], 
                 color='red', alpha=0.7, label='p - f(p) vectors')
        
        # Trace a path connecting a point to its image
        index = 0  # Start point
        ax.plot([boundary_points[index, 0], mapped_points[index, 0]], 
               [boundary_points[index, 1], mapped_points[index, 1]], 
               'g-', linewidth=2, label='Path from p to f(p)')
        
        # Add title and labels
        ax.set_title('Cannot Have a Function Without a Fixed Point')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.legend()
        
        # Add an annotation explaining the winding number argument
        ax.text(0, -1.4, "If a function f had no fixed points, then the vector field v(p) = p - f(p)\n"
               "would never be zero. We could normalize it to get a non-vanishing vector field\n"
               "on the disk, which is impossible due to the hairy ball theorem for even dimensions.",
               ha='center', va='top', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        print("The Brouwer fixed point theorem can also be proven using degree theory or winding numbers.")
        print("If a function f: D → D had no fixed points, we could define g(p) = p - f(p) / |p - f(p)|.")
        print("This would give a continuous non-vanishing vector field on the disk.")
        print("But when restricted to the boundary, this field would have a non-zero winding number,")
        print("which is impossible for a field that extends continuously to the interior.")


class FixedPointApplications:
    """
    Applications of the Brouwer fixed point theorem in different areas.
    """
    
    @staticmethod
    def game_theory_equilibrium(num_strategies_p1=3, num_strategies_p2=3):
        """
        Demonstrate the application of fixed point theorems in game theory.
        
        The Nash equilibrium in a game can be found using fixed point theorems.
        
        Parameters:
        ----------
        num_strategies_p1, num_strategies_p2 : int, optional
            Number of strategies for each player
            
        Returns:
        -------
        None
        """
        # Create a random game (payoff matrices)
        np.random.seed(42)  # For reproducibility
        payoff_p1 = np.random.rand(num_strategies_p1, num_strategies_p2)
        payoff_p2 = np.random.rand(num_strategies_p1, num_strategies_p2)
        
        # Normalize payoffs to [0, 1]
        payoff_p1 = (payoff_p1 - np.min(payoff_p1)) / (np.max(payoff_p1) - np.min(payoff_p1))
        payoff_p2 = (payoff_p2 - np.min(payoff_p2)) / (np.max(payoff_p2) - np.min(payoff_p2))
        
        # Define the best response function
        def best_response(mixed_strategy, payoff_matrix, player=1):
            if player == 1:
                # Player 1's best response to player 2's strategy
                expected_payoffs = payoff_matrix @ mixed_strategy
            else:
                # Player 2's best response to player 1's strategy
                expected_payoffs = mixed_strategy @ payoff_matrix
                
            # Find best strategy (could be multiple if there are ties)
            best_indices = np.where(expected_payoffs == np.max(expected_payoffs))[0]
            
            # Return a mixed strategy that puts equal weight on all best responses
            best_strat = np.zeros_like(expected_payoffs)
            best_strat[best_indices] = 1.0 / len(best_indices)
            return best_strat
        
        # Define a function that maps a pair of mixed strategies to a new pair
        def strategy_update(strategies):
            # Extract strategies
            p1_strategy = strategies[:num_strategies_p1]
            p2_strategy = strategies[num_strategies_p1:]
            
            # Ensure they are valid probability distributions
            p1_strategy = np.maximum(p1_strategy, 0)
            p2_strategy = np.maximum(p2_strategy, 0)
            
            # Normalize
            if np.sum(p1_strategy) > 0:
                p1_strategy = p1_strategy / np.sum(p1_strategy)
            else:
                p1_strategy = np.ones(num_strategies_p1) / num_strategies_p1
                
            if np.sum(p2_strategy) > 0:
                p2_strategy = p2_strategy / np.sum(p2_strategy)
            else:
                p2_strategy = np.ones(num_strategies_p2) / num_strategies_p2
            
            # Compute best responses
            p1_response = best_response(p2_strategy, payoff_p1, player=1)
            p2_response = best_response(p1_strategy, payoff_p2, player=2)
            
            # Dampen the update for better convergence
            alpha = 0.5  # Learning rate
            p1_new = (1 - alpha) * p1_strategy + alpha * p1_response
            p2_new = (1 - alpha) * p2_strategy + alpha * p2_response
            
            return np.concatenate([p1_new, p2_new])
        
        # Create a BrouwerFixedPointTheorem object
        bfpt = BrouwerFixedPointTheorem(dimension=num_strategies_p1 + num_strategies_p2)
        
        # Find the fixed point (Nash equilibrium)
        # Initial guess: uniform mixed strategies
        initial_guess = np.ones(num_strategies_p1 + num_strategies_p2) / (num_strategies_p1 + num_strategies_p2)
        
        # Domain is the Cartesian product of two simplexes
        # Since we're normalizing inside the function, we can use a hypercube as domain
        domain = [(-0.1, 1.1)] * (num_strategies_p1 + num_strategies_p2)
        
        # Find the fixed point
        fixed_point = bfpt.find_fixed_point(strategy_update, domain, method='iteration')
        
        # Extract the equilibrium strategies
        p1_equilibrium = fixed_point[:num_strategies_p1]
        p2_equilibrium = fixed_point[num_strategies_p1:]
        
        # Normalize again just to be sure
        p1_equilibrium = p1_equilibrium / np.sum(p1_equilibrium)
        p2_equilibrium = p2_equilibrium / np.sum(p2_equilibrium)
        
        # Visualize the game and solution
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot payoff matrices
        ax = axs[0]
        im = ax.imshow(payoff_p1, cmap='viridis')
        ax.set_title('Player 1 Payoff Matrix')
        ax.set_xlabel('Player 2 Strategy')
        ax.set_ylabel('Player 1 Strategy')
        for i in range(num_strategies_p1):
            for j in range(num_strategies_p2):
                ax.text(j, i, f'{payoff_p1[i, j]:.2f}', ha='center', va='center', color='white')
        plt.colorbar(im, ax=ax)
        
        ax = axs[1]
        im = ax.imshow(payoff_p2, cmap='viridis')
        ax.set_title('Player 2 Payoff Matrix')
        ax.set_xlabel('Player 2 Strategy')
        ax.set_ylabel('Player 1 Strategy')
        for i in range(num_strategies_p1):
            for j in range(num_strategies_p2):
                ax.text(j, i, f'{payoff_p2[i, j]:.2f}', ha='center', va='center', color='white')
        plt.colorbar(im, ax=ax)
        
        # Plot equilibrium strategies
        ax = axs[2]
        x1 = np.arange(num_strategies_p1)
        x2 = np.arange(num_strategies_p1, num_strategies_p1 + num_strategies_p2)
        width = 0.35
        
        ax.bar(x1, p1_equilibrium, width, label='Player 1')
        ax.bar(x2, p2_equilibrium, width, label='Player 2')
        
        ax.set_title('Nash Equilibrium Mixed Strategies')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Probability')
        ax.set_xticks(np.arange(num_strategies_p1 + num_strategies_p2))
        ax.set_xticklabels([f'P1-{i+1}' for i in range(num_strategies_p1)] + 
                           [f'P2-{i+1}' for i in range(num_strategies_p2)])
        ax.legend()
        
        # Check if it's a true Nash equilibrium
        p1_best = best_response(p2_equilibrium, payoff_p1, player=1)
        p2_best = best_response(p1_equilibrium, payoff_p2, player=2)
        
        p1_expected = np.dot(p1_equilibrium, payoff_p1 @ p2_equilibrium)
        p1_best_expected = np.dot(p1_best, payoff_p1 @ p2_equilibrium)
        
        p2_expected = np.dot(p1_equilibrium @ payoff_p2, p2_equilibrium)
        p2_best_expected = np.dot(p1_equilibrium @ payoff_p2, p2_best)
        
        is_nash = np.isclose(p1_expected, p1_best_expected) and np.isclose(p2_expected, p2_best_expected)
        
        plt.figtext(0.5, 0.01, f"Is Nash Equilibrium: {is_nash}\n"
                   f"Player 1 expected payoff: {p1_expected:.4f}, best possible: {p1_best_expected:.4f}\n"
                   f"Player 2 expected payoff: {p2_expected:.4f}, best possible: {p2_best_expected:.4f}",
                   ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
        
        plt.tight_layout()
        plt.show()
        
        print("Nash's theorem guarantees that every finite game has a mixed strategy equilibrium.")
        print("This is proven using Brouwer's fixed point theorem or its generalizations.")
        print("The equilibrium is found as a fixed point of the best response function.")
    
    @staticmethod
    def economic_equilibrium():
        """
        Demonstrate the application of fixed point theorems in economic equilibrium.
        
        Returns:
        -------
        None
        """
        # Define a simple exchange economy with 2 goods and 2 consumers
        # Utility functions: Cobb-Douglas U(x, y) = x^a * y^(1-a)
        # Initial endowments: e_i = (e_i1, e_i2)
        
        alpha = [0.3, 0.7]  # Preference parameters
        endowments = np.array([[1.0, 3.0], [3.0, 1.0]])  # Initial endowments
        
        # Total resources
        total_resources = np.sum(endowments, axis=0)
        
        # Define the excess demand function
        # This maps prices to excess demand
        def excess_demand(prices):
            # Normalize prices to sum to 1 (homogeneity of demand functions)
            prices = np.array(prices)
            prices = prices / np.sum(prices)
            
            # Ensure prices are positive
            prices = np.maximum(prices, 1e-10)
            
            # Calculate wealth for each consumer
            wealth = np.dot(endowments, prices)
            
            # Calculate demand for each good by each consumer
            demand = np.zeros_like(endowments)
            for i in range(2):  # For each consumer
                # Cobb-Douglas demand functions
                demand[i, 0] = alpha[i] * wealth[i] / prices[0]  # Demand for good 1
                demand[i, 1] = (1 - alpha[i]) * wealth[i] / prices[1]  # Demand for good 2
            
            # Total demand
            total_demand = np.sum(demand, axis=0)
            
            # Excess demand = total demand - total resources
            excess = total_demand - total_resources
            
            return excess
        
        # Walrasian equilibrium is when excess demand is zero
        # Define a function whose fixed point is the equilibrium price vector
        def price_adjustment(prices):
            # Normalize prices
            prices = np.array(prices)
            prices = prices / np.sum(prices)
            
            # Calculate excess demand
            excess = excess_demand(prices)
            
            # Adjust prices: increase if excess demand is positive
            new_prices = prices + 0.1 * excess
            
            # Ensure prices remain positive
            new_prices = np.maximum(new_prices, 1e-10)
            
            # Normalize again
            new_prices = new_prices / np.sum(new_prices)
            
            return new_prices
        
        # Create a BrouwerFixedPointTheorem object
        bfpt = BrouwerFixedPointTheorem(dimension=2)
        
        # Find the fixed point (equilibrium prices)
        # Initial guess: equal prices
        initial_guess = np.array([0.5, 0.5])
        
        # Domain is the price simplex
        domain = [(0, 1), (0, 1)]
        
        # Find the fixed point
        equilibrium_prices = bfpt.find_fixed_point(price_adjustment, domain, method='iteration')
        
        # Normalize the equilibrium prices
        equilibrium_prices = equilibrium_prices / np.sum(equilibrium_prices)
        
        # Calculate equilibrium allocations
        wealth = np.dot(endowments, equilibrium_prices)
        allocations = np.zeros_like(endowments)
        for i in range(2):
            allocations[i, 0] = alpha[i] * wealth[i] / equilibrium_prices[0]
            allocations[i, 1] = (1 - alpha[i]) * wealth[i] / equilibrium_prices[1]
        
        # Calculate excess demand at equilibrium
        equilibrium_excess = excess_demand(equilibrium_prices)
        
        # Visualize the economy and equilibrium
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot the endowments
        ax = axs[0]
        ax.bar(np.array([0, 1]) - 0.2, endowments[0], width=0.4, label='Consumer 1')
        ax.bar(np.array([0, 1]) + 0.2, endowments[1], width=0.4, label='Consumer 2')
        ax.set_title('Initial Endowments')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Good 1', 'Good 2'])
        ax.legend()
        
        # Plot the equilibrium prices and excess demand
        ax = axs[1]
        ax.bar([0, 1], equilibrium_prices, color='blue', alpha=0.7, label='Prices')
        ax.set_title('Equilibrium Prices')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Good 1', 'Good 2'])
        ax.set_ylabel('Price')
        
        # Add a twin axis for excess demand
        ax2 = ax.twinx()
        ax2.bar([0, 1], equilibrium_excess, color='red', alpha=0.4, label='Excess Demand')
        ax2.set_ylabel('Excess Demand')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add a combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot the equilibrium allocations
        ax = axs[2]
        ax.bar(np.array([0, 1]) - 0.2, allocations[0], width=0.4, label='Consumer 1')
        ax.bar(np.array([0, 1]) + 0.2, allocations[1], width=0.4, label='Consumer 2')
        ax.set_title('Equilibrium Allocations')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Good 1', 'Good 2'])
        ax.legend()
        
        # Add information about the equilibrium
        plt.figtext(0.5, 0.01, 
                   f"Equilibrium prices: ({equilibrium_prices[0]:.4f}, {equilibrium_prices[1]:.4f})\n"
                   f"Excess demands: ({equilibrium_excess[0]:.4f}, {equilibrium_excess[1]:.4f})\n"
                   f"Consumer 1 wealth: {wealth[0]:.4f}, Consumer 2 wealth: {wealth[1]:.4f}",
                   ha='center', fontsize=10, bbox={"facecolor":"green", "alpha":0.1, "pad":5})
        
        plt.tight_layout()
        plt.show()
        
        print("Economic equilibrium theory relies heavily on fixed point theorems.")
        print("Arrow and Debreu proved the existence of general equilibrium")
        print("using fixed point theorems (Kakutani's generalization of Brouwer's theorem).")
        print("This example shows a simple exchange economy with 2 goods and 2 consumers,")
        print("where equilibrium prices equate supply and demand.")
    
    @staticmethod
    def nonlinear_system_solution():
        """
        Demonstrate how to use fixed point theorems to solve nonlinear systems of equations.
        
        Returns:
        -------
        None
        """
        # Define a nonlinear system: f(x) = 0
        # Example: 
        # f1(x,y) = x^2 + y^2 - 1 = 0
        # f2(x,y) = x^2 - y = 0
        
        def nonlinear_system(p):
            x, y = p
            return np.array([
                x**2 + y**2 - 1,
                x**2 - y
            ])
        
        # Convert to a fixed point problem: x = g(x)
        # We can use various transformations, such as x = x - alpha*f(x)
        def fixed_point_map(p, alpha=0.1):
            return p - alpha * nonlinear_system(p)
        
        # Create a BrouwerFixedPointTheorem object
        bfpt = BrouwerFixedPointTheorem(dimension=2)
        
        # Find the fixed points
        # This nonlinear system has two solutions
        initial_guesses = [
            np.array([0.5, 0.25]),
            np.array([-0.5, 0.25])
        ]
        
        domain = [(-2, 2), (-2, 2)]
        solutions = []
        
        for initial_guess in initial_guesses:
            # Convert the fixed_point_map to a function with one argument
            g = lambda p: fixed_point_map(p, alpha=0.1)
            
            # Find a fixed point
            solution = bfpt.find_fixed_point(g, domain, method='iteration')
            
            # Check if it's a valid solution
            residual = np.linalg.norm(nonlinear_system(solution))
            if residual < 1e-6:
                solutions.append(solution)
        
        # Visualize the nonlinear system and its solutions
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot the curves defined by each equation
        ax = axs[0]
        
        # Create a grid
        x = np.linspace(-1.5, 1.5, 100)
        y = np.linspace(-1.5, 1.5, 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute values for each equation
        F1 = X**2 + Y**2 - 1
        F2 = X**2 - Y
        
        # Plot the contour at level 0 (the curves)
        ax.contour(X, Y, F1, levels=[0], colors='blue', linewidths=2, label='x² + y² = 1')
        ax.contour(X, Y, F2, levels=[0], colors='red', linewidths=2, label='x² = y')
        
        # Plot the solutions
        for i, solution in enumerate(solutions):
            ax.plot(solution[0], solution[1], 'go', markersize=10, 
                   label=f'Solution {i+1}: ({solution[0]:.4f}, {solution[1]:.4f})')
        
        # Add labels and legend
        ax.set_title('Nonlinear System Visualization')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        # Plot the convergence of fixed point iteration
        ax = axs[1]
        
        # Choose one initial guess
        initial_guess = initial_guesses[0]
        
        # Perform iterations manually
        iterations = []
        p = initial_guess.copy()
        iterations.append(p.copy())
        
        g = lambda p: fixed_point_map(p, alpha=0.1)
        
        for _ in range(10):
            p = g(p)
            iterations.append(p.copy())
        
        # Convert to array for easier plotting
        iterations = np.array(iterations)
        
        # Plot the iteration path
        ax.plot(iterations[:, 0], iterations[:, 1], 'o-', label='Iteration Path')
        
        # Also plot the contours
        ax.contour(X, Y, F1, levels=[0], colors='blue', linewidths=2, alpha=0.7)
        ax.contour(X, Y, F2, levels=[0], colors='red', linewidths=2, alpha=0.7)
        
        # Add arrows to show direction
        for i in range(len(iterations)-1):
            ax.annotate('', 
                       xy=(iterations[i+1, 0], iterations[i+1, 1]),
                       xytext=(iterations[i, 0], iterations[i, 1]),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
        
        # Add labels and legend
        ax.set_title('Fixed Point Iteration Convergence')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        print("Nonlinear systems of equations can be transformed into fixed point problems.")
        print("If g(x) = x - alpha*f(x), then fixed points of g correspond to zeros of f.")
        print("This approach is used in many numerical methods like Newton's method.")
        
        # Print detailed results
        print("\nSolutions found:")
        for i, solution in enumerate(solutions):
            residual = np.linalg.norm(nonlinear_system(solution))
            print(f"Solution {i+1}: ({solution[0]:.6f}, {solution[1]:.6f}), Residual: {residual:.6e}")


def main():
    """
    Main function to demonstrate the Brouwer fixed point theorem.
    """
    print("BROUWER FIXED POINT THEOREM DEMONSTRATION")
    print("=========================================")
    print("\nThe Brouwer Fixed Point Theorem states that any continuous function")
    print("from a compact convex set to itself has at least one fixed point.")
    print("A fixed point is a point x such that f(x) = x.")
    print("\nThis program demonstrates the theorem in different dimensions,")
    print("provides visualizations, and shows applications in various fields.")
    
    # Create the BrouwerFixedPointTheorem object
    bfpt = BrouwerFixedPointTheorem()
    
    # 1D Example - Functions on an interval
    print("\n1. Functions on a 1D interval:")
    
    # Define some 1D functions mapping [-1,1] to itself
    def f1(x):
        return 0.5 * x
    
    def f2(x):
        return np.sin(x) / 2
    
    def f3(x):
        return x**3
    
    bfpt.dimension = 1
    bfpt.visualize_fixed_points_1d([f1, f2, f3])
    
    # 2D Example - Functions on a disk
    print("\n2. Functions on a 2D disk:")
    
    # Define some 2D functions mapping the unit disk to itself
    def f2d_1(p):
        x, y = p
        # Rotation + contraction
        return 0.5 * np.array([x*np.cos(np.pi/4) - y*np.sin(np.pi/4),
                              x*np.sin(np.pi/4) + y*np.cos(np.pi/4)])
    
    def f2d_2(p):
        x, y = p
        # Nonlinear contraction
        r = np.sqrt(x**2 + y**2)
        if r < 1e-10:
            return np.array([0, 0])
        else:
            return 0.7 * r * np.array([x/r, y/r])
    
    def f2d_3(p):
        x, y = p
        # Reflection + contraction
        return 0.5 * np.array([-x, y])
    
    bfpt.dimension = 2
    bfpt.visualize_fixed_points_2d([f2d_1, f2d_2, f2d_3])
    
    # 3D Example - Functions on a ball
    print("\n3. Functions on a 3D ball:")
    
    # Define some 3D functions mapping the unit ball to itself
    def f3d_1(p):
        x, y, z = p
        # Rotation + contraction
        return 0.5 * np.array([
            x*np.cos(np.pi/4) - y*np.sin(np.pi/4),
            x*np.sin(np.pi/4) + y*np.cos(np.pi/4),
            z
        ])
    
    def f3d_2(p):
        x, y, z = p
        # Nonlinear transformation
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-10:
            return np.array([0, 0, 0])
        else:
            # Twist around z-axis + contraction
            theta = np.arctan2(y, x) + z
            return 0.7 * np.array([
                r * np.cos(theta),
                r * np.sin(theta),
                0.5 * z
            ])
    
    bfpt.dimension = 3
    bfpt.visualize_fixed_points_3d([f3d_1, f3d_2])
    
    # Proof techniques
    print("\n4. Proof Techniques:")
    
    # Sperner's Lemma
    print("\n4.1 Sperner's Lemma:")
    bfpt.dimension = 2
    
    def rotation_function(p):
        x, y = p
        return np.array([
            0.5 + 0.3 * ((x - 0.5) * np.cos(np.pi/3) - (y - 0.5) * np.sin(np.pi/3)),
            0.5 + 0.3 * ((x - 0.5) * np.sin(np.pi/3) + (y - 0.5) * np.cos(np.pi/3))
        ])
    
    bfpt.implement_sperner_lemma_2d(grid_size=10, func=rotation_function)
    
    # No retraction principle
    print("\n4.2 No Retraction Principle:")
    bfpt.demonstrate_no_retraction(domain_type='disk')
    
    # Winding number argument
    print("\n4.3 Winding Number Argument:")
    bfpt.demonstrate_winding_number()
    
    # Applications
    print("\n5. Applications:")
    
    # Game theory
    print("\n5.1 Game Theory - Nash Equilibrium:")
    FixedPointApplications.game_theory_equilibrium(num_strategies_p1=3, num_strategies_p2=3)
    
    # Economic equilibrium
    print("\n5.2 Economic Equilibrium:")
    FixedPointApplications.economic_equilibrium()
    
    # Nonlinear systems
    print("\n5.3 Nonlinear System Solution:")
    FixedPointApplications.nonlinear_system_solution()
    
    print("\nDemonstration Complete!")


if __name__ == "__main__":
    main()