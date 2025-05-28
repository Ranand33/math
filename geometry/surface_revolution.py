import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import sympy as sp
from matplotlib import cm

class RevolutionVolume:
    """
    Class for calculating volumes of surfaces of revolution using various methods
    and providing visualizations.
    """
    
    def __init__(self):
        """Initialize the calculator."""
        self.precision = 1000  # Number of points for numerical integration/visualization
    
    def disk_method(self, f, a, b, axis='x', method='numerical', steps=1000):
        """
        Calculate volume using the disk method.
        
        Args:
            f: Function that defines the curve (lambda or function)
            a: Lower bound of integration
            b: Upper bound of integration
            axis: Axis of revolution ('x' or 'y')
            method: Calculation method ('numerical' or 'symbolic')
            steps: Number of steps for numerical integration
            
        Returns:
            Volume of the solid of revolution
        """
        if method == 'numerical':
            if axis == 'x':
                # Integrate π * [f(x)]² dx from a to b
                result, error = integrate.quad(lambda x: np.pi * f(x)**2, a, b)
                return result
            elif axis == 'y':
                # Integrate π * [f(y)]² dy from a to b
                result, error = integrate.quad(lambda y: np.pi * f(y)**2, a, b)
                return result
        elif method == 'symbolic':
            # Define symbolic variables
            x, y = sp.symbols('x y')
            
            if axis == 'x':
                # Symbolic expression for disk method around x-axis
                expr = sp.pi * f(x)**2
                return float(sp.integrate(expr, (x, a, b)))
            elif axis == 'y':
                # Symbolic expression for disk method around y-axis
                expr = sp.pi * f(y)**2
                return float(sp.integrate(expr, (y, a, b)))
    
    def washer_method(self, f_outer, f_inner, a, b, axis='x', method='numerical'):
        """
        Calculate volume using the washer method (nested disks).
        
        Args:
            f_outer: Outer function defining the curve (lambda or function)
            f_inner: Inner function defining the curve (lambda or function)
            a: Lower bound of integration
            b: Upper bound of integration
            axis: Axis of revolution ('x' or 'y')
            method: Calculation method ('numerical' or 'symbolic')
            
        Returns:
            Volume of the solid of revolution
        """
        if method == 'numerical':
            if axis == 'x':
                # Integrate π * ([f_outer(x)]² - [f_inner(x)]²) dx from a to b
                result, error = integrate.quad(
                    lambda x: np.pi * (f_outer(x)**2 - f_inner(x)**2), a, b
                )
                return result
            elif axis == 'y':
                # Integrate π * ([f_outer(y)]² - [f_inner(y)]²) dy from a to b
                result, error = integrate.quad(
                    lambda y: np.pi * (f_outer(y)**2 - f_inner(y)**2), a, b
                )
                return result
        elif method == 'symbolic':
            # Define symbolic variables
            x, y = sp.symbols('x y')
            
            if axis == 'x':
                # Symbolic expression for washer method around x-axis
                expr = sp.pi * (f_outer(x)**2 - f_inner(x)**2)
                return float(sp.integrate(expr, (x, a, b)))
            elif axis == 'y':
                # Symbolic expression for washer method around y-axis
                expr = sp.pi * (f_outer(y)**2 - f_inner(y)**2)
                return float(sp.integrate(expr, (y, a, b)))
    
    def shell_method(self, f, a, b, axis='y', method='numerical'):
        """
        Calculate volume using the shell method.
        
        Args:
            f: Function that defines the curve (lambda or function)
            a: Lower bound of integration
            b: Upper bound of integration
            axis: Axis perpendicular to revolution axis ('y' for x-axis rotation, 'x' for y-axis rotation)
            method: Calculation method ('numerical' or 'symbolic')
            
        Returns:
            Volume of the solid of revolution
        """
        if method == 'numerical':
            if axis == 'y':  # Rotation around x-axis, integrating with respect to y
                # Integrate 2π * y * f(y) dy from a to b
                result, error = integrate.quad(lambda y: 2 * np.pi * y * f(y), a, b)
                return result
            elif axis == 'x':  # Rotation around y-axis, integrating with respect to x
                # Integrate 2π * x * f(x) dx from a to b
                result, error = integrate.quad(lambda x: 2 * np.pi * x * f(x), a, b)
                return result
        elif method == 'symbolic':
            # Define symbolic variables
            x, y = sp.symbols('x y')
            
            if axis == 'y':
                # Symbolic expression for shell method around x-axis
                expr = 2 * sp.pi * y * f(y)
                return float(sp.integrate(expr, (y, a, b)))
            elif axis == 'x':
                # Symbolic expression for shell method around y-axis
                expr = 2 * sp.pi * x * f(x)
                return float(sp.integrate(expr, (x, a, b)))
    
    def pappus_guldinus_theorem(self, area, centroid_distance):
        """
        Calculate volume using Pappus-Guldinus theorem.
        
        Args:
            area: Area of the region being rotated
            centroid_distance: Distance from the centroid to the axis of rotation
            
        Returns:
            Volume of the solid of revolution
        """
        return 2 * np.pi * centroid_distance * area
    
    def general_rotation(self, f, a, b, rotation_axis, distance, method='numerical'):
        """
        Calculate volume when rotating around an arbitrary parallel axis.
        
        Args:
            f: Function that defines the curve (lambda or function)
            a: Lower bound of integration
            b: Upper bound of integration
            rotation_axis: Axis parallel to which rotation occurs ('x' or 'y')
            distance: Distance from the standard axis to the rotation axis
            method: Calculation method ('numerical' or 'symbolic')
            
        Returns:
            Volume of the solid of revolution
        """
        if method == 'numerical':
            if rotation_axis == 'x':
                # Rotating around a line parallel to x-axis at distance d from it
                # Integrate π * [(f(x) + d)² - d²] dx from a to b
                result, error = integrate.quad(
                    lambda x: np.pi * ((f(x) + distance)**2 - distance**2), a, b
                )
                return result
            elif rotation_axis == 'y':
                # Rotating around a line parallel to y-axis at distance d from it
                # Integrate π * [(f(y) + d)² - d²] dy from a to b
                result, error = integrate.quad(
                    lambda y: np.pi * ((f(y) + distance)**2 - distance**2), a, b
                )
                return result
        elif method == 'symbolic':
            # Define symbolic variables
            x, y = sp.symbols('x y')
            
            if rotation_axis == 'x':
                # Symbolic expression for rotation around line parallel to x-axis
                expr = sp.pi * ((f(x) + distance)**2 - distance**2)
                return float(sp.integrate(expr, (x, a, b)))
            elif rotation_axis == 'y':
                # Symbolic expression for rotation around line parallel to y-axis
                expr = sp.pi * ((f(y) + distance)**2 - distance**2)
                return float(sp.integrate(expr, (y, a, b)))
    
    def parametric_volume(self, x_func, y_func, t_min, t_max, axis='x', method='numerical'):
        """
        Calculate volume of revolution for a parametric curve.
        
        Args:
            x_func: Parametric function for x-coordinate (lambda or function)
            y_func: Parametric function for y-coordinate (lambda or function)
            t_min: Lower bound of parameter
            t_max: Upper bound of parameter
            axis: Axis of revolution ('x' or 'y')
            method: Calculation method ('numerical' or 'symbolic')
            
        Returns:
            Volume of the solid of revolution
        """
        if method == 'numerical':
            if axis == 'x':
                # Disk method around x-axis for parametric curve
                # Integrate π * [y(t)]² * dx/dt dt from t_min to t_max
                result, error = integrate.quad(
                    lambda t: np.pi * y_func(t)**2 * np.abs(derivative_estimate(x_func, t)), 
                    t_min, t_max
                )
                return result
            elif axis == 'y':
                # Disk method around y-axis for parametric curve
                # Integrate π * [x(t)]² * dy/dt dt from t_min to t_max
                result, error = integrate.quad(
                    lambda t: np.pi * x_func(t)**2 * np.abs(derivative_estimate(y_func, t)), 
                    t_min, t_max
                )
                return result
        elif method == 'symbolic':
            # Define symbolic variables
            t = sp.symbols('t')
            
            # Define symbolic functions
            x_sym = x_func(t)
            y_sym = y_func(t)
            
            if axis == 'x':
                # Symbolic derivative of x with respect to t
                dx_dt = sp.diff(x_sym, t)
                # Symbolic expression for disk method around x-axis
                expr = sp.pi * y_sym**2 * dx_dt
                return float(sp.integrate(expr, (t, t_min, t_max)))
            elif axis == 'y':
                # Symbolic derivative of y with respect to t
                dy_dt = sp.diff(y_sym, t)
                # Symbolic expression for disk method around y-axis
                expr = sp.pi * x_sym**2 * dy_dt
                return float(sp.integrate(expr, (t, t_min, t_max)))
    
    def polar_volume(self, r_func, theta_min, theta_max, axis='x', method='numerical'):
        """
        Calculate volume of revolution for a curve defined in polar coordinates.
        
        Args:
            r_func: Polar function defining the radius (lambda or function)
            theta_min: Lower bound of theta (in radians)
            theta_max: Upper bound of theta (in radians)
            axis: Axis of revolution ('x' or 'y')
            method: Calculation method ('numerical' or 'symbolic')
            
        Returns:
            Volume of the solid of revolution
        """
        if method == 'numerical':
            if axis == 'x':
                # Volume = ∫ π * (r*sin(θ))² * r*cos(θ) dθ from θ_min to θ_max
                result, error = integrate.quad(
                    lambda theta: np.pi * (r_func(theta) * np.sin(theta))**2 * r_func(theta) * np.cos(theta),
                    theta_min, theta_max
                )
                return result
            elif axis == 'y':
                # Volume = ∫ π * (r*cos(θ))² * r*sin(θ) dθ from θ_min to θ_max
                result, error = integrate.quad(
                    lambda theta: np.pi * (r_func(theta) * np.cos(theta))**2 * r_func(theta) * np.sin(theta),
                    theta_min, theta_max
                )
                return result
        elif method == 'symbolic':
            # Define symbolic variables
            theta = sp.symbols('theta')
            
            # Define polar coordinates in terms of theta
            r_sym = r_func(theta)
            
            if axis == 'x':
                # Convert to Cartesian and apply disk method
                y_expr = r_sym * sp.sin(theta)
                dx_expr = r_sym * sp.cos(theta) * sp.diff(theta)
                expr = sp.pi * y_expr**2 * dx_expr
                return float(sp.integrate(expr, (theta, theta_min, theta_max)))
            elif axis == 'y':
                # Convert to Cartesian and apply disk method
                x_expr = r_sym * sp.cos(theta)
                dy_expr = r_sym * sp.sin(theta) * sp.diff(theta)
                expr = sp.pi * x_expr**2 * dy_expr
                return float(sp.integrate(expr, (theta, theta_min, theta_max)))
    
    def visualize_surface_of_revolution(self, f, a, b, axis='x', points=100, ax=None):
        """
        Visualize the surface of revolution.
        
        Args:
            f: Function that defines the curve (lambda or function)
            a: Lower bound
            b: Upper bound
            axis: Axis of revolution ('x' or 'y')
            points: Number of points for visualization
            ax: Optional matplotlib axis for plotting
            
        Returns:
            Matplotlib figure
        """
        # Create figure if not provided
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Generate points along the curve
        t = np.linspace(a, b, points)
        
        if axis == 'x':
            # Generate surface of revolution around x-axis
            x = t
            y = f(t)
            
            # Create a meshgrid
            theta = np.linspace(0, 2*np.pi, points)
            t_grid, theta_grid = np.meshgrid(t, theta)
            
            # Calculate surface coordinates
            X = t_grid
            Y = f(t_grid) * np.cos(theta_grid)
            Z = f(t_grid) * np.sin(theta_grid)
            
            # Plot the surface
            ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
            
            # Plot the generating curve
            ax.plot(x, y, np.zeros_like(x), 'r-', linewidth=2)
            
            # Labels
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('Surface of Revolution around X-axis')
            
        elif axis == 'y':
            # Generate surface of revolution around y-axis
            x = f(t)
            y = t
            
            # Create a meshgrid
            theta = np.linspace(0, 2*np.pi, points)
            t_grid, theta_grid = np.meshgrid(t, theta)
            
            # Calculate surface coordinates
            X = f(t_grid) * np.cos(theta_grid)
            Y = t_grid
            Z = f(t_grid) * np.sin(theta_grid)
            
            # Plot the surface
            ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
            
            # Plot the generating curve
            ax.plot(x, y, np.zeros_like(x), 'r-', linewidth=2)
            
            # Labels
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('Surface of Revolution around Y-axis')
        
        return fig
    
    def visualize_disks(self, f, a, b, axis='x', disks=20, ax=None):
        """
        Visualize the disk method for calculating volumes.
        
        Args:
            f: Function that defines the curve (lambda or function)
            a: Lower bound
            b: Upper bound
            axis: Axis of revolution ('x' or 'y')
            disks: Number of disks to show
            ax: Optional matplotlib axis for plotting
            
        Returns:
            Matplotlib figure
        """
        # Create figure if not provided
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Points for the boundary curve
        t = np.linspace(a, b, 100)
        
        if axis == 'x':
            # Position of disks along x-axis
            x_disks = np.linspace(a, b, disks)
            
            # Plot the generating curve
            ax.plot(t, f(t), np.zeros_like(t), 'r-', linewidth=2)
            
            # Create and plot each disk
            for x_i in x_disks:
                r = f(x_i)  # Radius at this x position
                
                # Circle points
                theta = np.linspace(0, 2*np.pi, 50)
                y = r * np.cos(theta)
                z = r * np.sin(theta)
                
                # Plot the circle edge
                ax.plot(np.ones_like(theta) * x_i, y, z, 'b-', alpha=0.3)
                
                # Create a circular disk (filled)
                theta_grid, r_grid = np.meshgrid(theta, np.linspace(0, r, 5))
                y_grid = r_grid * np.cos(theta_grid)
                z_grid = r_grid * np.sin(theta_grid)
                x_grid = np.ones_like(y_grid) * x_i
                
                # Plot the disk surface
                ax.plot_surface(x_grid, y_grid, z_grid, color='blue', alpha=0.1)
            
            # Labels
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('Disk Method Visualization (X-axis Revolution)')
            
        elif axis == 'y':
            # Position of disks along y-axis
            y_disks = np.linspace(a, b, disks)
            
            # Plot the generating curve
            ax.plot(f(t), t, np.zeros_like(t), 'r-', linewidth=2)
            
            # Create and plot each disk
            for y_i in y_disks:
                r = f(y_i)  # Radius at this y position
                
                # Circle points
                theta = np.linspace(0, 2*np.pi, 50)
                x = r * np.cos(theta)
                z = r * np.sin(theta)
                
                # Plot the circle edge
                ax.plot(x, np.ones_like(theta) * y_i, z, 'b-', alpha=0.3)
                
                # Create a circular disk (filled)
                theta_grid, r_grid = np.meshgrid(theta, np.linspace(0, r, 5))
                x_grid = r_grid * np.cos(theta_grid)
                z_grid = r_grid * np.sin(theta_grid)
                y_grid = np.ones_like(x_grid) * y_i
                
                # Plot the disk surface
                ax.plot_surface(x_grid, y_grid, z_grid, color='blue', alpha=0.1)
            
            # Labels
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('Disk Method Visualization (Y-axis Revolution)')
        
        return fig
    
    def visualize_shells(self, f, a, b, axis='y', shells=20, ax=None):
        """
        Visualize the shell method for calculating volumes.
        
        Args:
            f: Function that defines the curve (lambda or function)
            a: Lower bound
            b: Upper bound
            axis: Axis parallel to shells ('y' for rotation around x-axis, 'x' for rotation around y-axis)
            shells: Number of shells to show
            ax: Optional matplotlib axis for plotting
            
        Returns:
            Matplotlib figure
        """
        # Create figure if not provided
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Points for the boundary curve
        t = np.linspace(a, b, 100)
        
        if axis == 'y':  # Shells around x-axis
            # Positions for shells along y-axis
            y_shells = np.linspace(a, b, shells)
            
            # Plot the generating curve
            ax.plot(f(t), t, np.zeros_like(t), 'r-', linewidth=2)
            
            # Create and plot each cylindrical shell
            for y_i in y_shells:
                h = f(y_i)  # Height of shell
                r = y_i  # Radius of shell
                
                # Points around the cylinder
                theta = np.linspace(0, 2*np.pi, 50)
                
                # Calculate shell coordinates
                x = h
                y = r * np.cos(theta)
                z = r * np.sin(theta)
                
                # Plot the cylinder
                ax.plot(np.ones_like(theta) * h, y, z, 'b-', alpha=0.5)
                
                # Plot shell thickness
                if y_i < b - (b-a)/shells:
                    dr = (b-a)/shells  # Shell thickness
                    theta_sample = [0, np.pi/2, np.pi, 3*np.pi/2]
                    for th in theta_sample:
                        ax.plot([h, h], 
                                [r*np.cos(th), (r+dr)*np.cos(th)], 
                                [r*np.sin(th), (r+dr)*np.sin(th)], 'b-', alpha=0.2)
            
            # Labels
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('Shell Method Visualization (Revolution around X-axis)')
            
        elif axis == 'x':  # Shells around y-axis
            # Positions for shells along x-axis
            x_shells = np.linspace(a, b, shells)
            
            # Plot the generating curve
            ax.plot(t, f(t), np.zeros_like(t), 'r-', linewidth=2)
            
            # Create and plot each cylindrical shell
            for x_i in x_shells:
                h = f(x_i)  # Height of shell
                r = x_i  # Radius of shell
                
                # Points around the cylinder
                theta = np.linspace(0, 2*np.pi, 50)
                
                # Calculate shell coordinates
                x = r * np.cos(theta)
                y = h
                z = r * np.sin(theta)
                
                # Plot the cylinder
                ax.plot(x, np.ones_like(theta) * h, z, 'b-', alpha=0.5)
                
                # Plot shell thickness
                if x_i < b - (b-a)/shells:
                    dr = (b-a)/shells  # Shell thickness
                    theta_sample = [0, np.pi/2, np.pi, 3*np.pi/2]
                    for th in theta_sample:
                        ax.plot([(r)*np.cos(th), (r+dr)*np.cos(th)], 
                                [h, h], 
                                [(r)*np.sin(th), (r+dr)*np.sin(th)], 'b-', alpha=0.2)
            
            # Labels
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('Shell Method Visualization (Revolution around Y-axis)')
        
        return fig


def derivative_estimate(func, t, h=1e-6):
    """
    Estimate the derivative of a function at point t using central difference.
    
    Args:
        func: Function to differentiate
        t: Point at which to evaluate the derivative
        h: Step size for difference
        
    Returns:
        Estimated derivative value
    """
    return (func(t + h) - func(t - h)) / (2 * h)


# Examples and test cases
if __name__ == "__main__":
    # Create a volume calculator
    vol_calc = RevolutionVolume()
    
    print("VOLUME CALCULATIONS FOR COMMON SHAPES")
    print("-" * 50)
    
    # Example 1: Sphere using Disk Method
    radius = 5
    sphere_func = lambda x: np.sqrt(radius**2 - x**2)
    sphere_vol = vol_calc.disk_method(sphere_func, -radius, radius)
    exact_sphere_vol = (4/3) * np.pi * radius**3
    print(f"Sphere with radius {radius}:")
    print(f"  Calculated volume: {sphere_vol:.4f}")
    print(f"  Exact volume: {exact_sphere_vol:.4f}")
    print(f"  Error: {abs(sphere_vol - exact_sphere_vol):.4e}")
    
    # Example 2: Cone using Disk Method
    height = 10
    base_radius = 4
    cone_func = lambda x: (base_radius * (height - x)) / height
    cone_vol = vol_calc.disk_method(cone_func, 0, height)
    exact_cone_vol = (1/3) * np.pi * base_radius**2 * height
    print(f"\nCone with height {height} and base radius {base_radius}:")
    print(f"  Calculated volume: {cone_vol:.4f}")
    print(f"  Exact volume: {exact_cone_vol:.4f}")
    print(f"  Error: {abs(cone_vol - exact_cone_vol):.4e}")
    
    # Example 3: Torus using Parametric Method
    major_radius = 5
    minor_radius = 2
    
    # Parametric functions for a circle shifted by R in x-direction
    x_func = lambda t: major_radius + minor_radius * np.cos(t)
    y_func = lambda t: minor_radius * np.sin(t)
    
    torus_vol = vol_calc.parametric_volume(x_func, y_func, 0, 2*np.pi, axis='y')
    exact_torus_vol = 2 * np.pi**2 * major_radius * minor_radius**2
    print(f"\nTorus with major radius {major_radius} and minor radius {minor_radius}:")
    print(f"  Calculated volume: {torus_vol:.4f}")
    print(f"  Exact volume: {exact_torus_vol:.4f}")
    print(f"  Error: {abs(torus_vol - exact_torus_vol):.4e}")
    
    # Example 4: Ellipsoid using Washer Method
    a, b, c = 3, 4, 5  # Semi-principal axes lengths
    
    # Function for ellipse in xy-plane with z-axis as axis of rotation
    outer_func = lambda x: b * np.sqrt(1 - (x/a)**2)
    inner_func = lambda x: 0  # Inner radius is 0
    
    ellipsoid_vol = vol_calc.washer_method(outer_func, inner_func, -a, a)
    exact_ellipsoid_vol = (4/3) * np.pi * a * b * c
    print(f"\nEllipsoid with semi-axes {a}, {b}, {c}:")
    print(f"  Calculated volume: {ellipsoid_vol:.4f}")
    print(f"  Exact volume: {exact_ellipsoid_vol:.4f}")
    print(f"  Error: {abs(ellipsoid_vol - exact_ellipsoid_vol):.4e}")
    
    # Example 5: Gabriel's Horn (y = 1/x, x ≥ 1) using Shell Method
    print("\nGabriel's Horn (y = 1/x, x ≥ 1, rotated around x-axis):")
    gabriel_func = lambda x: 1/x
    
    # For finite section of Gabriel's Horn from x=1 to x=10
    finite_gabriel_vol = vol_calc.disk_method(gabriel_func, 1, 10, axis='x')
    print(f"  Volume from x=1 to x=10: {finite_gabriel_vol:.4f}")
    
    # The exact volume is π, which is a famous result
    print(f"  As x approaches infinity, the volume approaches π = {np.pi:.4f}")
    
    # Example 6: Paraboloid using Symbolic Method
    print("\nParaboloid (y = x^2, rotated around y-axis):")
    x = sp.symbols('x')
    paraboloid_sym_func = lambda x: x**2
    
    # Calculate volume from x=0 to x=3
    paraboloid_vol = vol_calc.disk_method(paraboloid_sym_func, 0, 3, axis='y', method='symbolic')
    exact_paraboloid_vol = np.pi * 3**3 / 2
    print(f"  Calculated volume (symbolic): {paraboloid_vol:.4f}")
    print(f"  Exact volume: {exact_paraboloid_vol:.4f}")
    print(f"  Error: {abs(paraboloid_vol - exact_paraboloid_vol):.4e}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Visualize a sphere
    sphere_fig = vol_calc.visualize_surface_of_revolution(sphere_func, -radius, radius)
    sphere_fig.suptitle('Sphere (Surface of Revolution)')
    
    # Visualize disk method for cone
    cone_disk_fig = vol_calc.visualize_disks(cone_func, 0, height, disks=10)
    cone_disk_fig.suptitle('Cone (Disk Method)')
    
    # Visualize shell method for paraboloid
    parabola_func = lambda x: x**2
    parabola_shell_fig = vol_calc.visualize_shells(parabola_func, 0, 3, axis='x', shells=12)
    parabola_shell_fig.suptitle('Paraboloid (Shell Method)')
    
    # Show all visualizations
    plt.show()