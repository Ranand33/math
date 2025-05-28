import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc, Wedge, FancyArrowPatch
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

class PlayfairAxiom:
    """
    A class to demonstrate and visualize Playfair's Axiom in different geometries.
    
    Playfair's Axiom states: "In a plane, given a line and a point not on it, 
    at most one line parallel to the given line can be drawn through the point."
    """
    
    def __init__(self):
        """Initialize the Playfair's Axiom visualizer."""
        self.euclidean_color = 'blue'
        self.hyperbolic_color = 'red'
        self.elliptic_color = 'green'
        
    def visualize_playfair_axiom(self, ax=None):
        """
        Create a basic visualization of Playfair's Axiom in Euclidean geometry.
        
        Args:
            ax: Matplotlib axis to draw on (creates one if None)
            
        Returns:
            The matplotlib axis with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw the given line L
        ax.plot([-5, 5], [0, 0], 'k-', linewidth=2, label='Line L')
        
        # Draw the external point P
        point_y = 2
        ax.plot([0], [point_y], 'ko', markersize=8, label='Point P')
        ax.text(0.2, point_y + 0.2, 'P', fontsize=12)
        
        # Draw the unique parallel line through P
        ax.plot([-5, 5], [point_y, point_y], 'b-', linewidth=2, label='Parallel Line')
        
        # Draw some non-parallel lines through P (to show contrast)
        for angle in [-30, -15, 15, 30]:
            # Convert to radians
            angle_rad = angle * np.pi / 180
            
            # Calculate the line endpoints
            dx = 5 * np.cos(angle_rad)
            dy = 5 * np.sin(angle_rad)
            
            # Draw line
            ax.plot([0 - dx, 0 + dx], [point_y - dy, point_y + dy], 'r--', alpha=0.5)
        
        # Show intersection points with the original line
        for angle in [-30, -15, 15, 30]:
            # Convert to radians
            angle_rad = angle * np.pi / 180
            
            # Calculate intersection with x-axis (where the original line is)
            if np.sin(angle_rad) != 0:  # Avoid division by zero
                t = -point_y / np.sin(angle_rad)
                x_intersect = t * np.cos(angle_rad)
                
                # Plot the intersection points
                ax.plot([x_intersect], [0], 'kx', markersize=6)
        
        # Draw perpendicular line from P to L (to emphasize the distance)
        ax.plot([0, 0], [0, point_y], 'g--', linewidth=1.5, label='Perpendicular')
        ax.text(0.1, point_y/2, 'd', fontsize=12, color='green')
        
        # Add text explaining Playfair's axiom
        ax.text(3, 3, "Playfair's Axiom:\n"
                     "Given a line L and a point P not on it,\n"
                     "there exists exactly one line through P\n"
                     "that is parallel to L.", 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Set limits and labels
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("Playfair's Axiom in Euclidean Geometry")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return ax
    
    def visualize_euclidean_parallel_transport(self, ax=None):
        """
        Demonstrate parallel transport in Euclidean geometry, relating to Playfair's axiom.
        
        Args:
            ax: Matplotlib axis to draw on
            
        Returns:
            The matplotlib axis with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw a grid to show the Euclidean plane
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Start with a vector
        origin = np.array([0, 0])
        vector = np.array([1, 0.5])
        
        # Draw the original vector
        ax.arrow(origin[0], origin[1], vector[0], vector[1], 
                head_width=0.1, head_length=0.2, fc='blue', ec='blue', 
                label='Original Vector')
        
        # Draw the path along which we'll transport the vector
        path_points = np.array([
            [0, 0],
            [1, 1],
            [2, 0],
            [3, 2],
            [4, 1]
        ])
        
        # Draw the path
        ax.plot(path_points[:, 0], path_points[:, 1], 'k-', linewidth=2, label='Transport Path')
        
        # Transport the vector along the path
        for i in range(len(path_points)-1):
            # Calculate the displacement between path points
            displacement = path_points[i+1] - path_points[i]
            
            # Draw the parallel transported vector at each path point
            ax.arrow(path_points[i][0], path_points[i][1], 
                    vector[0], vector[1], 
                    head_width=0.1, head_length=0.2, fc='blue', ec='blue', alpha=0.7)
            
            # For the last point
            if i == len(path_points)-2:
                ax.arrow(path_points[i+1][0], path_points[i+1][1], 
                        vector[0], vector[1], 
                        head_width=0.1, head_length=0.2, fc='blue', ec='blue', alpha=0.7)
        
        # Add explanation
        ax.text(2.5, 3, "Parallel Transport in Euclidean Geometry:\n"
                       "A vector remains parallel to itself when transported along any path.\n"
                       "This property is closely related to Playfair's Axiom.", 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Set limits and labels
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 4)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("Parallel Transport in Euclidean Geometry")
        ax.legend(loc='upper left')
        
        return ax
    
    def visualize_parallel_postulate_forms(self, ax=None):
        """
        Visualize different but equivalent forms of the parallel postulate.
        
        Args:
            ax: Matplotlib axis to draw on
            
        Returns:
            The matplotlib axis with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set up the figure with 4 subplots for different forms
        if hasattr(ax, 'get_figure'):
            fig = ax.get_figure()
            fig.clear()
            
            # Create 4 subplots
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
        else:
            # If ax is None or not a valid axis
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Playfair's Axiom
        self.visualize_playfair_axiom(ax1)
        ax1.set_title("Playfair's Axiom")
        ax1.get_legend().remove()  # Remove legend to save space
        
        # 2. Euclid's fifth postulate (sum of interior angles of a triangle = 180°)
        self._draw_triangle_angles(ax2)
        
        # 3. Sum of interior angles of a triangle = 180°
        self._draw_triangle_angles(ax3, include_sum=True)
        
        # 4. Existence of rectangles/similar triangles
        self._draw_rectangle_existence(ax4)
        
        plt.tight_layout()
        return fig
    
    def _draw_triangle_angles(self, ax, include_sum=False):
        """Helper method to draw a triangle with marked angles."""
        # Draw a triangle
        triangle_points = np.array([
            [0, 0],
            [4, 1],
            [1, 3]
        ])
        
        ax.plot([triangle_points[0, 0], triangle_points[1, 0], triangle_points[2, 0], triangle_points[0, 0]],
               [triangle_points[0, 1], triangle_points[1, 1], triangle_points[2, 1], triangle_points[0, 1]],
               'b-', linewidth=2)
        
        # Mark the angles
        angles = []
        for i in range(3):
            # Get the vertices
            p1 = triangle_points[i]
            p2 = triangle_points[(i+1) % 3]
            p3 = triangle_points[(i+2) % 3]
            
            # Calculate the vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate the angle
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(np.degrees(angle))
            
            # Draw the angle arc
            arc_radius = 0.4
            start_angle = np.arctan2(v1[1], v1[0]) * 180 / np.pi
            end_angle = np.arctan2(v2[1], v2[0]) * 180 / np.pi
            
            if start_angle < 0:
                start_angle += 360
            if end_angle < 0:
                end_angle += 360
                
            # Ensure we get the interior angle
            angle_diff = end_angle - start_angle
            if angle_diff < 0:
                angle_diff += 360
            if angle_diff > 180:
                start_angle, end_angle = end_angle, start_angle
                
            # Draw the arc
            arc = Arc((p2[0], p2[1]), arc_radius, arc_radius, 
                     theta1=start_angle, theta2=end_angle, 
                     color='red', lw=1.5)
            ax.add_patch(arc)
            
            # Add the angle label
            mid_angle = (start_angle + end_angle) / 2
            if mid_angle > 180:
                mid_angle -= 180
            label_x = p2[0] + 0.7 * arc_radius * np.cos(mid_angle * np.pi / 180)
            label_y = p2[1] + 0.7 * arc_radius * np.sin(mid_angle * np.pi / 180)
            
            ax.text(label_x, label_y, f"α{i+1}", fontsize=12, ha='center', va='center')
        
        # Show that the sum is 180°
        if include_sum:
            sum_text = f"α1 + α2 + α3 = {angles[0]:.1f}° + {angles[1]:.1f}° + {angles[2]:.1f}° = 180°"
            ax.text(2, -0.5, sum_text, fontsize=12, ha='center')
            
            # Extra explanatory text
            ax.text(2, 4, "In Euclidean geometry, the sum of interior angles\n"
                           "of any triangle is always 180° (π radians).\n"
                           "This is equivalent to Playfair's Axiom.", 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        else:
            # Explanatory text for Euclid's version
            ax.text(2, 4, "Euclid's Fifth Postulate (Original Form):\n"
                           "If a line crosses two other lines and makes interior angles\n"
                           "on the same side less than 180°, then the two lines\n"
                           "will intersect on that side if extended far enough.", 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
            
            # Draw Euclid's original formulation
            l1_start = [-1, 1]
            l1_end = [5, 1]
            l2_start = [0, 0]
            l2_end = [3, 3]
            l3_start = [3, 0]
            l3_end = [4.5, 3]
            
            ax.plot([l1_start[0], l1_end[0]], [l1_start[1], l1_end[1]], 'k-', linewidth=1.5)
            ax.plot([l2_start[0], l2_end[0]], [l2_start[1], l2_end[1]], 'g-', linewidth=1.5)
            ax.plot([l3_start[0], l3_end[0]], [l3_start[1], l3_end[1]], 'g-', linewidth=1.5)
            
            # Mark the angles
            angle_pos = [2, 1]
            arc = Arc((angle_pos[0], angle_pos[1]), 0.5, 0.5, 
                     theta1=0, theta2=30, 
                     color='red', lw=1.5)
            ax.add_patch(arc)
            
            angle_pos = [3, 1]
            arc = Arc((angle_pos[0], angle_pos[1]), 0.5, 0.5, 
                     theta1=0, theta2=40, 
                     color='red', lw=1.5)
            ax.add_patch(arc)
            
            # Label
            ax.text(2.3, 1.3, "γ1", fontsize=12)
            ax.text(3.3, 1.3, "γ2", fontsize=12)
            ax.text(4, 2, "γ1 + γ2 < 180°", fontsize=10)
        
        # Set limits and labels
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        if include_sum:
            ax.set_title("Sum of Triangle Angles = 180°")
        else:
            ax.set_title("Euclid's Fifth Postulate")
        
        return ax
    
    def _draw_rectangle_existence(self, ax):
        """Helper method to draw the rectangle existence form of the parallel postulate."""
        # Draw a rectangle
        rectangle = np.array([
            [1, 1],
            [4, 1],
            [4, 3],
            [1, 3]
        ])
        
        ax.plot([rectangle[0, 0], rectangle[1, 0], rectangle[2, 0], rectangle[3, 0], rectangle[0, 0]],
               [rectangle[0, 1], rectangle[1, 1], rectangle[2, 1], rectangle[3, 1], rectangle[0, 1]],
               'b-', linewidth=2)
        
        # Mark the right angles
        for i in range(4):
            corner = rectangle[i]
            ax.plot([corner[0], corner[0] + 0.2], [corner[1], corner[1]], 'k-', linewidth=1)
            ax.plot([corner[0], corner[0]], [corner[1], corner[1] + 0.2], 'k-', linewidth=1)
        
        # Draw some similar triangles
        triangle1 = np.array([
            [0.5, 0.5],
            [2.5, 0.5],
            [0.5, 1.5]
        ])
        
        triangle2 = np.array([
            [5, 2],
            [6, 2],
            [5, 2.5]
        ])
        
        ax.plot([triangle1[0, 0], triangle1[1, 0], triangle1[2, 0], triangle1[0, 0]],
               [triangle1[0, 1], triangle1[1, 1], triangle1[2, 1], triangle1[0, 1]],
               'g-', linewidth=2)
        
        ax.plot([triangle2[0, 0], triangle2[1, 0], triangle2[2, 0], triangle2[0, 0]],
               [triangle2[0, 1], triangle2[1, 1], triangle2[2, 1], triangle2[0, 1]],
               'g-', linewidth=2)
        
        # Add explanatory text
        ax.text(3, 4, "Equivalent forms of the parallel postulate:\n"
                     "1. Rectangles exist\n"
                     "2. Similar triangles exist\n"
                     "3. The Pythagorean theorem holds\n"
                     "All these are equivalent to Playfair's Axiom.",
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Set limits and labels
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("Existence of Rectangles & Similar Triangles")
        
        return ax
    
    def visualize_non_euclidean_comparison(self, ax=None):
        """
        Compare Playfair's axiom in Euclidean, hyperbolic, and elliptic geometries.
        
        Args:
            ax: Matplotlib axis to draw on
            
        Returns:
            The matplotlib figure with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 8))
            
        # Set up the figure with 3 subplots for different geometries
        if hasattr(ax, 'get_figure'):
            fig = ax.get_figure()
            fig.clear()
            
            # Create 3 subplots
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
        else:
            # If ax is None or not a valid axis
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
        
        # 1. Euclidean geometry
        self._draw_euclidean_case(ax1)
        
        # 2. Hyperbolic geometry (Poincaré disk model)
        self._draw_hyperbolic_case(ax2)
        
        # 3. Elliptic/spherical geometry
        self._draw_elliptic_case(ax3)
        
        plt.tight_layout()
        fig.suptitle("Playfair's Axiom in Different Geometries", fontsize=16, y=1.05)
        
        return fig
    
    def _draw_euclidean_case(self, ax):
        """Helper method to draw the Euclidean case."""
        # Draw a grid to represent the Euclidean plane
        for x in np.arange(-5, 6, 1):
            ax.plot([x, x], [-5, 5], 'gray', alpha=0.3)
        for y in np.arange(-5, 6, 1):
            ax.plot([-5, 5], [y, y], 'gray', alpha=0.3)
            
        # Draw the original line L
        ax.plot([-5, 5], [0, 0], 'k-', linewidth=2, label='Line L')
        
        # Draw the point P not on L
        ax.plot([0], [2], 'ko', markersize=8, label='Point P')
        ax.text(0.2, 2.2, 'P', fontsize=12)
        
        # Draw the unique parallel line through P
        ax.plot([-5, 5], [2, 2], self.euclidean_color, linewidth=2, label='Unique Parallel')
        
        # Draw some non-parallel lines through P
        for angle in [-30, -15, 15, 30]:
            # Convert to radians
            angle_rad = angle * np.pi / 180
            
            # Calculate the line endpoints
            dx = 5 * np.cos(angle_rad)
            dy = 5 * np.sin(angle_rad)
            
            # Draw line
            ax.plot([0 - dx, 0 + dx], [2 - dy, 2 + dy], 'r--', alpha=0.5)
            
            # Mark intersection with original line
            if dy != 0:  # Avoid division by zero
                t = -2 / dy
                x_intersect = t * dx
                ax.plot([x_intersect], [0], 'kx', markersize=6)
        
        # Add explanation
        ax.text(0, -3.5, "Euclidean Geometry:\n"
                        "Exactly ONE line parallel to L\n"
                        "can be drawn through point P.\n"
                        "This is Playfair's Axiom.", 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title("Euclidean Geometry")
        ax.legend(loc='upper left')
        
        return ax
    
    def _draw_hyperbolic_case(self, ax):
        """Helper method to draw the hyperbolic case using the Poincaré disk model."""
        # Draw Poincaré disk
        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Draw some hyperbolic lines for context (geodesics in the Poincaré model)
        # These are arcs of circles orthogonal to the boundary
        for angle in np.arange(0, 360, 30):
            rad = angle * np.pi / 180
            # Create a circle with center outside the disk
            center_dist = 1.5
            center_x = center_dist * np.cos(rad)
            center_y = center_dist * np.sin(rad)
            
            # Radius is distance from center to disk edge
            radius = np.sqrt(center_dist**2 - 1)
            
            # Create the circle (we'll only see the part inside the disk)
            arc = plt.Circle((center_x, center_y), radius, fill=False, edgecolor='gray', alpha=0.3)
            ax.add_patch(arc)
        
        # Draw the original hyperbolic line L (here a diameter for simplicity)
        ax.plot([-1, 1], [0, 0], 'k-', linewidth=2, label='Line L')
        
        # Draw the point P not on L
        ax.plot([0], [0.5], 'ko', markersize=8, label='Point P')
        ax.text(0.05, 0.55, 'P', fontsize=12)
        
        # Draw multiple parallels to L through P (a key feature of hyperbolic geometry)
        # These are arcs of circles orthogonal to the boundary
        
        # Draw a few parallel hyperbolic lines through P
        num_parallels = 5
        for i in range(num_parallels):
            # Parameters for hyperbolic line (circle arc orthogonal to boundary)
            center_y = 2.5 - i*0.8  # Different centers to create multiple parallels
            center_x = 0
            
            # Radius based on center position (to make orthogonal to boundary)
            radius = np.sqrt(center_y**2 + center_x**2 - 1)
            
            # Draw the arc
            arc = plt.Circle((center_x, center_y), radius, fill=False, 
                           edgecolor=self.hyperbolic_color, linewidth=2, alpha=0.7)
            ax.add_patch(arc)
            
            # Label the first one
            if i == 0:
                ax.text(0.7, 0.75, "Parallel 1", color=self.hyperbolic_color, fontsize=10)
            elif i == num_parallels-1:
                ax.text(-0.7, 0.25, f"Parallel {num_parallels}", color=self.hyperbolic_color, fontsize=10)
        
        # Add explanation
        ax.text(0, -1.3, "Hyperbolic Geometry:\n"
                         "INFINITELY MANY lines parallel to L\n"
                         "can be drawn through point P.\n"
                         "Playfair's Axiom does NOT hold.", 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title("Hyperbolic Geometry\n(Poincaré Disk Model)")
        ax.legend(loc='upper left')
        
        return ax
    
    def _draw_elliptic_case(self, ax):
        """Helper method to draw the elliptic case (visualize on a sphere)."""
        # Create a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)
        
        # Draw the original "line" L (a great circle on the sphere)
        # For simplicity, we'll use the equator
        theta = np.linspace(0, 2*np.pi, 100)
        x_equator = np.cos(theta)
        y_equator = np.sin(theta)
        z_equator = np.zeros_like(theta)
        
        ax.plot(x_equator, y_equator, z_equator, 'k-', linewidth=2, label='Line L (Equator)')
        
        # Draw the point P not on L
        point_lat = np.pi/4  # 45 degrees north
        point_lon = np.pi/4  # 45 degrees east
        p_x = np.cos(point_lon) * np.cos(point_lat)
        p_y = np.sin(point_lon) * np.cos(point_lat)
        p_z = np.sin(point_lat)
        
        ax.plot([p_x], [p_y], [p_z], 'ko', markersize=8, label='Point P')
        ax.text(p_x+0.1, p_y+0.1, p_z+0.1, 'P', fontsize=12)
        
        # In elliptic geometry, there are NO parallel lines
        # All great circles (geodesics) intersect each other
        
        # Draw a great circle through P with the same "direction" as L
        # This would be a longitudinal great circle
        phi = np.linspace(0, 2*np.pi, 100)
        x_gc = np.cos(point_lon) * np.cos(phi)
        y_gc = np.sin(point_lon) * np.cos(phi)
        z_gc = np.sin(phi)
        
        ax.plot(x_gc, y_gc, z_gc, color=self.elliptic_color, linewidth=2, label='Great Circle through P')
        
        # Mark the intersection points with the equator
        intersection1 = [np.cos(point_lon), np.sin(point_lon), 0]
        intersection2 = [-np.cos(point_lon), -np.sin(point_lon), 0]
        
        ax.plot([intersection1[0]], [intersection1[1]], [intersection1[2]], 'rx', markersize=8)
        ax.plot([intersection2[0]], [intersection2[1]], [intersection2[2]], 'rx', markersize=8)
        
        # Add explanation as text in 3D
        ax.text(0, 0, -1.5, "Elliptic Geometry:\n"
                          "NO lines parallel to L can be drawn through point P.\n"
                          "All geodesics (great circles) eventually intersect.\n"
                          "Playfair's Axiom does NOT hold.", 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title("Elliptic Geometry\n(Spherical Model)")
        ax.legend(loc='upper left')
        
        # Set view angle
        ax.view_init(elev=20, azim=30)
        
        return ax
    
    def visualize_triangle_angles(self, ax=None):
        """
        Visualize how triangle angles differ in different geometries.
        
        Args:
            ax: Matplotlib axis to draw on
            
        Returns:
            The matplotlib figure with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 5))
            
        # Set up the figure with 3 subplots for different geometries
        if hasattr(ax, 'get_figure'):
            fig = ax.get_figure()
            fig.clear()
            
            # Create 3 subplots
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133, projection='3d')
        else:
            # If ax is None or not a valid axis
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133, projection='3d')
        
        # 1. Euclidean triangle (sum = 180°)
        self._draw_euclidean_triangle(ax1)
        
        # 2. Hyperbolic triangle (sum < 180°)
        self._draw_hyperbolic_triangle(ax2)
        
        # 3. Elliptic/spherical triangle (sum > 180°)
        self._draw_spherical_triangle(ax3)
        
        plt.tight_layout()
        fig.suptitle("Triangle Angles in Different Geometries", fontsize=16, y=1.05)
        
        return fig
    
    def _draw_euclidean_triangle(self, ax):
        """Helper method to draw a Euclidean triangle and its angles."""
        # Define triangle vertices
        triangle = np.array([
            [0, 0],
            [4, 0],
            [2, 3]
        ])
        
        # Draw the triangle
        ax.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
               [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]],
               color=self.euclidean_color, linewidth=2)
        
        # Label the vertices
        ax.text(triangle[0, 0] - 0.3, triangle[0, 1] - 0.3, 'A', fontsize=12)
        ax.text(triangle[1, 0] + 0.3, triangle[1, 1] - 0.3, 'B', fontsize=12)
        ax.text(triangle[2, 0] + 0.3, triangle[2, 1] + 0.3, 'C', fontsize=12)
        
        # Calculate and mark the angles
        angles = []
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i+1) % 3]
            p3 = triangle[(i+2) % 3]
            
            # Vectors for angle calculation
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(np.degrees(angle))
            
            # Draw angle arc
            radius = 0.4
            
            start_angle = np.arctan2(v1[1], v1[0]) * 180 / np.pi
            end_angle = np.arctan2(v2[1], v2[0]) * 180 / np.pi
            
            if start_angle < 0:
                start_angle += 360
            if end_angle < 0:
                end_angle += 360
                
            # Ensure we get the interior angle
            angle_diff = end_angle - start_angle
            if angle_diff < 0:
                angle_diff += 360
            if angle_diff > 180:
                start_angle, end_angle = end_angle, start_angle
                
            arc = Arc((p2[0], p2[1]), radius, radius, 
                     theta1=start_angle, theta2=end_angle, 
                     color='red', lw=1.5)
            ax.add_patch(arc)
            
            # Add angle label
            angle_label = f"{angles[-1]:.1f}°"
            
            # Position the label at the middle of the arc
            mid_angle_rad = (start_angle + end_angle) / 2 * np.pi / 180
            label_x = p2[0] + radius * 0.7 * np.cos(mid_angle_rad)
            label_y = p2[1] + radius * 0.7 * np.sin(mid_angle_rad)
            
            ax.text(label_x, label_y, angle_label, fontsize=10, ha='center', va='center')
        
        # Show the angle sum
        angle_sum = sum(angles)
        ax.text(2, -1, f"Sum of angles: {angles[0]:.1f}° + {angles[1]:.1f}° + {angles[2]:.1f}° = {angle_sum:.1f}°", 
               fontsize=12, ha='center')
        
        # Add explanation
        ax.text(2, 4, "Euclidean Geometry:\n"
                     "Sum of triangle angles = 180°\n"
                     "This is consistent with Playfair's Axiom", 
               fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1.5, 4.5)
        ax.set_aspect('equal')
        ax.set_title("Euclidean Triangle")
        
        return ax
    
    def _draw_hyperbolic_triangle(self, ax):
        """Helper method to draw a hyperbolic triangle and its angles."""
        # Draw Poincaré disk boundary
        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Define triangle vertices in the Poincaré disk
        # We'll use three points and create hyperbolic lines (geodesics) between them
        triangle_points = np.array([
            [0, 0],       # Center of disk
            [0.6, 0],     # Point on positive x-axis
            [0.3, 0.5]    # Point in first quadrant
        ])
        
        # Draw the hyperbolic lines (geodesics) between vertices
        # In the Poincaré model, geodesics are arcs of circles orthogonal to the boundary
        
        # Line 1: From (0,0) to (0.6,0) - This is a diameter
        ax.plot([triangle_points[0, 0], triangle_points[1, 0]], 
               [triangle_points[0, 1], triangle_points[1, 1]], 
               color=self.hyperbolic_color, linewidth=2)
        
        # Line 2: From (0,0) to (0.3,0.5) - This is also a diameter
        ax.plot([triangle_points[0, 0], triangle_points[2, 0]], 
               [triangle_points[0, 1], triangle_points[2, 1]], 
               color=self.hyperbolic_color, linewidth=2)
        
        # Line 3: From (0.6,0) to (0.3,0.5) - This is an arc of a circle orthogonal to boundary
        # Calculate the center and radius of the circle
        p1 = triangle_points[1]
        p2 = triangle_points[2]
        
        # The Poincaré disk model requires a circle orthogonal to the boundary
        # connecting these two points
        
        # Calculate circle center
        # We solve for a circle passing through p1, p2, and orthogonal to the unit circle
        
        # Let's calculate a simple approximation for demonstration
        # This is not the exact formula for the hyperbolic geodesic
        midpoint = (p1 + p2) / 2
        dist_to_origin = np.linalg.norm(midpoint)
        
        # Scale the midpoint to find the center of the circle
        # This is a simplified approach
        center = midpoint * (1 + 0.5/dist_to_origin**2)
        
        # Calculate radius from center to one of the points
        radius = np.linalg.norm(center - p1)
        
        # Draw the arc
        # Calculate the angles for the arc
        angle1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
        angle2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
        
        # Convert to degrees
        angle1_deg = angle1 * 180 / np.pi
        angle2_deg = angle2 * 180 / np.pi
        
        # Create the arc
        arc = Arc((center[0], center[1]), 2*radius, 2*radius, 
                 theta1=min(angle1_deg, angle2_deg), theta2=max(angle1_deg, angle2_deg), 
                 color=self.hyperbolic_color, linewidth=2)
        ax.add_patch(arc)
        
        # Label the vertices
        ax.text(triangle_points[0, 0] - 0.1, triangle_points[0, 1] - 0.1, 'A', fontsize=12)
        ax.text(triangle_points[1, 0] + 0.1, triangle_points[1, 1] - 0.1, 'B', fontsize=12)
        ax.text(triangle_points[2, 0] + 0.1, triangle_points[2, 1] + 0.1, 'C', fontsize=12)
        
        # Draw angles
        # Simplification: We'll represent the angles in the Poincaré model as they appear in the figure
        # Note: This is a visual approximation, not the true hyperbolic angles
        
        # Angle at A
        ax.add_patch(Arc((triangle_points[0, 0], triangle_points[0, 1]), 0.3, 0.3, 
                        theta1=0, theta2=np.arctan2(triangle_points[2, 1], triangle_points[2, 0])*180/np.pi, 
                        color='red', linewidth=1.5))
        
        # Angle at B
        start_angle = 180
        end_angle = np.arctan2(p2[1] - center[1], p2[0] - center[0]) * 180 / np.pi
        if end_angle < start_angle:
            end_angle += 360
        ax.add_patch(Arc((triangle_points[1, 0], triangle_points[1, 1]), 0.3, 0.3, 
                        theta1=start_angle, theta2=end_angle, 
                        color='red', linewidth=1.5))
        
        # Angle at C
        start_angle = np.arctan2(center[1] - p2[1], center[0] - p2[0]) * 180 / np.pi
        if start_angle < 0:
            start_angle += 360
        end_angle = np.arctan2(-triangle_points[2, 1], -triangle_points[2, 0]) * 180 / np.pi
        if end_angle < 0:
            end_angle += 360
        ax.add_patch(Arc((triangle_points[2, 0], triangle_points[2, 1]), 0.3, 0.3, 
                        theta1=start_angle, theta2=end_angle, 
                        color='red', linewidth=1.5))
        
        # Approximated angles for demonstration
        angle_A = 50
        angle_B = 55
        angle_C = 60
        
        # Place angle labels
        ax.text(triangle_points[0, 0] + 0.15, triangle_points[0, 1] + 0.05, f"{angle_A}°", fontsize=10)
        ax.text(triangle_points[1, 0] - 0.25, triangle_points[1, 1] + 0.05, f"{angle_B}°", fontsize=10)
        ax.text(triangle_points[2, 0] - 0.05, triangle_points[2, 1] - 0.2, f"{angle_C}°", fontsize=10)
        
        # Show the angle sum
        angle_sum = angle_A + angle_B + angle_C
        ax.text(0, -1.3, f"Sum of angles: {angle_A}° + {angle_B}° + {angle_C}° = {angle_sum}° < 180°", 
               fontsize=12, ha='center')
        
        # Add explanation
        ax.text(0, 1.3, "Hyperbolic Geometry:\n"
                       "Sum of triangle angles < 180°\n"
                       "The 'defect' increases with triangle size", 
               fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title("Hyperbolic Triangle\n(Poincaré Disk Model)")
        
        return ax
    
    def _draw_spherical_triangle(self, ax):
        """Helper method to draw a spherical triangle and its angles."""
        # Create a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
        
        # Define three points on the sphere (vertices of the triangle)
        # We'll use spherical coordinates for clarity
        # (longitude, latitude) pairs converted to Cartesian
        
        # Convert spherical coordinates to Cartesian
        def sph_to_cart(lon, lat):
            return [
                np.cos(lon) * np.cos(lat),
                np.sin(lon) * np.cos(lat),
                np.sin(lat)
            ]
        
        # Points for a spherical triangle (90-90-90 is a common example)
        A = sph_to_cart(0, 0)            # Point on the equator at 0 longitude
        B = sph_to_cart(np.pi/2, 0)      # Point on the equator at 90° longitude
        C = sph_to_cart(0, np.pi/2)      # North pole
        
        # Draw great circle arcs between the points (geodesics on a sphere)
        def draw_great_circle(p1, p2, color):
            # For a great circle between two points, we can parameterize as:
            # p(t) = p1*sin((1-t)*alpha) + p2*sin(t*alpha) / sin(alpha)
            # where alpha is the angle between the points
            
            # Calculate the angle between the points
            cos_alpha = np.dot(p1, p2)
            alpha = np.arccos(max(min(cos_alpha, 1.0), -1.0))  # Clamp to avoid numerical issues
            
            # Generate points along the great circle
            t = np.linspace(0, 1, 100)
            points = []
            
            for time in t:
                if abs(alpha) < 1e-10:  # Points are too close
                    point = p1
                else:
                    point = [p1[i] * np.sin((1-time)*alpha) + p2[i] * np.sin(time*alpha) for i in range(3)]
                    point = [p / np.sin(alpha) for p in point]
                
                # Normalize to ensure point is on the sphere
                norm = np.sqrt(sum(p**2 for p in point))
                point = [p / norm for p in point]
                points.append(point)
            
            # Extract x, y, z coordinates
            x, y, z = zip(*points)
            ax.plot(x, y, z, color=color, linewidth=2)
        
        # Draw the sides of the triangle
        draw_great_circle(A, B, self.elliptic_color)
        draw_great_circle(B, C, self.elliptic_color)
        draw_great_circle(C, A, self.elliptic_color)
        
        # Mark the vertices
        ax.scatter([A[0]], [A[1]], [A[2]], color='black', s=50)
        ax.scatter([B[0]], [B[1]], [B[2]], color='black', s=50)
        ax.scatter([C[0]], [C[1]], [C[2]], color='black', s=50)
        
        # Label the vertices
        ax.text(A[0]+0.1, A[1]+0.1, A[2]+0.1, 'A', fontsize=12)
        ax.text(B[0]+0.1, B[1]+0.1, B[2]+0.1, 'B', fontsize=12)
        ax.text(C[0]+0.1, C[1]+0.1, C[2]+0.1, 'C', fontsize=12)
        
        # For this triangle, all angles are 90° (right angles)
        # Draw small markers for the right angles
        def mark_right_angle(p, v1, v2):
            # Create a small right angle marker at point p
            # v1 and v2 are the directions of the adjacent sides
            
            # Calculate tangent vectors at p in the directions of v1 and v2
            # Tangent to a sphere at point p is perpendicular to p
            
            # Normalize vectors
            p_norm = np.sqrt(sum(pi**2 for pi in p))
            p_unit = [pi/p_norm for pi in p]
            
            v1_norm = np.sqrt(sum(vi**2 for vi in v1))
            v1_unit = [vi/v1_norm for vi in v1]
            
            v2_norm = np.sqrt(sum(vi**2 for vi in v2))
            v2_unit = [vi/v2_norm for vi in v2]
            
            # Tangent vectors (simplified approach)
            t1 = [v1_unit[i] - p_unit[i] * np.dot(v1_unit, p_unit) for i in range(3)]
            t2 = [v2_unit[i] - p_unit[i] * np.dot(v2_unit, p_unit) for i in range(3)]
            
            # Normalize
            t1_norm = np.sqrt(sum(ti**2 for ti in t1))
            t1 = [ti/t1_norm * 0.1 for ti in t1]
            
            t2_norm = np.sqrt(sum(ti**2 for ti in t2))
            t2 = [ti/t2_norm * 0.1 for ti in t2]
            
            # Draw small right angle symbol
            ax.plot([p[0], p[0]+t1[0]], [p[1], p[1]+t1[1]], [p[2], p[2]+t1[2]], 'r-', linewidth=1.5)
            ax.plot([p[0]+t1[0], p[0]+t1[0]+t2[0]], [p[1]+t1[1], p[1]+t1[1]+t2[1]], 
                   [p[2]+t1[2], p[2]+t1[2]+t2[2]], 'r-', linewidth=1.5)
        
        # Mark the right angles (simplified for clarity)
        # In this 90-90-90 triangle, all angles are right angles
        angle_A = 90
        angle_B = 90
        angle_C = 90
        
        # Place angle labels
        ax.text(A[0]-0.3, A[1], A[2], f"{angle_A}°", color='red', fontsize=10)
        ax.text(B[0], B[1]-0.3, B[2], f"{angle_B}°", color='red', fontsize=10)
        ax.text(C[0], C[1], C[2]+0.2, f"{angle_C}°", color='red', fontsize=10)
        
        # Show the angle sum
        angle_sum = angle_A + angle_B + angle_C
        ax.text(0, 0, -1.5, f"Sum of angles: {angle_A}° + {angle_B}° + {angle_C}° = {angle_sum}° > 180°", 
               fontsize=12, ha='center')
        
        # Add explanation
        ax.text(0, 0, 1.5, "Elliptic/Spherical Geometry:\n"
                          "Sum of triangle angles > 180°\n"
                          "The 'excess' increases with triangle size", 
               fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title("Spherical Triangle")
        
        # Set view angle
        ax.view_init(elev=20, azim=30)
        
        return ax
    
    def interactive_demo(self):
        """
        Create an interactive demonstration of Playfair's axiom.
        
        Returns:
            Interactive matplotlib figure
        """
        # Create figure and subplots
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initial setup
        line_y = 0
        point_y = 2
        
        def update(event):
            ax.clear()
            
            # Draw the given line L
            ax.plot([-5, 5], [line_y, line_y], 'k-', linewidth=2, label='Line L')
            
            # Draw the external point P
            ax.plot([0], [point_y], 'ko', markersize=8, label='Point P')
            ax.text(0.2, point_y + 0.2, 'P', fontsize=12)
            
            # Draw the unique parallel line through P
            ax.plot([-5, 5], [point_y, point_y], 'b-', linewidth=2, label='Parallel Line')
            
            # Draw some non-parallel lines through P
            for angle in [-30, -15, 15, 30]:
                # Convert to radians
                angle_rad = angle * np.pi / 180
                
                # Calculate the line endpoints
                dx = 5 * np.cos(angle_rad)
                dy = 5 * np.sin(angle_rad)
                
                # Draw line
                ax.plot([0 - dx, 0 + dx], [point_y - dy, point_y + dy], 'r--', alpha=0.5)
                
                # Mark intersection with the original line
                if dy != 0:  # Avoid division by zero
                    t = (line_y - point_y) / dy
                    x_intersect = t * dx
                    ax.plot([x_intersect], [line_y], 'kx', markersize=6)
            
            # Draw perpendicular from P to L
            ax.plot([0, 0], [line_y, point_y], 'g--', linewidth=1.5, label='Perpendicular')
            ax.text(0.1, (line_y + point_y)/2, f'd = {abs(point_y - line_y):.1f}', 
                   fontsize=12, color='green')
            
            # Add text explaining Playfair's axiom
            ax.text(3, 3, "Playfair's Axiom:\n"
                         "Given a line L and a point P not on it,\n"
                         "there exists exactly one line through P\n"
                         "that is parallel to L.", 
                     fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            # Set limits and labels
            ax.set_xlim(-5, 5)
            ax.set_ylim(-3, 5)
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title("Playfair's Axiom - Interactive Demo")
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            fig.canvas.draw_idle()
        
        # Add slider for moving the line
        ax_line = plt.axes([0.2, 0.02, 0.65, 0.03])
        line_slider = Slider(ax_line, 'Line Y Position', -3, 3, valinit=line_y)
        
        # Add slider for moving the point
        ax_point = plt.axes([0.2, 0.06, 0.65, 0.03])
        point_slider = Slider(ax_point, 'Point Y Position', -3, 5, valinit=point_y)
        
        # Connect the sliders to the update function
        def update_line(val):
            nonlocal line_y
            line_y = val
            update(None)
            
        def update_point(val):
            nonlocal point_y
            point_y = val
            update(None)
        
        line_slider.on_changed(update_line)
        point_slider.on_changed(update_point)
        
        # Initial draw
        update(None)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        return fig, line_slider, point_slider
    
    def euclidean_proof(self, ax=None):
        """
        Visualize a proof of Playfair's axiom in Euclidean geometry.
        
        Args:
            ax: Matplotlib axis to draw on
            
        Returns:
            The matplotlib axis with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw the given line L
        ax.plot([-5, 5], [0, 0], 'k-', linewidth=2, label='Line L')
        
        # Draw the external point P
        point_y = 2
        ax.plot([0], [point_y], 'ko', markersize=8, label='Point P')
        ax.text(0.2, point_y + 0.2, 'P', fontsize=12)
        
        # Draw the perpendicular from P to L
        ax.plot([0, 0], [0, point_y], 'g-', linewidth=1.5, label='Perpendicular')
        
        # Mark the foot of the perpendicular
        ax.plot([0], [0], 'go', markersize=6)
        ax.text(0.2, -0.2, 'Q', fontsize=12, color='green')
        
        # Draw the parallel line through P
        ax.plot([-5, 5], [point_y, point_y], 'b-', linewidth=2, label='m (Parallel to L)')
        
        # Assume a different line n through P that's parallel to L
        angle = -15  # degrees
        angle_rad = angle * np.pi / 180
        
        # Calculate the line endpoints
        dx = 5 * np.cos(angle_rad)
        dy = 5 * np.sin(angle_rad)
        
        # Draw the hypothetical second parallel
        ax.plot([0 - dx, 0 + dx], [point_y - dy, point_y + dy], 'r--', linewidth=2, 
               label='n (Hypothetical second parallel)')
        
        # Mark the angle between the perpendicular and the second line
        arc = Arc((0, point_y), 0.5, 0.5, 
                 theta1=90, theta2=90+angle, 
                 color='red', lw=1.5)
        ax.add_patch(arc)
        ax.text(0.25, point_y+0.25, 'θ', color='red', fontsize=12)
        
        # Calculate intersection with the x-axis
        if dy != 0:  # Avoid division by zero
            t = -point_y / dy
            x_intersect = t * dx
            
            # Highlight the intersection point
            ax.plot([x_intersect], [0], 'rx', markersize=10)
            ax.text(x_intersect+0.2, -0.2, 'R', color='red', fontsize=12)
            
            # Add contradiction text
            ax.text(x_intersect, -1, "Contradiction!\nLine n intersects L at point R\n"
                               "Therefore, n is not parallel to L", 
                   fontsize=12, bbox=dict(facecolor='mistyrose', alpha=0.7), ha='center')
        
        # Add proof explanation
        ax.text(3, 3, "Proof by Contradiction:\n\n"
                     "1. Let m be the line through P parallel to L\n"
                     "   (constructed by making PQ perpendicular to L)\n\n"
                     "2. Assume another line n through P is also parallel to L\n\n"
                     "3. If n makes angle θ≠0 with the perpendicular,\n"
                     "   it must intersect L at some point R\n\n"
                     "4. Therefore, n is not parallel to L (contradiction)\n\n"
                     "5. Thus, exactly one parallel to L passes through P",
               fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Set limits and labels
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1.5, 5)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("Proof of Playfair's Axiom in Euclidean Geometry")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return ax
    
    def visualize_parallel_transport_comparison(self, ax=None):
        """
        Compare parallel transport in different geometries to highlight
        the connection to Playfair's axiom.
        
        Args:
            ax: Matplotlib axis to draw on
            
        Returns:
            The matplotlib figure with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 5))
            
        # Set up the figure with 3 subplots for different geometries
        if hasattr(ax, 'get_figure'):
            fig = ax.get_figure()
            fig.clear()
            
            # Create 3 subplots
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133, projection='3d')
        else:
            # If ax is None or not a valid axis
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133, projection='3d')
        
        # 1. Euclidean parallel transport (no change in vector)
        self._draw_euclidean_transport(ax1)
        
        # 2. Hyperbolic parallel transport
        self._draw_hyperbolic_transport(ax2)
        
        # 3. Spherical parallel transport
        self._draw_spherical_transport(ax3)
        
        plt.tight_layout()
        fig.suptitle("Parallel Transport in Different Geometries", fontsize=16, y=1.05)
        
        return fig
    
    def _draw_euclidean_transport(self, ax):
        """Helper method to draw parallel transport in Euclidean geometry."""
        # Draw a grid for reference
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Define a closed path
        t = np.linspace(0, 2*np.pi, 100)
        x = 2 * np.cos(t)
        y = 1.5 * np.sin(t)
        
        # Draw the path
        ax.plot(x, y, 'k-', linewidth=2, label='Path')
        
        # Draw arrows along the path to indicate direction
        for i in range(0, len(t), 10):
            if i < len(t) - 1:
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    dx /= length
                    dy /= length
                    
                    ax.arrow(x[i], y[i], dx*0.2, dy*0.2, 
                            head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.7)
        
        # Draw the parallel transported vector at different points
        vector_length = 0.5
        vector_angle = 30  # degrees
        
        # Convert to radians
        vector_angle_rad = vector_angle * np.pi / 180
        
        # Calculate vector components
        vx = vector_length * np.cos(vector_angle_rad)
        vy = vector_length * np.sin(vector_angle_rad)
        
        # Draw vectors at intervals along the path
        for i in range(0, len(t), 20):
            ax.arrow(x[i], y[i], vx, vy, 
                    head_width=0.1, head_length=0.15, fc=self.euclidean_color, ec=self.euclidean_color)
        
        # Add explanation
        ax.text(0, -2.5, "Euclidean Geometry:\n"
                        "Parallel transport preserves vector direction.\n"
                        "This is consistent with Playfair's Axiom.", 
               fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title("Euclidean Parallel Transport")
        
        return ax
    
    def _draw_hyperbolic_transport(self, ax):
        """Helper method to draw parallel transport in hyperbolic geometry."""
        # Draw Poincaré disk
        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Draw a path in the hyperbolic plane
        # We'll use a circular arc inside the Poincaré disk
        t = np.linspace(0, 2*np.pi, 100)
        center_x, center_y = 0.3, 0.2
        radius = 0.4
        
        path_x = center_x + radius * np.cos(t)
        path_y = center_y + radius * np.sin(t)
        
        # Ensure all points are inside the disk
        inside_disk = [(x**2 + y**2 < 0.95**2) for x, y in zip(path_x, path_y)]
        path_x = path_x[inside_disk]
        path_y = path_y[inside_disk]
        
        # Draw the path
        ax.plot(path_x, path_y, 'k-', linewidth=2, label='Path')
        
        # Draw arrows along the path to indicate direction
        for i in range(0, len(path_x), 10):
            if i < len(path_x) - 1:
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    dx /= length
                    dy /= length
                    
                    ax.arrow(path_x[i], path_y[i], dx*0.05, dy*0.05, 
                            head_width=0.03, head_length=0.03, fc='k', ec='k', alpha=0.7)
        
        # Draw vectors along the path with angular deviation
        # In hyperbolic geometry, parallel transport along a closed path
        # results in a rotated vector
        
        vector_length = 0.2
        initial_angle = 30  # degrees
        
        # Total angular deviation for the full path (simplification for visualization)
        total_deviation = 50  # degrees
        
        # Draw vectors at intervals
        num_vectors = 5
        for i in range(num_vectors):
            # Calculate position on path
            idx = int(i * len(path_x) / num_vectors)
            
            # Calculate angle for this position
            current_angle = initial_angle + i * total_deviation / (num_vectors - 1)
            angle_rad = current_angle * np.pi / 180
            
            # Calculate vector components
            vx = vector_length * np.cos(angle_rad)
            vy = vector_length * np.sin(angle_rad)
            
            # Draw the vector
            ax.arrow(path_x[idx], path_y[idx], vx, vy, 
                    head_width=0.05, head_length=0.07, fc=self.hyperbolic_color, ec=self.hyperbolic_color)
            
            # Add angle label for first and last vector
            if i == 0:
                ax.text(path_x[idx] + vx + 0.05, path_y[idx] + vy, 
                       f"{initial_angle}°", fontsize=8, color=self.hyperbolic_color)
            elif i == num_vectors - 1:
                ax.text(path_x[idx] + vx + 0.05, path_y[idx] + vy, 
                       f"{initial_angle + total_deviation}°", fontsize=8, color=self.hyperbolic_color)
        
        # Add explanation
        ax.text(0, -1.3, "Hyperbolic Geometry:\n"
                        "Parallel transport along a closed path\n"
                        "results in a rotated vector.\n"
                        "This is inconsistent with Playfair's Axiom.", 
               fontsize=11, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title("Hyperbolic Parallel Transport")
        
        return ax
    
    def _draw_spherical_transport(self, ax):
        """Helper method to draw parallel transport on a sphere."""
        # Create a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
        
        # Define a path on the sphere (we'll use a latitude circle)
        t = np.linspace(0, 2*np.pi, 100)
        latitude = np.pi/4  # 45 degrees north
        
        path_x = np.cos(t) * np.cos(latitude)
        path_y = np.sin(t) * np.cos(latitude)
        path_z = np.sin(latitude) * np.ones_like(t)
        
        # Draw the path
        ax.plot(path_x, path_y, path_z, 'k-', linewidth=2, label='Path')
        
        # Draw arrows along the path to indicate direction
        for i in range(0, len(t), 15):
            if i < len(t) - 1:
                # Calculate tangent vector to the path
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                dz = path_z[i+1] - path_z[i]
                length = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if length > 0:
                    dx /= length
                    dy /= length
                    dz /= length
                    
                    ax.quiver(path_x[i], path_y[i], path_z[i], 
                             dx*0.1, dy*0.1, dz*0.1, 
                             color='k', alpha=0.7)
        
        # Draw parallel transported vector
        # In spherical geometry, parallel transport around a closed path
        # also results in a rotated vector
        
        vector_length = 0.3
        initial_angle = 0  # Start with vector pointing "north"
        
        # Total angular deviation (simplification for visualization)
        total_deviation = 60  # degrees
        
        # Draw vectors at intervals
        num_vectors = 5
        for i in range(num_vectors):
            # Calculate position on path
            idx = int(i * len(t) / num_vectors)
            
            # Calculate angle for this position
            current_angle = initial_angle + i * total_deviation / (num_vectors - 1)
            angle_rad = current_angle * np.pi / 180
            
            # For sphere, we need the local tangent space
            # Normal vector at this point on the sphere
            normal = [path_x[idx], path_y[idx], path_z[idx]]
            
            # Tangent vector in the direction of increasing longitude
            tangent1 = [-path_y[idx], path_x[idx], 0]
            length1 = np.sqrt(sum(t**2 for t in tangent1))
            tangent1 = [t/length1 for t in tangent1]
            
            # Tangent vector in the direction of increasing latitude
            tangent2 = [
                -path_z[idx] * path_x[idx],
                -path_z[idx] * path_y[idx],
                path_x[idx]**2 + path_y[idx]**2
            ]
            length2 = np.sqrt(sum(t**2 for t in tangent2))
            tangent2 = [t/length2 for t in tangent2]
            
            # Calculate vector components in the local tangent space
            vector = [
                tangent1[j] * np.cos(angle_rad) + tangent2[j] * np.sin(angle_rad) 
                for j in range(3)
            ]
            
            # Scale vector
            vector = [v * vector_length for v in vector]
            
            # Draw the vector
            ax.quiver(path_x[idx], path_y[idx], path_z[idx], 
                     vector[0], vector[1], vector[2], 
                     color=self.elliptic_color, arrow_length_ratio=0.3)
            
            # Add label for first and last vector
            if i == 0 or i == num_vectors - 1:
                angle_text = f"{initial_angle + i * total_deviation / (num_vectors - 1):.0f}°"
                ax.text(path_x[idx] + vector[0]*1.2, 
                       path_y[idx] + vector[1]*1.2, 
                       path_z[idx] + vector[2]*1.2, 
                       angle_text, fontsize=10, color=self.elliptic_color)
        
        # Add explanation
        ax.text(0, 0, -1.5, "Elliptic/Spherical Geometry:\n"
                          "Parallel transport along a closed path\n"
                          "results in a rotated vector.\n"
                          "This is inconsistent with Playfair's Axiom.", 
               fontsize=11, bbox=dict(facecolor='white', alpha=0.7), ha='center')
        
        # Set limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title("Spherical Parallel Transport")
        
        # Set view angle
        ax.view_init(elev=20, azim=30)
        
        return ax


def historical_context():
    """Create a figure showing the historical context of Playfair's axiom."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, "HISTORICAL CONTEXT OF PLAYFAIR'S AXIOM", 
           fontsize=18, weight='bold', ha='center', va='top')
    
    # Timeline
    timeline_y = 0.7
    ax.axhline(y=timeline_y, xmin=0.1, xmax=0.9, color='black', linewidth=2)
    
    # Key events
    events = [
        (-300, "Euclid's Elements\nOriginal fifth postulate"),
        (830, "Al-Kindi\nEarly attempts to prove\nthe fifth postulate"),
        (1300, "Nasir al-Din al-Tusi\nWork on the theory of parallels"),
        (1733, "Girolamo Saccheri\nNearly discovered\nnon-Euclidean geometry"),
        (1795, "John Playfair\nRestated Euclid's fifth\npostulate in simpler form"),
        (1830, "Lobachevsky & Bolyai\nIndependently developed\nhyperbolic geometry"),
        (1854, "Bernhard Riemann\nDeveloped elliptic geometry\nand curved spaces"),
        (1915, "Einstein's General Relativity\nUsed non-Euclidean geometry\nto describe gravity")
    ]
    
    # Add events to timeline
    min_year = min(events, key=lambda x: x[0])[0]
    max_year = max(events, key=lambda x: x[0])[0]
    year_range = max_year - min_year
    
    for year, description in events:
        # Calculate position on timeline
        pos = 0.1 + 0.8 * (year - min_year) / year_range
        
        # Draw marker
        ax.scatter([pos], [timeline_y], s=80, color='blue', zorder=3)
        
        # Draw year label
        if year < 0:
            year_str = f"{abs(year)} BCE"
        else:
            year_str = f"{year} CE"
            
        ax.text(pos, timeline_y - 0.03, year_str, ha='center', va='top', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.7), zorder=3)
        
        # Draw description
        # Alternate above and below
        if events.index((year, description)) % 2 == 0:
            ax.text(pos, timeline_y + 0.05, description, ha='center', va='bottom', fontsize=11,
                   bbox=dict(facecolor='lightyellow', alpha=0.7, boxstyle='round'))
            # Draw connector line
            ax.plot([pos, pos], [timeline_y, timeline_y + 0.04], 'k-', linewidth=1)
        else:
            ax.text(pos, timeline_y - 0.15, description, ha='center', va='top', fontsize=11,
                   bbox=dict(facecolor='lightyellow', alpha=0.7, boxstyle='round'))
            # Draw connector line
            ax.plot([pos, pos], [timeline_y, timeline_y - 0.04], 'k-', linewidth=1)
    
    # Add explanatory text blocks
    # Euclid's original formulation
    euclid_text = (
        "Euclid's Fifth Postulate (Original):\n\n"
        "\"If a straight line falling on two straight lines makes the interior angles on the "
        "same side less than two right angles, the two straight lines, if produced indefinitely, "
        "meet on that side on which the angles are less than two right angles.\"\n\n"
        "This complex formulation was seen as less intuitive than the other postulates, "
        "leading mathematicians to attempt to prove it from the other four for over 2000 years."
    )
    ax.text(0.2, 0.4, euclid_text, fontsize=10, ha='left', va='top',
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
    
    # Playfair's formulation
    playfair_text = (
        "Playfair's Axiom (1795):\n\n"
        "\"Given a line and a point not on the line, at most one line can be drawn through "
        "the point parallel to the given line.\"\n\n"
        "This clearer formulation by John Playfair appeared in his textbook 'Elements of Geometry' "
        "and quickly became the standard way to express the parallel postulate. It is logically "
        "equivalent to Euclid's original formulation but more intuitive."
    )
    ax.text(0.6, 0.4, playfair_text, fontsize=10, ha='left', va='top',
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
    
    # Non-Euclidean implications
    non_euclidean_text = (
        "Implications for Non-Euclidean Geometry:\n\n"
        "The realization that the parallel postulate is independent of the other axioms led to "
        "the development of non-Euclidean geometries:\n\n"
        "• Hyperbolic Geometry (Lobachevsky, Bolyai): Given a line and a point not on it, "
        "infinitely many lines can be drawn through the point parallel to the given line.\n\n"
        "• Elliptic Geometry (Riemann): No parallel lines exist; all lines eventually intersect.\n\n"
        "These discoveries revolutionized mathematics and later provided Einstein with the "
        "mathematical framework needed for general relativity."
    )
    ax.text(0.5, 0.15, non_euclidean_text, fontsize=10, ha='center', va='top',
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
    
    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate Playfair's axiom."""
    # Create the PlayfairAxiom instance
    playfair = PlayfairAxiom()
    
    print("Visualizing Playfair's Axiom and Related Concepts")
    print("-" * 60)
    
    # Basic visualization
    print("1. Creating basic visualization of Playfair's Axiom...")
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    playfair.visualize_playfair_axiom(ax1)
    
    # Equivalent forms of the parallel postulate
    print("2. Visualizing equivalent forms of the parallel postulate...")
    fig2 = playfair.visualize_parallel_postulate_forms()
    
    # Comparison in different geometries
    print("3. Comparing Playfair's axiom in different geometries...")
    fig3 = playfair.visualize_non_euclidean_comparison()
    
    # Triangle angles comparison
    print("4. Comparing triangle angles in different geometries...")
    fig4 = playfair.visualize_triangle_angles()
    
    # Historical context
    print("5. Creating historical context visualization...")
    fig5 = historical_context()
    
    # Proof of Playfair's axiom
    print("6. Visualizing a proof of Playfair's axiom...")
    fig6 = plt.figure(figsize=(10, 8))
    ax6 = fig6.add_subplot(111)
    playfair.euclidean_proof(ax6)
    
    # Parallel transport comparison
    print("7. Comparing parallel transport in different geometries...")
    fig7 = playfair.visualize_parallel_transport_comparison()
    
    # Interactive demonstration (commented out for simplicity)
    # print("8. Creating interactive demonstration...")
    # fig8, slider1, slider2 = playfair.interactive_demo()
    
    print("\nAll visualizations created successfully!")
    print("Displaying figures...")
    
    plt.show()


if __name__ == "__main__":
    main()