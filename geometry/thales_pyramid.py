import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HeightCalculator:
    """
    Class for calculating heights of objects like pyramids using various methods:
    - Shadow method
    - Angle of elevation method
    - Triangulation method
    """
    
    def __init__(self):
        self.earth_radius = 6371 * 1000  # Earth radius in meters
    
    def height_from_shadow(self, object_shadow_length, reference_height, reference_shadow_length):
        """
        Calculate height using the shadow method and similar triangles.
        
        Args:
            object_shadow_length: Length of the shadow cast by the object (e.g., pyramid)
            reference_height: Height of a reference object (e.g., measuring stick)
            reference_shadow_length: Length of the shadow cast by the reference object
            
        Returns:
            Estimated height of the object
        """
        if reference_shadow_length == 0:
            raise ValueError("Reference shadow length cannot be zero")
            
        # Using similar triangles: h1/s1 = h2/s2
        return (reference_height * object_shadow_length) / reference_shadow_length
    
    def height_from_angle(self, distance, angle_degrees):
        """
        Calculate height using distance and angle of elevation.
        
        Args:
            distance: Horizontal distance from the observer to the object
            angle_degrees: Angle of elevation from the observer to the top of the object
            
        Returns:
            Height of the object
        """
        # Convert angle to radians
        angle_radians = math.radians(angle_degrees)
        
        # h = d * tan(θ)
        return distance * math.tan(angle_radians)
    
    def height_from_two_angles(self, distance_between_observations, angle1_degrees, angle2_degrees):
        """
        Calculate height using two observation points with different angles of elevation.
        
        Args:
            distance_between_observations: Distance between the two observation points
            angle1_degrees: Angle of elevation from first observation point
            angle2_degrees: Angle of elevation from second observation point
            
        Returns:
            Height of the object and distance from first observation point
        """
        # Convert angles to radians
        angle1 = math.radians(angle1_degrees)
        angle2 = math.radians(angle2_degrees)
        
        # Calculate distance from first observation point to the base of the object
        distance = (distance_between_observations * math.tan(angle2)) / (math.tan(angle1) - math.tan(angle2))
        
        # Calculate height using the distance and first angle
        height = distance * math.tan(angle1)
        
        return height, distance
    
    def pyramid_height_from_base_and_slant(self, base_length, slant_height, is_square=True):
        """
        Calculate the height of a pyramid given its base length and slant height.
        
        Args:
            base_length: Length of the base side
            slant_height: Length from apex to middle of a base side
            is_square: Whether the base is a square (True) or equilateral triangle (False)
            
        Returns:
            Height of the pyramid
        """
        if is_square:
            # For square base: h² = s² - (b/2)²
            half_base = base_length / 2
            return math.sqrt(slant_height**2 - half_base**2)
        else:
            # For triangular base: h² = s² - (b²/12)
            return math.sqrt(slant_height**2 - (base_length**2 / 12))
    
    def pyramid_height_from_volume(self, volume, base_area):
        """
        Calculate the height of a pyramid given its volume and base area.
        
        Args:
            volume: Volume of the pyramid
            base_area: Area of the base
            
        Returns:
            Height of the pyramid
        """
        # For any pyramid: V = (1/3) × base_area × height
        return 3 * volume / base_area
    
    def truncated_pyramid_height(self, lower_length, upper_length, slant_height, is_square=True):
        """
        Calculate the height of a truncated pyramid (frustum).
        
        Args:
            lower_length: Length of the lower base side
            upper_length: Length of the upper base side
            slant_height: Slant height of the truncated pyramid
            is_square: Whether the bases are squares (True) or equilateral triangles (False)
            
        Returns:
            Height of the truncated pyramid
        """
        if is_square:
            # For square base
            width_diff = (lower_length - upper_length) / 2
            return math.sqrt(slant_height**2 - width_diff**2)
        else:
            # For triangular base - more complex
            # This is an approximation using the average difference
            width_diff = (lower_length - upper_length) / 2
            return math.sqrt(slant_height**2 - width_diff**2)


class DistanceCalculator:
    """
    Class for calculating distances to objects like ships using various methods:
    - Angle depression method
    - Horizon and curvature calculations
    - Triangulation
    """
    
    def __init__(self):
        self.earth_radius = 6371 * 1000  # Earth radius in meters
    
    def distance_from_angle(self, observer_height, angle_degrees):
        """
        Calculate distance to an object using the observer's height and angle of depression.
        
        Args:
            observer_height: Height of the observer above reference level (e.g., sea level)
            angle_degrees: Angle of depression from the observer to the object
            
        Returns:
            Distance to the object
        """
        # Convert angle to radians (note: angle of depression is measured downward)
        angle_radians = math.radians(angle_degrees)
        
        # d = h / tan(θ)
        return observer_height / math.tan(angle_radians)
    
    def horizon_distance(self, observer_height):
        """
        Calculate the theoretical horizon distance, taking Earth's curvature into account.
        
        Args:
            observer_height: Height of the observer above sea level in meters
            
        Returns:
            Distance to the horizon in meters
        """
        # Formula: d = sqrt(2 * R * h), where R is Earth's radius and h is observer height
        return math.sqrt(2 * self.earth_radius * observer_height)
    
    def ship_visibility_distance(self, observer_height, ship_height):
        """
        Calculate the maximum distance at which a ship becomes visible,
        taking into account Earth's curvature.
        
        Args:
            observer_height: Height of the observer above sea level in meters
            ship_height: Height of the ship (portion above water) in meters
            
        Returns:
            Maximum visibility distance in meters
        """
        # Formula: d = sqrt(2 * R * h1) + sqrt(2 * R * h2)
        observer_horizon = math.sqrt(2 * self.earth_radius * observer_height)
        ship_horizon = math.sqrt(2 * self.earth_radius * ship_height)
        
        return observer_horizon + ship_horizon
    
    def distance_by_triangulation(self, baseline, angle1_degrees, angle2_degrees):
        """
        Calculate distance to an object using triangulation.
        
        Args:
            baseline: Length of the baseline (distance between two observation points)
            angle1_degrees: Angle from first observation point to the object (from baseline)
            angle2_degrees: Angle from second observation point to the object (from baseline)
            
        Returns:
            Distance from the baseline to the object
        """
        # Convert angles to radians
        angle1 = math.radians(angle1_degrees)
        angle2 = math.radians(angle2_degrees)
        
        # Calculate total angle in the triangle
        total_angle = math.pi - (angle1 + angle2)
        
        if total_angle <= 0:
            raise ValueError("Invalid angles: sum of angles must be less than 180 degrees")
        
        # Using law of sines to find the distance
        # baseline / sin(total_angle) = distance / sin(angle1)
        return (baseline * math.sin(angle2)) / math.sin(total_angle)
    
    def distance_from_lighthouse(self, lighthouse_height, visible_height, ship_height=0):
        """
        Calculate the distance of a ship from a lighthouse using the visible height method.
        
        Args:
            lighthouse_height: Total height of the lighthouse in meters
            visible_height: Portion of the lighthouse visible from the ship in meters
            ship_height: Height of the observation point on the ship in meters
            
        Returns:
            Distance from the lighthouse to the ship in meters
        """
        # Height hidden by Earth's curvature
        hidden_height = lighthouse_height - visible_height
        
        # Formula based on Earth's curvature: d² = 2 * R * h
        if hidden_height <= 0:
            return 0  # Lighthouse fully visible
            
        # Distance where the hidden_height would be just below the horizon
        distance_squared = 2 * self.earth_radius * hidden_height
        
        # Adjust for observer height on the ship
        if ship_height > 0:
            ship_horizon = math.sqrt(2 * self.earth_radius * ship_height)
            return math.sqrt(distance_squared) - ship_horizon
        
        return math.sqrt(distance_squared)


class GeometryVisualizer:
    """
    Class for visualizing geometric problems related to heights and distances.
    """
    
    def visualize_pyramid_shadow(self, pyramid_height, pyramid_shadow, stick_height, stick_shadow):
        """
        Visualize the pyramid height calculation using the shadow method.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw the reference stick and its shadow
        ax.plot([0, 0], [0, stick_height], 'b-', linewidth=3, label='Reference Stick')
        ax.plot([0, stick_shadow], [0, 0], 'b--', linewidth=2, label='Stick Shadow')
        
        # Draw the pyramid and its shadow
        pyramid_x = stick_shadow + 5  # Offset for clarity
        ax.plot([pyramid_x, pyramid_x], [0, pyramid_height], 'r-', linewidth=4, label='Pyramid')
        ax.plot([pyramid_x, pyramid_x + pyramid_shadow], [0, 0], 'r--', linewidth=2, label='Pyramid Shadow')
        
        # Draw the sun rays
        sun_angle = math.atan2(stick_height, stick_shadow)
        ray_length = max(stick_height, pyramid_height) * 1.2
        
        ax.plot([stick_shadow, 0], [0, stick_height], 'y-', alpha=0.5)
        ax.plot([pyramid_x + pyramid_shadow, pyramid_x], [0, pyramid_height], 'y-', alpha=0.5)
        
        # Set axis properties
        ax.set_xlim(-2, pyramid_x + pyramid_shadow + 2)
        ax.set_ylim(0, max(stick_height, pyramid_height) * 1.1)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Height')
        ax.set_title('Pyramid Height Calculation Using Shadow Method')
        ax.grid(True)
        ax.legend()
        
        # Add height calculation annotation
        height_text = f"Pyramid Height = {pyramid_height:.2f} units\n"
        height_text += f"(Calculation: {stick_height} × {pyramid_shadow} ÷ {stick_shadow})"
        ax.annotate(height_text, xy=(pyramid_x + 1, pyramid_height/2),
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def visualize_pyramid_angle(self, distance, height, angle_degrees):
        """
        Visualize the pyramid height calculation using the angle of elevation method.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw ground
        ax.plot([-2, distance + 2], [0, 0], 'k-', linewidth=1)
        
        # Draw the pyramid
        ax.plot([distance, distance], [0, height], 'r-', linewidth=4, label='Pyramid')
        
        # Draw the observation line
        ax.plot([0, distance], [0, 0], 'g-', linewidth=2, label='Baseline Distance')
        ax.plot([0, distance], [0, height], 'b-', linewidth=2, label='Line of Sight')
        
        # Mark angle
        angle_radius = distance * 0.2
        angle_rad = math.radians(angle_degrees)
        
        # Draw angle arc
        theta = np.linspace(0, angle_rad, 100)
        x = angle_radius * np.cos(theta)
        y = angle_radius * np.sin(theta)
        ax.plot(x, y, 'g-', linewidth=1.5)
        
        # Label the angle
        angle_label_x = angle_radius * 0.7 * math.cos(angle_rad/2)
        angle_label_y = angle_radius * 0.7 * math.sin(angle_rad/2)
        ax.text(angle_label_x, angle_label_y, f"{angle_degrees}°", fontsize=12,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        
        # Set axis properties
        ax.set_xlim(-2, distance + 2)
        ax.set_ylim(0, height * 1.1)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Height')
        ax.set_title('Pyramid Height Calculation Using Angle of Elevation')
        ax.grid(True)
        ax.legend(loc='upper left')
        
        # Add height calculation annotation
        height_text = f"Pyramid Height = {height:.2f} units\n"
        height_text += f"(Calculation: {distance} × tan({angle_degrees}°))"
        ax.annotate(height_text, xy=(distance/2, height * 0.7),
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def visualize_ship_distance(self, observer_height, ship_distance, earth_radius=6371e3):
        """
        Visualize the distance to a ship taking into account Earth's curvature.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw Earth's curve
        angle_range = np.linspace(-0.1, 0.1, 1000)
        earth_x = earth_radius * np.sin(angle_range)
        earth_y = earth_radius * np.cos(angle_range) - earth_radius
        
        ax.plot(earth_x, earth_y, 'b-', linewidth=2, label='Earth Surface')
        
        # Draw observer
        ax.plot([0], [observer_height], 'ro', markersize=8, label='Observer')
        
        # Calculate ship position on Earth's surface
        ship_angle = ship_distance / earth_radius
        ship_x = earth_radius * np.sin(ship_angle)
        ship_y = earth_radius * np.cos(ship_angle) - earth_radius
        
        # Draw ship
        ship_height = 30  # Example ship height above water
        ax.plot([ship_x], [ship_y + ship_height], 'go', markersize=8, label='Ship')
        ax.plot([ship_x], [ship_y], 'bx', markersize=6)
        ax.plot([ship_x, ship_x], [ship_y, ship_y + ship_height], 'g-', linewidth=2)
        
        # Draw line of sight
        ax.plot([0, ship_x], [observer_height, ship_y + ship_height], 'r--', linewidth=1.5, label='Line of Sight')
        
        # Draw tangent line (horizon)
        horizon_distance = math.sqrt(2 * earth_radius * observer_height)
        horizon_angle = horizon_distance / earth_radius
        horizon_x = earth_radius * np.sin(horizon_angle)
        horizon_y = earth_radius * np.cos(horizon_angle) - earth_radius
        
        ax.plot([0, horizon_x], [observer_height, horizon_y], 'k--', linewidth=1.5, label='Horizon Line')
        ax.plot([horizon_x], [horizon_y], 'kx', markersize=6)
        
        # Set axis limits to zoom in on the relevant portion
        ax_range = max(ship_distance * 1.2, horizon_distance * 1.2)
        ax.set_xlim(-ax_range * 0.2, ax_range)
        ax.set_ylim(-ax_range * 0.1, observer_height * 1.5)
        
        # Labels and title
        ax.set_xlabel('Distance (meters)')
        ax.set_ylabel('Height (meters)')
        ax.set_title('Ship Distance Visualization with Earth Curvature')
        ax.grid(True)
        ax.legend()
        
        # Add distance annotations
        horizon_text = f"Horizon Distance: {horizon_distance:.2f} meters"
        ship_text = f"Ship Distance: {ship_distance:.2f} meters"
        if ship_distance > horizon_distance:
            ship_text += " (Beyond Horizon)"
        else:
            ship_text += " (Visible)"
            
        ax.annotate(horizon_text, xy=(horizon_distance/2, observer_height/2),
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        ax.annotate(ship_text, xy=(ship_distance/2, observer_height),
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def visualize_3d_pyramid(self, base_length, height, is_square=True):
        """
        Create a 3D visualization of a pyramid.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if is_square:
            # Square base pyramid
            # Base vertices
            half_base = base_length / 2
            base_points = [
                [-half_base, -half_base, 0],
                [half_base, -half_base, 0],
                [half_base, half_base, 0],
                [-half_base, half_base, 0]
            ]
            
            # Draw base
            base_x = [p[0] for p in base_points + [base_points[0]]]
            base_y = [p[1] for p in base_points + [base_points[0]]]
            base_z = [p[2] for p in base_points + [base_points[0]]]
            ax.plot(base_x, base_y, base_z, 'b-', linewidth=2)
            
            # Apex
            apex = [0, 0, height]
            
            # Draw edges from base to apex
            for point in base_points:
                ax.plot([point[0], apex[0]], [point[1], apex[1]], [point[2], apex[2]], 'r-', linewidth=2)
                
            # Create polygon faces
            for i in range(4):
                face_x = [base_points[i][0], base_points[(i+1)%4][0], apex[0]]
                face_y = [base_points[i][1], base_points[(i+1)%4][1], apex[1]]
                face_z = [base_points[i][2], base_points[(i+1)%4][2], apex[2]]
                ax.plot_trisurf(face_x, face_y, face_z, color='gold', alpha=0.3)
                
        else:
            # Triangular base pyramid
            # Base vertices for equilateral triangle
            base_points = [
                [0, base_length/(2*math.sqrt(3)), 0],
                [-base_length/2, -base_length/(2*math.sqrt(3)), 0],
                [base_length/2, -base_length/(2*math.sqrt(3)), 0]
            ]
            
            # Draw base
            base_x = [p[0] for p in base_points + [base_points[0]]]
            base_y = [p[1] for p in base_points + [base_points[0]]]
            base_z = [p[2] for p in base_points + [base_points[0]]]
            ax.plot(base_x, base_y, base_z, 'b-', linewidth=2)
            
            # Apex
            apex = [0, 0, height]
            
            # Draw edges from base to apex
            for point in base_points:
                ax.plot([point[0], apex[0]], [point[1], apex[1]], [point[2], apex[2]], 'r-', linewidth=2)
                
            # Create polygon faces
            for i in range(3):
                face_x = [base_points[i][0], base_points[(i+1)%3][0], apex[0]]
                face_y = [base_points[i][1], base_points[(i+1)%3][1], apex[1]]
                face_z = [base_points[i][2], base_points[(i+1)%3][2], apex[2]]
                ax.plot_trisurf(face_x, face_y, face_z, color='gold', alpha=0.3)
        
        # Set axis properties
        max_range = max(base_length, height)
        ax.set_xlim(-max_range/1.5, max_range/1.5)
        ax.set_ylim(-max_range/1.5, max_range/1.5)
        ax.set_zlim(0, height * 1.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_title('3D Pyramid Visualization')
        
        # Add dimensions
        ax.text(0, 0, height/2, f"Height: {height}", color='black')
        if is_square:
            ax.text(0, -half_base*1.1, 0, f"Base: {base_length} × {base_length}", color='black')
        else:
            ax.text(0, 0, 0, f"Base Side: {base_length}", color='black')
            
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create calculators
    height_calc = HeightCalculator()
    distance_calc = DistanceCalculator()
    visualizer = GeometryVisualizer()
    
    # Example 1: Calculate pyramid height using shadow method
    stick_height = 2  # meters
    stick_shadow = 3  # meters
    pyramid_shadow = 120  # meters
    
    pyramid_height = height_calc.height_from_shadow(
        pyramid_shadow, stick_height, stick_shadow
    )
    print(f"Pyramid height (shadow method): {pyramid_height:.2f} meters")
    
    # Example 2: Calculate pyramid height using angle of elevation
    distance_to_pyramid = 200  # meters
    angle_of_elevation = 28  # degrees
    
    pyramid_height2 = height_calc.height_from_angle(
        distance_to_pyramid, angle_of_elevation
    )
    print(f"Pyramid height (angle method): {pyramid_height2:.2f} meters")
    
    # Example 3: Calculate ship distance
    lighthouse_height = 50  # meters
    observer_height = 20  # meters
    
    horizon_distance = distance_calc.horizon_distance(observer_height)
    print(f"Distance to horizon: {horizon_distance:.2f} meters")
    
    # Calculate ship visibility with height of 10 meters
    ship_height = 10  # meters
    visibility_distance = distance_calc.ship_visibility_distance(observer_height, ship_height)
    print(f"Maximum visibility distance for ship: {visibility_distance:.2f} meters")
    
    # Example 4: Calculate distance using triangulation
    baseline = 100  # meters
    angle1 = 32  # degrees
    angle2 = 28  # degrees
    
    triangulation_distance = distance_calc.distance_by_triangulation(baseline, angle1, angle2)
    print(f"Distance by triangulation: {triangulation_distance:.2f} meters")
    
    # Example 5: Calculate height of a truncated pyramid
    lower_base = 20  # meters
    upper_base = 10  # meters
    slant_height = 15  # meters
    
    truncated_height = height_calc.truncated_pyramid_height(lower_base, upper_base, slant_height)
    print(f"Truncated pyramid height: {truncated_height:.2f} meters")
    
    # Visualizations
    vis_fig1 = visualizer.visualize_pyramid_shadow(pyramid_height, pyramid_shadow, stick_height, stick_shadow)
    vis_fig2 = visualizer.visualize_pyramid_angle(distance_to_pyramid, pyramid_height2, angle_of_elevation)
    vis_fig3 = visualizer.visualize_ship_distance(observer_height, visibility_distance / 2)
    vis_fig4 = visualizer.visualize_3d_pyramid(50, 80, is_square=True)
    
    plt.show()