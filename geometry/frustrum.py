import math

class TruncatedPyramid:
    """
    A class representing a truncated pyramid (frustum) with regular polygonal bases.
    
    Attributes:
        lower_side_length (float): Length of the side of the lower base
        upper_side_length (float): Length of the side of the upper base
        height (float): Height of the truncated pyramid
        n_sides (int): Number of sides in the polygonal bases (default is 4 for square)
    """
    
    def __init__(self, lower_side_length, upper_side_length, height, n_sides=4):
        """
        Initialize a truncated pyramid with given dimensions.
        
        Args:
            lower_side_length (float): Length of the side of the lower base
            upper_side_length (float): Length of the side of the upper base
            height (float): Height of the truncated pyramid
            n_sides (int): Number of sides in the polygonal bases (default is 4 for square)
        """
        if lower_side_length <= 0 or upper_side_length <= 0 or height <= 0:
            raise ValueError("All dimensions must be positive")
        if n_sides < 3:
            raise ValueError("Number of sides must be at least 3")
            
        self.lower_side_length = lower_side_length
        self.upper_side_length = upper_side_length
        self.height = height
        self.n_sides = n_sides
        
    def lower_base_area(self):
        """Calculate the area of the lower base."""
        return self._polygon_area(self.lower_side_length, self.n_sides)
    
    def upper_base_area(self):
        """Calculate the area of the upper base."""
        return self._polygon_area(self.upper_side_length, self.n_sides)
    
    def _polygon_area(self, side_length, n_sides):
        """Helper method to calculate the area of a regular polygon."""
        return (n_sides * side_length**2) / (4 * math.tan(math.pi / n_sides))
    
    def lateral_surface_area(self):
        """Calculate the lateral (side) surface area of the truncated pyramid."""
        # Calculate perimeters
        lower_perimeter = self.n_sides * self.lower_side_length
        upper_perimeter = self.n_sides * self.upper_side_length
        
        # Calculate slant height
        apothem_lower = self.lower_side_length / (2 * math.tan(math.pi / self.n_sides))
        apothem_upper = self.upper_side_length / (2 * math.tan(math.pi / self.n_sides))
        lateral_height = math.sqrt(self.height**2 + (apothem_lower - apothem_upper)**2)
        
        # Apply the formula for lateral surface area
        return 0.5 * (lower_perimeter + upper_perimeter) * lateral_height
    
    def total_surface_area(self):
        """Calculate the total surface area of the truncated pyramid."""
        return self.lower_base_area() + self.upper_base_area() + self.lateral_surface_area()
    
    def volume(self):
        """Calculate the volume of the truncated pyramid."""
        a1 = self.lower_base_area()
        a2 = self.upper_base_area()
        return (self.height / 3) * (a1 + a2 + math.sqrt(a1 * a2))
    
    def slant_height(self):
        """Calculate the slant height of the truncated pyramid."""
        # Distance from center to side (apothem)
        apothem_lower = self.lower_side_length / (2 * math.tan(math.pi / self.n_sides))
        apothem_upper = self.upper_side_length / (2 * math.tan(math.pi / self.n_sides))
        
        # Pythagoras to find slant height
        return math.sqrt(self.height**2 + (apothem_lower - apothem_upper)**2)
    
    def lower_perimeter(self):
        """Calculate the perimeter of the lower base."""
        return self.n_sides * self.lower_side_length
    
    def upper_perimeter(self):
        """Calculate the perimeter of the upper base."""
        return self.n_sides * self.upper_side_length


# Example usage
if __name__ == "__main__":
    # Create a truncated square pyramid with lower side 10, upper side 6, height 8
    frustum = TruncatedPyramid(10, 6, 8)
    
    print(f"Volume: {frustum.volume():.2f} cubic units")
    print(f"Lateral Surface Area: {frustum.lateral_surface_area():.2f} square units")
    print(f"Total Surface Area: {frustum.total_surface_area():.2f} square units")
    print(f"Slant Height: {frustum.slant_height():.2f} units")
    
    # Create a truncated hexagonal pyramid
    hex_frustum = TruncatedPyramid(10, 5, 12, n_sides=6)
    print("\nHexagonal Frustum Properties:")
    print(f"Volume: {hex_frustum.volume():.2f} cubic units")
    print(f"Total Surface Area: {hex_frustum.total_surface_area():.2f} square units")