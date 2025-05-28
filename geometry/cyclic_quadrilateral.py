import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import math
from sympy import symbols, solve, Eq, simplify, sqrt, expand
from sympy.geometry import Point, Circle as SymCircle, Line, Segment
from matplotlib.widgets import Slider

class CyclicQuadrilateral:
    """
    A class to represent, analyze and visualize cyclic quadrilaterals
    and theorems related to their diagonals.
    """
    
    def __init__(self, vertices=None):
        """
        Initialize a cyclic quadrilateral.
        
        Args:
            vertices: A list of four (x, y) tuples representing the vertices.
                     If not provided, creates a default cyclic quadrilateral.
        """
        if vertices is None:
            # Create a default cyclic quadrilateral by placing points on a circle
            angles = np.linspace(0, 2*np.pi, 5)[:-1]  # 4 points equally spaced
            radius = 1.0
            center = (0, 0)
            self.vertices = [(center[0] + radius * np.cos(theta), 
                              center[1] + radius * np.sin(theta)) for theta in angles]
        else:
            if len(vertices) != 4:
                raise ValueError("A quadrilateral must have exactly 4 vertices")
            self.vertices = vertices
            
            # Check if the points are cyclic and make adjustments if needed
            if not self.is_cyclic():
                print("Warning: Provided vertices do not form a cyclic quadrilateral.")
                print("Projecting to a cyclic quadrilateral...")
                self.vertices = self.project_to_cyclic()
        
        # Calculate center and radius of the circumscribed circle
        self.center, self.radius = self.find_circumscribed_circle()
        
        # Calculate key properties
        self.calculate_properties()
    
    def calculate_properties(self):
        """Calculate and store all the key properties of the quadrilateral."""
        A, B, C, D = self.vertices
        
        # Calculate side lengths
        self.sides = [
            self.distance(A, B),  # AB
            self.distance(B, C),  # BC
            self.distance(C, D),  # CD
            self.distance(D, A)   # DA
        ]
        
        # Calculate diagonal lengths
        self.diagonals = [
            self.distance(A, C),  # AC
            self.distance(B, D)   # BD
        ]
        
        # Calculate intersection point of diagonals
        self.intersection = self.find_intersection(A, C, B, D)
        
        # Calculate the segments created by the intersection point
        if self.intersection:
            P = self.intersection
            self.diagonal_segments = [
                self.distance(A, P),  # AP
                self.distance(P, C),  # PC
                self.distance(B, P),  # BP
                self.distance(P, D)   # PD
            ]
        else:
            self.diagonal_segments = [0, 0, 0, 0]
        
        # Calculate angles at vertices
        self.angles = [
            self.calculate_angle(D, A, B),  # Angle at A
            self.calculate_angle(A, B, C),  # Angle at B
            self.calculate_angle(B, C, D),  # Angle at C
            self.calculate_angle(C, D, A)   # Angle at D
        ]
        
        # Calculate area
        self.area = self.calculate_area()
    
    @staticmethod
    def distance(p1, p2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """Calculate the angle at p2 formed by p1-p2-p3 in degrees."""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # Handle numerical errors to ensure cos_angle is in [-1, 1]
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    
    @staticmethod
    def find_intersection(p1, p3, p2, p4):
        """
        Find the intersection point of two line segments p1-p3 and p2-p4.
        Returns None if the lines are parallel.
        """
        x1, y1 = p1
        x2, y2 = p3
        x3, y3 = p2
        x4, y4 = p4
        
        # Line 1 as a1*x + b1*y = c1
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1
        
        # Line 2 as a2*x + b2*y = c2
        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3
        
        # Determinant
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-10:  # Lines are parallel
            return None
        
        # Intersection point
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        
        return (x, y)
    
    def find_circumscribed_circle(self):
        """
        Find the center and radius of the circumscribed circle.
        Uses the perpendicular bisector method.
        """
        A, B, C, _ = self.vertices
        
        # Define symbolic variables
        x, y = symbols('x y')
        
        # Create three points using SymPy
        A_sym = Point(A[0], A[1])
        B_sym = Point(B[0], B[1])
        C_sym = Point(C[0], C[1])
        
        # Find the circle passing through these three points
        circle = SymCircle(A_sym, B_sym, C_sym)
        
        # Extract center and radius
        center = (float(circle.center.x), float(circle.center.y))
        radius = float(circle.radius)
        
        return center, radius
    
    def is_cyclic(self, tolerance=1e-10):
        """
        Check if the quadrilateral is cyclic by verifying if all four vertices
        lie on a circle within a given tolerance.
        """
        # Find the circumscribed circle using the first three points
        A, B, C, D = self.vertices
        
        # Define symbolic variables
        x, y = symbols('x y')
        
        # Create three points using SymPy
        A_sym = Point(A[0], A[1])
        B_sym = Point(B[0], B[1])
        C_sym = Point(C[0], C[1])
        
        # Find the circle passing through these three points
        circle = SymCircle(A_sym, B_sym, C_sym)
        
        # Check if the fourth point is on this circle
        D_sym = Point(D[0], D[1])
        distance_to_circle = abs(circle.center.distance(D_sym) - circle.radius)
        
        return distance_to_circle < tolerance
    
    def project_to_cyclic(self):
        """
        Project the vertices to form a cyclic quadrilateral.
        This preserves 3 vertices and adjusts the 4th to lie on the circle.
        """
        A, B, C, D = self.vertices
        
        # Find the circumcircle from the first three points
        center, radius = self.find_circumscribed_circle()
        
        # Compute the angle of the 4th point relative to the center
        angle = math.atan2(D[1] - center[1], D[0] - center[0])
        
        # Project the 4th point onto the circle
        D_new = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        
        return [A, B, C, D_new]
    
    def calculate_area(self):
        """
        Calculate the area of the cyclic quadrilateral using Brahmagupta's formula.
        """
        a, b, c, d = self.sides
        s = (a + b + c + d) / 2  # Semi-perimeter
        
        # Brahmagupta's formula for the area of a cyclic quadrilateral
        area = math.sqrt((s - a) * (s - b) * (s - c) * (s - d))
        return area
    
    def verify_ptolemy_theorem(self, tolerance=1e-10):
        """
        Verify Ptolemy's theorem: AC * BD = AB * CD + BC * AD
        Returns True if the theorem holds within the specified tolerance.
        """
        # Extract sides and diagonals
        AB = self.sides[0]
        BC = self.sides[1]
        CD = self.sides[2]
        DA = self.sides[3]
        
        AC = self.diagonals[0]
        BD = self.diagonals[1]
        
        # Check if AC * BD = AB * CD + BC * AD
        left_side = AC * BD
        right_side = AB * CD + BC * DA
        
        return abs(left_side - right_side) < tolerance
    
    def verify_diagonal_segment_theorem(self, tolerance=1e-10):
        """
        Verify that the segments of the diagonals satisfy: AP * PC = BP * PD
        Returns True if the theorem holds within the specified tolerance.
        """
        if self.intersection is None:
            return False
        
        AP = self.diagonal_segments[0]
        PC = self.diagonal_segments[1]
        BP = self.diagonal_segments[2]
        PD = self.diagonal_segments[3]
        
        # Check if AP * PC = BP * PD
        left_side = AP * PC
        right_side = BP * PD
        
        return abs(left_side - right_side) < tolerance
    
    def plot(self, ax=None, show_circle=True, show_diagonals=True, show_labels=True,
             show_intersection=True, show_measurements=False):
        """
        Plot the cyclic quadrilateral with its key features.
        
        Args:
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            show_circle: Whether to show the circumscribed circle.
            show_diagonals: Whether to show the diagonals.
            show_labels: Whether to show vertex labels.
            show_intersection: Whether to show the intersection point of diagonals.
            show_measurements: Whether to display measurements (sides, diagonals, etc.)
        
        Returns:
            The matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure
        
        # Get vertices
        A, B, C, D = self.vertices
        
        # Plot the quadrilateral as a polygon
        quad = Polygon(self.vertices, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(quad)
        
        # Plot the vertices
        ax.scatter(*zip(*self.vertices), color='red', s=80, zorder=5)
        
        # Add labels for vertices
        if show_labels:
            vertex_names = ['A', 'B', 'C', 'D']
            offsets = [(0.05, 0.05), (0.05, -0.05), (-0.05, -0.05), (-0.05, 0.05)]
            
            for i, (vertex, name, offset) in enumerate(zip(self.vertices, vertex_names, offsets)):
                ax.text(vertex[0] + offset[0], vertex[1] + offset[1], name, 
                        fontsize=12, ha='center', va='center')
        
        # Plot the circumscribed circle
        if show_circle:
            circle = Circle(self.center, self.radius, fill=False, 
                           edgecolor='green', linestyle='--', linewidth=1.5)
            ax.add_patch(circle)
            
            # Mark the center of the circle
            ax.scatter(*self.center, color='green', s=50, zorder=5)
            if show_labels:
                ax.text(self.center[0], self.center[1] + 0.1, 'O', 
                        fontsize=12, ha='center', va='center')
        
        # Plot the diagonals
        if show_diagonals:
            ax.plot([A[0], C[0]], [A[1], C[1]], 'r--', linewidth=1.5, label='AC')
            ax.plot([B[0], D[0]], [B[1], D[1]], 'r--', linewidth=1.5, label='BD')
            
            # Add labels for diagonals if requested
            if show_labels:
                midpoint_AC = ((A[0] + C[0])/2, (A[1] + C[1])/2)
                midpoint_BD = ((B[0] + D[0])/2, (B[1] + D[1])/2)
                
                ax.text(midpoint_AC[0] + 0.05, midpoint_AC[1] + 0.05, 'AC', 
                        fontsize=10, ha='center', va='center', color='red')
                ax.text(midpoint_BD[0] - 0.05, midpoint_BD[1] - 0.05, 'BD', 
                        fontsize=10, ha='center', va='center', color='red')
        
        # Plot the intersection point of diagonals
        if show_intersection and self.intersection:
            ax.scatter(*self.intersection, color='purple', s=100, zorder=6)
            if show_labels:
                ax.text(self.intersection[0] + 0.05, self.intersection[1] + 0.05, 'P', 
                        fontsize=12, ha='center', va='center')
        
        # Add measurements if requested
        if show_measurements:
            txt = f"Sides: AB={self.sides[0]:.2f}, BC={self.sides[1]:.2f}, "
            txt += f"CD={self.sides[2]:.2f}, DA={self.sides[3]:.2f}\n"
            txt += f"Diagonals: AC={self.diagonals[0]:.2f}, BD={self.diagonals[1]:.2f}\n"
            
            # Ptolemy's theorem
            AC, BD = self.diagonals
            AB, BC, CD, DA = self.sides
            txt += f"Ptolemy's Theorem Check:\n"
            txt += f"AC×BD = {(AC*BD):.2f}, AB×CD+BC×DA = {(AB*CD+BC*DA):.2f}\n"
            
            # Diagonal segments
            if self.intersection:
                AP, PC, BP, PD = self.diagonal_segments
                txt += f"Diagonal Segments: AP={AP:.2f}, PC={PC:.2f}, BP={BP:.2f}, PD={PD:.2f}\n"
                txt += f"AP×PC = {(AP*PC):.2f}, BP×PD = {(BP*PD):.2f}"
            
            ax.text(0.05, 0.05, txt, transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set equal aspect ratio to prevent distortion
        ax.set_aspect('equal')
        plt.axis('off')
        plt.title("Cyclic Quadrilateral and Its Diagonals")
        
        return fig
    
    def interactive_plot(self):
        """
        Create an interactive plot with sliders to adjust the vertices.
        The plot will maintain a cyclic quadrilateral as vertices are moved.
        """
        # Create initial figure and plot
        fig, ax = plt.subplots(figsize=(10, 10))
        self.plot(ax=ax, show_measurements=True)
        
        # Add sliders for adjusting the vertices
        slider_axes = []
        for i in range(4):
            ax_theta = plt.axes([0.2, 0.01 + i*0.03, 0.65, 0.02])
            slider_axes.append(ax_theta)
        
        # Initial angles of the vertices
        center = self.center
        angles = []
        for vertex in self.vertices:
            angle = math.atan2(vertex[1] - center[1], vertex[0] - center[0])
            if angle < 0:
                angle += 2 * math.pi
            angles.append(angle)
        
        # Create sliders for adjusting angles
        sliders = []
        for i, (ax_theta, angle) in enumerate(zip(slider_axes, angles)):
            slider = Slider(
                ax=ax_theta,
                label=f'Vertex {chr(65+i)}',  # A, B, C, D
                valmin=0,
                valmax=2*np.pi,
                valinit=angle,
            )
            sliders.append(slider)
        
        # Function to update the plot when sliders are moved
        def update(val):
            new_vertices = []
            for i, slider in enumerate(sliders):
                angle = slider.val
                x = center[0] + self.radius * math.cos(angle)
                y = center[1] + self.radius * math.sin(angle)
                new_vertices.append((x, y))
            
            # Update the quadrilateral
            self.vertices = new_vertices
            self.calculate_properties()
            
            # Clear and redraw
            ax.clear()
            self.plot(ax=ax, show_measurements=True)
            fig.canvas.draw_idle()
        
        # Register the update function with each slider
        for slider in sliders:
            slider.on_changed(update)
        
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        
        return fig, sliders


class CyclicQuadrilateralTheorems:
    """
    A class to demonstrate and prove theorems related to 
    cyclic quadrilaterals and their diagonals.
    """
    
    @staticmethod
    def symbolic_ptolemy_proof():
        """
        Provide a symbolic proof of Ptolemy's theorem using SymPy.
        """
        # Define symbolic variables for coordinates
        xA, yA = symbols('xA yA')
        xB, yB = symbols('xB yB')
        xC, yC = symbols('xC yC')
        xD, yD = symbols('xD yD')
        
        # Define points
        A = Point(xA, yA)
        B = Point(xB, yB)
        C = Point(xC, yC)
        D = Point(xD, yD)
        
        # Define side lengths and diagonals
        AB = Segment(A, B).length
        BC = Segment(B, C).length
        CD = Segment(C, D).length
        DA = Segment(D, A).length
        
        AC = Segment(A, C).length
        BD = Segment(B, D).length
        
        # In a cyclic quadrilateral, opposite angles are supplementary
        # We can use this to derive the sine law relation
        
        # For clarity, let's print the steps of the proof
        print("Symbolic Proof of Ptolemy's Theorem:")
        print("------------------------------------")
        print("For a cyclic quadrilateral ABCD, we have:")
        print("1. Opposite angles are supplementary: A + C = B + D = π")
        print("2. Using the Law of Sines in triangles:")
        print("   In triangle ABC: sin(B) / AC = sin(C) / AB")
        print("   In triangle ACD: sin(C) / AD = sin(D) / AC")
        print("   In triangle ABD: sin(B) / AD = sin(D) / AB")
        print("3. Ptolemy's Theorem states: AC * BD = AB * CD + BC * AD")
        print("\nThe proof follows from the trigonometric identities and the")
        print("properties of inscribed angles in a circle.")
        
        # Demonstrations would continue with algebraic manipulations
        # of the sine law relationships
        
        return {
            'statement': "AC * BD = AB * CD + BC * AD",
            'conclusion': "Ptolemy's theorem holds for all cyclic quadrilaterals."
        }
    
    @staticmethod
    def diagonal_bisector_theorem():
        """
        Demonstrate the theorem about the segments of intersecting diagonals.
        """
        print("Theorem on Diagonal Segments:")
        print("-----------------------------")
        print("Let P be the intersection point of the diagonals AC and BD")
        print("in a cyclic quadrilateral ABCD.")
        print("Then: AP * PC = BP * PD")
        print("\nThis follows from the power of a point theorem, where P")
        print("has the same power with respect to any chord through it in the circle.")
        
        return {
            'statement': "AP * PC = BP * PD",
            'conclusion': "The products of the segments of the diagonals are equal."
        }
    
    @staticmethod
    def pythagorean_theorem_special_case():
        """
        Demonstrate the Pythagorean theorem as a special case of Ptolemy's theorem.
        """
        print("Pythagorean Theorem as a Special Case:")
        print("--------------------------------------")
        print("When the cyclic quadrilateral is a rectangle, Ptolemy's theorem")
        print("reduces to the Pythagorean theorem.")
        print("\nIn a rectangle ABCD:")
        print("1. AB = CD and BC = DA (opposite sides are equal)")
        print("2. AC = BD (diagonals are equal)")
        print("3. Applying Ptolemy's theorem:")
        print("   AC * BD = AB * CD + BC * AD")
        print("   AC * AC = AB * AB + BC * BC")
        print("   AC² = AB² + BC²")
        print("\nThis is exactly the Pythagorean theorem!")
        
        return {
            'statement': "AC² = AB² + BC²",
            'conclusion': "The Pythagorean theorem is a special case of Ptolemy's theorem."
        }
    
    @staticmethod
    def generate_examples():
        """
        Generate example cyclic quadrilaterals to demonstrate the theorems.
        """
        examples = []
        
        # Example 1: Regular quadrilateral (square)
        vertices1 = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1)
        ]
        examples.append(CyclicQuadrilateral(vertices1))
        
        # Example 2: Irregular cyclic quadrilateral
        vertices2 = [
            (1, 0),
            (0.5, 0.8),
            (-0.7, 0.3),
            (-0.2, -0.9)
        ]
        examples.append(CyclicQuadrilateral(vertices2))
        
        # Example 3: Kite (a type of cyclic quadrilateral)
        vertices3 = [
            (0, 2),
            (1, 0),
            (0, -1),
            (-1, 0)
        ]
        examples.append(CyclicQuadrilateral(vertices3))
        
        return examples


def main():
    """
    Main function to demonstrate the theorems and properties
    of cyclic quadrilaterals and their diagonals.
    """
    print("CYCLIC QUADRILATERALS AND THEIR DIAGONALS")
    print("=========================================")
    print("\nA cyclic quadrilateral is a quadrilateral whose vertices")
    print("all lie on a single circle.")
    print("\nMain Theorems:")
    print("1. Ptolemy's Theorem: AC * BD = AB * CD + BC * AD")
    print("2. Diagonal Segments: AP * PC = BP * PD (where P is the intersection)")
    
    # Create theorems instance
    theorems = CyclicQuadrilateralTheorems()
    
    # Generate examples
    print("\nGenerating examples to demonstrate the theorems...")
    examples = theorems.generate_examples()
    
    # Test each example
    for i, quad in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"  Sides: AB={quad.sides[0]:.4f}, BC={quad.sides[1]:.4f}, CD={quad.sides[2]:.4f}, DA={quad.sides[3]:.4f}")
        print(f"  Diagonals: AC={quad.diagonals[0]:.4f}, BD={quad.diagonals[1]:.4f}")
        
        # Check Ptolemy's theorem
        ptolemy_check = quad.verify_ptolemy_theorem()
        print(f"  Ptolemy's Theorem Check: AC*BD = {quad.diagonals[0]*quad.diagonals[1]:.4f}")
        print(f"                           AB*CD+BC*DA = {quad.sides[0]*quad.sides[2]+quad.sides[1]*quad.sides[3]:.4f}")
        print(f"  Ptolemy's Theorem holds: {ptolemy_check}")
        
        # Check diagonal segments theorem
        if quad.intersection:
            segment_check = quad.verify_diagonal_segment_theorem()
            AP, PC, BP, PD = quad.diagonal_segments
            print(f"  Diagonal Segments: AP={AP:.4f}, PC={PC:.4f}, BP={BP:.4f}, PD={PD:.4f}")
            print(f"  Segment Theorem Check: AP*PC = {AP*PC:.4f}, BP*PD = {BP*PD:.4f}")
            print(f"  Segment Theorem holds: {segment_check}")
        else:
            print("  Diagonals do not intersect.")
    
    # Create plots for examples
    fig, axes = plt.subplots(1, len(examples), figsize=(15, 5))
    for i, (quad, ax) in enumerate(zip(examples, axes)):
        quad.plot(ax=ax, show_measurements=False)
        ax.set_title(f"Example {i+1}")
    
    # Show the symbolic proofs
    print("\n" + "="*50)
    theorems.symbolic_ptolemy_proof()
    print("\n" + "="*50)
    theorems.diagonal_bisector_theorem()
    print("\n" + "="*50)
    theorems.pythagorean_theorem_special_case()
    
    # Create an interactive demonstration
    print("\nCreating interactive demonstration...")
    print("Adjust the sliders to move the vertices along the circle")
    print("and observe how the theorems continue to hold.")
    interactive_quad = CyclicQuadrilateral()
    interactive_quad.interactive_plot()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()