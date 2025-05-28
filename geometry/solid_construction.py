import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.spatial import ConvexHull
import math
from itertools import combinations, product

class PlatonicSolids:
    """
    A class to construct, analyze, and visualize the five Platonic solids
    according to ancient Greek geometric methods.
    """
    
    def __init__(self):
        """Initialize the class with the geometric data for each Platonic solid."""
        # Phi is the golden ratio, central to the construction of several Platonic solids
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Standard radius for all solids (distance from center to vertices)
        self.radius = 1.0
        
        # Dictionary to store the vertices of each solid
        self.vertices = {}
        
        # Dictionary to store the faces of each solid
        self.faces = {}
        
        # Dictionary to store the edges of each solid
        self.edges = {}
        
        # Dictionary to store the dual relationships
        self.duals = {
            'tetrahedron': 'tetrahedron',
            'cube': 'octahedron',
            'octahedron': 'cube',
            'dodecahedron': 'icosahedron',
            'icosahedron': 'dodecahedron'
        }
        
        # Generate all five Platonic solids
        self.construct_tetrahedron()
        self.construct_cube()
        self.construct_octahedron()
        self.construct_dodecahedron()
        self.construct_icosahedron()
        
        # Calculate properties for all solids
        self.calculate_properties()
    
    def construct_tetrahedron(self):
        """
        Construct a regular tetrahedron using the ancient method.
        The tetrahedron is constructed from an equilateral triangle base
        with a vertex above the center of the base.
        """
        # Method based on vertices of a cube
        vertices = np.array([
            [1, 1, 1],   # Vertex 0
            [1, -1, -1], # Vertex 1
            [-1, 1, -1], # Vertex 2
            [-1, -1, 1]  # Vertex 3
        ])
        
        # Normalize to have consistent radius
        vertices = self.normalize_vertices(vertices)
        
        # Define the faces as sets of vertex indices
        faces = [
            [0, 1, 2],  # Face 0
            [0, 3, 1],  # Face 1
            [0, 2, 3],  # Face 2
            [1, 3, 2]   # Face 3
        ]
        
        # Calculate edges as pairs of vertex indices
        edges = list(combinations(range(4), 2))
        
        # Store the tetrahedron data
        self.vertices['tetrahedron'] = vertices
        self.faces['tetrahedron'] = faces
        self.edges['tetrahedron'] = edges
    
    def construct_cube(self):
        """
        Construct a regular cube (hexahedron) using the ancient method.
        The cube is aligned with the coordinate axes.
        """
        # Method based on the 8 corners of a cube
        vertices = np.array(list(product([-1, 1], repeat=3)))
        
        # Normalize to have consistent radius
        vertices = self.normalize_vertices(vertices)
        
        # Define the faces as sets of vertex indices
        faces = [
            [0, 1, 3, 2],  # Face 0 (bottom)
            [4, 5, 7, 6],  # Face 1 (top)
            [0, 1, 5, 4],  # Face 2 (front)
            [2, 3, 7, 6],  # Face 3 (back)
            [0, 2, 6, 4],  # Face 4 (left)
            [1, 3, 7, 5]   # Face 5 (right)
        ]
        
        # Calculate edges as pairs of vertex indices
        edges = []
        for face in faces:
            n = len(face)
            for i in range(n):
                edge = [face[i], face[(i+1) % n]]
                edge.sort()
                if edge not in edges:
                    edges.append(edge)
        
        # Store the cube data
        self.vertices['cube'] = vertices
        self.faces['cube'] = faces
        self.edges['cube'] = edges
    
    def construct_octahedron(self):
        """
        Construct a regular octahedron using the ancient method.
        The octahedron is the dual of the cube.
        """
        # Method based on the 6 vertices along the coordinate axes
        vertices = np.array([
            [1, 0, 0],   # Vertex 0 (+x)
            [-1, 0, 0],  # Vertex 1 (-x)
            [0, 1, 0],   # Vertex 2 (+y)
            [0, -1, 0],  # Vertex 3 (-y)
            [0, 0, 1],   # Vertex 4 (+z)
            [0, 0, -1]   # Vertex 5 (-z)
        ])
        
        # Normalize to have consistent radius
        vertices = self.normalize_vertices(vertices)
        
        # Define the faces as sets of vertex indices
        faces = [
            [0, 2, 4],  # Face 0
            [0, 4, 3],  # Face 1
            [0, 3, 5],  # Face 2
            [0, 5, 2],  # Face 3
            [1, 2, 4],  # Face 4
            [1, 4, 3],  # Face 5
            [1, 3, 5],  # Face 6
            [1, 5, 2]   # Face 7
        ]
        
        # Calculate edges as pairs of vertex indices
        edges = []
        for face in faces:
            n = len(face)
            for i in range(n):
                edge = [face[i], face[(i+1) % n]]
                edge.sort()
                if edge not in edges:
                    edges.append(edge)
        
        # Store the octahedron data
        self.vertices['octahedron'] = vertices
        self.faces['octahedron'] = faces
        self.edges['octahedron'] = edges
    
    def construct_dodecahedron(self):
        """
        Construct a regular dodecahedron using the ancient method.
        The dodecahedron is constructed using the golden ratio.
        """
        # Method based on the 20 vertices derived from the golden ratio
        phi = self.phi
        
        # Vertices come from three groups:
        # 1. Permutations of (±1, ±1, ±1)
        # 2. Permutations of (0, ±phi, ±1/phi)
        # 3. Permutations of (±phi, ±1/phi, 0)
        # 4. Permutations of (±1/phi, 0, ±phi)
        
        vertices = []
        
        # Group 1: Permutations of (±1, ±1, ±1)
        vertices.extend([
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1]
        ])
        
        # Group 2: Permutations of (0, ±phi, ±1/phi)
        vertices.extend([
            [0, phi, 1/phi],
            [0, phi, -1/phi],
            [0, -phi, 1/phi],
            [0, -phi, -1/phi]
        ])
        
        # Group 3: Permutations of (±phi, ±1/phi, 0)
        vertices.extend([
            [phi, 1/phi, 0],
            [phi, -1/phi, 0],
            [-phi, 1/phi, 0],
            [-phi, -1/phi, 0]
        ])
        
        # Group 4: Permutations of (±1/phi, 0, ±phi)
        vertices.extend([
            [1/phi, 0, phi],
            [1/phi, 0, -phi],
            [-1/phi, 0, phi],
            [-1/phi, 0, -phi]
        ])
        
        vertices = np.array(vertices)
        
        # Find convex hull to determine faces
        hull = ConvexHull(vertices)
        
        # Normalize to have consistent radius
        vertices = self.normalize_vertices(vertices)
        
        # Extract faces from the convex hull
        faces = [simplex for simplex in hull.simplices if len(simplex) >= 3]
        
        # Ensure all faces have 5 vertices (pentagonal faces)
        pentagonal_faces = []
        for face_indices in hull.simplices:
            # Get vertices of face
            face_vertices = vertices[face_indices]
            
            # Find center of face
            center = np.mean(face_vertices, axis=0)
            
            # Normalize center to lie on the surface
            center = center / np.linalg.norm(center) * self.radius
            
            # Find all vertices close to this face plane
            # We use the dot product with the normal to determine closeness
            normal = center  # For a regular dodecahedron, the face center is proportional to the normal
            
            # Find vertices that are at a similar distance from the origin in the direction of the normal
            threshold = 0.01
            face_points = []
            
            for i, vertex in enumerate(vertices):
                # Project vertex onto normal
                projection = np.dot(vertex, normal) / np.linalg.norm(normal)
                
                # If this projection is close to the distance of the face from the origin
                if abs(projection - np.linalg.norm(center)) < threshold:
                    face_points.append(i)
            
            # If we found 5 vertices (a pentagon), add to our faces
            if len(face_points) == 5:
                # We need to sort the vertices to form a proper pentagon
                # This is a simplification; in practice, you'd need more complex ordering
                pentagonal_faces.append(face_points)
        
        # Use identified pentagonal faces
        faces = pentagonal_faces
        
        # Calculate edges as pairs of vertex indices
        edges = []
        for face in faces:
            n = len(face)
            for i in range(n):
                edge = [face[i], face[(i+1) % n]]
                edge.sort()
                if edge not in edges:
                    edges.append(edge)
        
        # Store the dodecahedron data
        self.vertices['dodecahedron'] = vertices
        self.faces['dodecahedron'] = faces
        self.edges['dodecahedron'] = edges
    
    def construct_icosahedron(self):
        """
        Construct a regular icosahedron using the ancient method.
        The icosahedron is constructed using the golden ratio.
        """
        # Method based on the 12 vertices derived from the golden ratio
        phi = self.phi
        
        # Vertices are placed at:
        # 1. (0, ±1, ±phi)
        # 2. (±1, ±phi, 0)
        # 3. (±phi, 0, ±1)
        
        vertices = []
        
        # Group 1: (0, ±1, ±phi)
        vertices.extend([
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi]
        ])
        
        # Group 2: (±1, ±phi, 0)
        vertices.extend([
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0]
        ])
        
        # Group 3: (±phi, 0, ±1)
        vertices.extend([
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1]
        ])
        
        vertices = np.array(vertices)
        
        # Find convex hull to determine faces
        hull = ConvexHull(vertices)
        
        # Normalize to have consistent radius
        vertices = self.normalize_vertices(vertices)
        
        # Extract faces from the convex hull
        faces = [list(simplex) for simplex in hull.simplices]
        
        # Calculate edges as pairs of vertex indices
        edges = []
        for face in faces:
            n = len(face)
            for i in range(n):
                edge = [face[i], face[(i+1) % n]]
                edge.sort()
                if edge not in edges:
                    edges.append(edge)
        
        # Store the icosahedron data
        self.vertices['icosahedron'] = vertices
        self.faces['icosahedron'] = faces
        self.edges['icosahedron'] = edges
    
    def normalize_vertices(self, vertices):
        """Normalize vertices to have a consistent radius from the origin."""
        norms = np.linalg.norm(vertices, axis=1)
        normalized = vertices / norms[:, np.newaxis] * self.radius
        return normalized
    
    def calculate_properties(self):
        """Calculate and store various properties for each Platonic solid."""
        self.properties = {}
        
        for solid_name in self.vertices:
            vertices = self.vertices[solid_name]
            faces = self.faces[solid_name]
            edges = self.edges[solid_name]
            
            # Count elements
            vertex_count = len(vertices)
            face_count = len(faces)
            edge_count = len(edges)
            
            # Verify Euler's formula: V - E + F = 2
            euler_characteristic = vertex_count - edge_count + face_count
            
            # Calculate face angles
            if solid_name == 'tetrahedron':
                face_angle = 60  # degrees
                dihedral_angle = np.arccos(1/3) * 180 / np.pi
            elif solid_name == 'cube':
                face_angle = 90  # degrees
                dihedral_angle = 90  # degrees
            elif solid_name == 'octahedron':
                face_angle = 60  # degrees
                dihedral_angle = np.arccos(-1/3) * 180 / np.pi
            elif solid_name == 'dodecahedron':
                face_angle = 108  # degrees
                dihedral_angle = np.arccos(-np.sqrt(5)/5) * 180 / np.pi
            elif solid_name == 'icosahedron':
                face_angle = 60  # degrees
                dihedral_angle = np.arccos(-np.sqrt(5)/3) * 180 / np.pi
            
            # Calculate volumes
            if solid_name == 'tetrahedron':
                volume = np.sqrt(2) / 12 * (2 * self.radius)**3
            elif solid_name == 'cube':
                # Length of edge for a cube with vertices at distance self.radius from center
                edge_length = 2 * self.radius / np.sqrt(3)
                volume = edge_length**3
            elif solid_name == 'octahedron':
                volume = np.sqrt(2) / 3 * (2 * self.radius)**3
            elif solid_name == 'dodecahedron':
                volume = (15 + 7*np.sqrt(5)) / 4 * self.radius**3
            elif solid_name == 'icosahedron':
                volume = (5 * (3 + np.sqrt(5))) / 12 * self.radius**3
            
            # Store properties
            self.properties[solid_name] = {
                'vertex_count': vertex_count,
                'face_count': face_count,
                'edge_count': edge_count,
                'euler_characteristic': euler_characteristic,
                'face_angle': face_angle,
                'dihedral_angle': dihedral_angle,
                'volume': volume
            }
    
    def calculate_dual(self, solid_name):
        """Calculate the dual of a given Platonic solid."""
        # The dual is already known from the duals dictionary
        dual_name = self.duals[solid_name]
        
        # For completeness, calculate dual vertices from face centers of the original
        vertices = self.vertices[solid_name]
        faces = self.faces[solid_name]
        dual_vertices = []
        
        for face in faces:
            face_vertices = [vertices[i] for i in face]
            face_center = np.mean(face_vertices, axis=0)
            face_center = face_center / np.linalg.norm(face_center) * self.radius
            dual_vertices.append(face_center)
        
        dual_vertices = np.array(dual_vertices)
        
        return dual_name, dual_vertices
    
    def get_solid_info(self, solid_name):
        """Get comprehensive information about a Platonic solid."""
        if solid_name not in self.properties:
            return f"Solid '{solid_name}' not found."
        
        props = self.properties[solid_name]
        dual_name = self.duals[solid_name]
        
        info = f"=== {solid_name.capitalize()} ===\n"
        info += f"Vertices: {props['vertex_count']}\n"
        info += f"Faces: {props['face_count']}\n"
        info += f"Edges: {props['edge_count']}\n"
        info += f"Euler Characteristic (V - E + F): {props['euler_characteristic']}\n"
        info += f"Face Angle: {props['face_angle']} degrees\n"
        info += f"Dihedral Angle: {props['dihedral_angle']:.2f} degrees\n"
        info += f"Volume (with radius {self.radius}): {props['volume']:.4f}\n"
        info += f"Dual Solid: {dual_name.capitalize()}\n"
        
        # Historical context
        if solid_name == 'tetrahedron':
            info += "\nHistorical Context: The tetrahedron was associated with the element Fire by Plato. "
            info += "It has the simplest structure of all Platonic solids."
        elif solid_name == 'cube':
            info += "\nHistorical Context: The cube was associated with the element Earth by Plato. "
            info += "It was considered the most stable and grounded of the solids."
        elif solid_name == 'octahedron':
            info += "\nHistorical Context: The octahedron was associated with the element Air by Plato. "
            info += "Its dual relationship with the cube was noted by ancient geometers."
        elif solid_name == 'dodecahedron':
            info += "\nHistorical Context: The dodecahedron was associated with the Cosmos by Plato. "
            info += "It was considered the most mysterious and was sometimes associated with Aether, "
            info += "the fifth element. Its construction relies on the golden ratio."
        elif solid_name == 'icosahedron':
            info += "\nHistorical Context: The icosahedron was associated with the element Water by Plato. "
            info += "Like the dodecahedron, its construction involves the golden ratio."
        
        # Construction method
        info += "\n\nAncient Construction Method:\n"
        if solid_name == 'tetrahedron':
            info += "1. Begin with an equilateral triangle as the base.\n"
            info += "2. From each vertex of the triangle, construct a line that makes equal angles "
            info += "with all three sides meeting at that vertex.\n"
            info += "3. These three lines meet at a point that becomes the fourth vertex of the tetrahedron."
        elif solid_name == 'cube':
            info += "1. Begin with a square as the base.\n"
            info += "2. Construct a line perpendicular to the plane of the square at its center.\n"
            info += "3. The length of this line is equal to the side length of the square.\n"
            info += "4. Connect the endpoints of this line to the vertices of the square to form the cube."
        elif solid_name == 'octahedron':
            info += "1. Begin with two squares in parallel planes, with the second square rotated 45° "
            info += "relative to the first.\n"
            info += "2. Connect the vertices of one square to the center of the other square.\n"
            info += "3. This forms an octahedron with 8 triangular faces."
        elif solid_name == 'dodecahedron':
            info += "1. Begin with a cube.\n"
            info += "2. Construct a golden rectangle on each face of the cube.\n"
            info += "3. The vertices of these golden rectangles form a dodecahedron.\n"
            info += "4. This construction illustrates why the dodecahedron's structure depends on the golden ratio."
        elif solid_name == 'icosahedron':
            info += "1. Start with three golden rectangles placed in three perpendicular planes.\n"
            info += "2. The 12 vertices of these rectangles form an icosahedron.\n"
            info += "3. Like the dodecahedron, the icosahedron's structure fundamentally depends on the golden ratio (φ)."
            
        return info
    
    def plot_solid(self, solid_name, ax=None, show_vertices=True, show_edges=True, 
                  show_faces=True, alpha=0.7, color='skyblue', edge_color='black', 
                  vertex_color='red', show_labels=False):
        """
        Plot a Platonic solid on the given axes.
        
        Args:
            solid_name: Name of the solid ('tetrahedron', 'cube', etc.)
            ax: Matplotlib 3D axes to plot on (if None, creates a new figure)
            show_vertices: Whether to show vertices as points
            show_edges: Whether to show edges as lines
            show_faces: Whether to show faces as polygons
            alpha: Transparency of faces
            color: Color of faces
            edge_color: Color of edges
            vertex_color: Color of vertices
            show_labels: Whether to show vertex labels
            
        Returns:
            The matplotlib figure containing the plot
        """
        if solid_name not in self.vertices:
            print(f"Solid '{solid_name}' not found.")
            return None
        
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        vertices = self.vertices[solid_name]
        faces = self.faces[solid_name]
        edges = self.edges[solid_name]
        
        # Plot faces
        if show_faces:
            poly_faces = []
            for face in faces:
                poly_face = [vertices[i] for i in face]
                poly_faces.append(poly_face)
            
            poly_collection = Poly3DCollection(poly_faces, alpha=alpha, linewidth=1, edgecolor=edge_color)
            poly_collection.set_facecolor(color)
            ax.add_collection3d(poly_collection)
        
        # Plot edges
        if show_edges:
            for edge in edges:
                ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]],
                        [vertices[edge[0]][1], vertices[edge[1]][1]],
                        [vertices[edge[0]][2], vertices[edge[1]][2]],
                        color=edge_color, linewidth=1.5)
        
        # Plot vertices
        if show_vertices:
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      color=vertex_color, s=50, depthshade=True)
            
            # Add vertex labels if requested
            if show_labels:
                for i, vertex in enumerate(vertices):
                    ax.text(vertex[0]*1.1, vertex[1]*1.1, vertex[2]*1.1, str(i),
                           fontsize=10)
        
        # Set plot properties
        max_val = self.radius * 1.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.set_title(f"{solid_name.capitalize()} (r={self.radius})")
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig
    
    def plot_all_solids(self):
        """Plot all five Platonic solids in a single figure."""
        fig = plt.figure(figsize=(18, 10))
        
        # Create 5 subplots for each solid
        solids = ['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron']
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, solid_name in enumerate(solids):
            ax = fig.add_subplot(1, 5, i+1, projection='3d')
            self.plot_solid(solid_name, ax=ax, color=colors[i])
            
        plt.tight_layout()
        plt.suptitle('The Five Platonic Solids', fontsize=20, y=1.05)
        
        return fig
    
    def plot_dual_pair(self, solid_name):
        """Plot a Platonic solid alongside its dual."""
        if solid_name not in self.vertices:
            print(f"Solid '{solid_name}' not found.")
            return None
        
        dual_name = self.duals[solid_name]
        
        fig = plt.figure(figsize=(15, 7))
        
        # Plot the original solid
        ax1 = fig.add_subplot(121, projection='3d')
        self.plot_solid(solid_name, ax=ax1, color='skyblue')
        
        # Plot the dual solid
        ax2 = fig.add_subplot(122, projection='3d')
        self.plot_solid(dual_name, ax=ax2, color='salmon')
        
        plt.suptitle(f'Dual Pair: {solid_name.capitalize()} and {dual_name.capitalize()}', 
                    fontsize=16, y=0.98)
        
        return fig
    
    def plot_nested_dual(self, solid_name):
        """Plot a Platonic solid with its dual nested inside."""
        if solid_name not in self.vertices:
            print(f"Solid '{solid_name}' not found.")
            return None
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the original solid
        self.plot_solid(solid_name, ax=ax, color='skyblue', alpha=0.3)
        
        # Calculate and plot the dual
        dual_name, dual_vertices = self.calculate_dual(solid_name)
        
        # Create a smaller version of the dual for visual clarity
        scale_factor = 0.7
        dual_vertices_scaled = dual_vertices * scale_factor
        
        # Store original vertices
        original_vertices = self.vertices[dual_name]
        
        # Temporarily replace with scaled dual vertices
        self.vertices[dual_name] = dual_vertices_scaled
        
        # Plot the scaled dual
        self.plot_solid(dual_name, ax=ax, color='salmon', alpha=0.7)
        
        # Restore original vertices
        self.vertices[dual_name] = original_vertices
        
        ax.set_title(f'{solid_name.capitalize()} with nested {dual_name.capitalize()} dual')
        
        return fig
    
    def animate_rotation(self, solid_name, elevation=30):
        """Create an animation of the solid rotating."""
        if solid_name not in self.vertices:
            print(f"Solid '{solid_name}' not found.")
            return None
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            ax.view_init(elev=elevation, azim=frame)
            self.plot_solid(solid_name, ax=ax)
            
            max_val = self.radius * 1.5
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.set_zlim(-max_val, max_val)
            
            ax.set_title(f'Rotating {solid_name.capitalize()}')
            ax.set_box_aspect([1, 1, 1])
            
            return ax,
        
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2),
                                      interval=50, blit=False)
        
        return anim
    
    def interactive_solid_explorer(self):
        """Create an interactive demonstration of all Platonic solids."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create radio buttons for solid selection
        rax = plt.axes([0.05, 0.7, 0.15, 0.15])
        radio_solids = RadioButtons(rax, ['tetrahedron', 'cube', 'octahedron', 
                                         'dodecahedron', 'icosahedron'])
        
        # Create radio buttons for display options
        rax2 = plt.axes([0.05, 0.4, 0.15, 0.15])
        radio_options = RadioButtons(rax2, ['solid', 'wireframe', 'vertices', 'dual'])
        
        # Create sliders for rotation
        ax_elev = plt.axes([0.25, 0.05, 0.65, 0.03])
        ax_azim = plt.axes([0.25, 0.1, 0.65, 0.03])
        
        slider_elev = Slider(ax_elev, 'Elevation', 0, 90, valinit=30)
        slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=30)
        
        # Create text box for information
        text_ax = plt.axes([0.75, 0.7, 0.2, 0.25])
        text_ax.axis('off')
        
        # Initial solid and view
        current_solid = 'tetrahedron'
        display_mode = 'solid'
        
        def update(val=None):
            ax.clear()
            
            # Set the viewpoint
            ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
            
            # Display based on selected mode
            if display_mode == 'solid':
                self.plot_solid(current_solid, ax=ax)
            elif display_mode == 'wireframe':
                self.plot_solid(current_solid, ax=ax, show_faces=False)
            elif display_mode == 'vertices':
                self.plot_solid(current_solid, ax=ax, show_faces=False, show_edges=False)
            elif display_mode == 'dual':
                self.plot_nested_dual(current_solid)
            
            # Update information text
            text_ax.clear()
            text_ax.axis('off')
            info = self.get_solid_info(current_solid)
            # Only display the first few lines to avoid crowding
            info_lines = info.split('\n')[:5]
            info = '\n'.join(info_lines)
            text_ax.text(0, 1, info, fontsize=10, va='top')
            
            fig.canvas.draw_idle()
        
        def solid_changed(label):
            nonlocal current_solid
            current_solid = label
            update()
        
        def option_changed(label):
            nonlocal display_mode
            display_mode = label
            update()
        
        radio_solids.on_clicked(solid_changed)
        radio_options.on_clicked(option_changed)
        slider_elev.on_changed(update)
        slider_azim.on_changed(update)
        
        # Initial update
        update()
        
        plt.subplots_adjust(bottom=0.25)
        return fig, radio_solids, radio_options, slider_elev, slider_azim
    
    def plot_construction_steps(self, solid_name):
        """
        Visualize the step-by-step ancient construction of a Platonic solid.
        This is a simplified representation of how ancient geometers might have
        constructed these solids.
        """
        if solid_name not in self.vertices:
            print(f"Solid '{solid_name}' not found.")
            return None
        
        fig = plt.figure(figsize=(15, 10))
        
        if solid_name == 'tetrahedron':
            # Step 1: Equilateral triangle (base)
            ax1 = fig.add_subplot(131, projection='3d')
            
            # Define the base triangle
            r = self.radius
            triangle_vertices = np.array([
                [r, 0, 0],
                [-r/2, r*np.sqrt(3)/2, 0],
                [-r/2, -r*np.sqrt(3)/2, 0]
            ])
            
            # Plot the triangle
            ax1.plot([triangle_vertices[0][0], triangle_vertices[1][0], triangle_vertices[2][0], triangle_vertices[0][0]],
                    [triangle_vertices[0][1], triangle_vertices[1][1], triangle_vertices[2][1], triangle_vertices[0][1]],
                    [triangle_vertices[0][2], triangle_vertices[1][2], triangle_vertices[2][2], triangle_vertices[0][2]],
                    'b-')
            
            ax1.scatter(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], 
                       color='red', s=50)
            
            ax1.set_title('Step 1: Equilateral Triangle Base')
            
            # Step 2: Adding the apex
            ax2 = fig.add_subplot(132, projection='3d')
            
            # Plot the triangle
            ax2.plot([triangle_vertices[0][0], triangle_vertices[1][0], triangle_vertices[2][0], triangle_vertices[0][0]],
                    [triangle_vertices[0][1], triangle_vertices[1][1], triangle_vertices[2][1], triangle_vertices[0][1]],
                    [triangle_vertices[0][2], triangle_vertices[1][2], triangle_vertices[2][2], triangle_vertices[0][2]],
                    'b-')
            
            # Calculate the apex point
            apex = np.array([0, 0, r*np.sqrt(2/3)])
            
            # Plot the apex
            ax2.scatter([apex[0]], [apex[1]], [apex[2]], color='green', s=100)
            
            # Draw lines from base to apex
            for vertex in triangle_vertices:
                ax2.plot([vertex[0], apex[0]], [vertex[1], apex[1]], [vertex[2], apex[2]], 'g--')
            
            ax2.set_title('Step 2: Adding the Apex')
            
            # Step 3: Complete tetrahedron
            ax3 = fig.add_subplot(133, projection='3d')
            self.plot_solid('tetrahedron', ax=ax3)
            ax3.set_title('Step 3: Complete Tetrahedron')
            
        elif solid_name == 'cube':
            # Step 1: Square base
            ax1 = fig.add_subplot(131, projection='3d')
            
            # Define the base square
            r = self.radius / np.sqrt(3)  # Adjusted for cube corners to be at distance self.radius
            square_vertices = np.array([
                [r, r, 0],
                [-r, r, 0],
                [-r, -r, 0],
                [r, -r, 0]
            ])
            
            # Plot the square
            ax1.plot([square_vertices[0][0], square_vertices[1][0], square_vertices[2][0], square_vertices[3][0], square_vertices[0][0]],
                    [square_vertices[0][1], square_vertices[1][1], square_vertices[2][1], square_vertices[3][1], square_vertices[0][1]],
                    [square_vertices[0][2], square_vertices[1][2], square_vertices[2][2], square_vertices[3][2], square_vertices[0][2]],
                    'b-')
            
            ax1.scatter(square_vertices[:, 0], square_vertices[:, 1], square_vertices[:, 2], 
                       color='red', s=50)
            
            ax1.set_title('Step 1: Square Base')
            
            # Step 2: Adding height
            ax2 = fig.add_subplot(132, projection='3d')
            
            # Plot the base square
            ax2.plot([square_vertices[0][0], square_vertices[1][0], square_vertices[2][0], square_vertices[3][0], square_vertices[0][0]],
                    [square_vertices[0][1], square_vertices[1][1], square_vertices[2][1], square_vertices[3][1], square_vertices[0][1]],
                    [square_vertices[0][2], square_vertices[1][2], square_vertices[2][2], square_vertices[3][2], square_vertices[0][2]],
                    'b-')
            
            # Calculate the top square
            top_square = square_vertices.copy()
            top_square[:, 2] = 2*r  # Height equals side length for a cube
            
            # Plot the top square
            ax2.plot([top_square[0][0], top_square[1][0], top_square[2][0], top_square[3][0], top_square[0][0]],
                    [top_square[0][1], top_square[1][1], top_square[2][1], top_square[3][1], top_square[0][1]],
                    [top_square[0][2], top_square[1][2], top_square[2][2], top_square[3][2], top_square[0][2]],
                    'g-')
            
            # Draw lines connecting the squares
            for i in range(4):
                ax2.plot([square_vertices[i][0], top_square[i][0]], 
                        [square_vertices[i][1], top_square[i][1]], 
                        [square_vertices[i][2], top_square[i][2]], 'g--')
            
            ax2.set_title('Step 2: Adding Height')
            
            # Step 3: Complete cube
            ax3 = fig.add_subplot(133, projection='3d')
            self.plot_solid('cube', ax=ax3)
            ax3.set_title('Step 3: Complete Cube')
            
        elif solid_name == 'octahedron':
            # Step 1: Square base
            ax1 = fig.add_subplot(131, projection='3d')
            
            # Define the middle square (in xy-plane)
            r = self.radius
            square_vertices = np.array([
                [r, 0, 0],
                [0, r, 0],
                [-r, 0, 0],
                [0, -r, 0]
            ])
            
            # Plot the square
            ax1.plot([square_vertices[0][0], square_vertices[1][0], square_vertices[2][0], square_vertices[3][0], square_vertices[0][0]],
                    [square_vertices[0][1], square_vertices[1][1], square_vertices[2][1], square_vertices[3][1], square_vertices[0][1]],
                    [square_vertices[0][2], square_vertices[1][2], square_vertices[2][2], square_vertices[3][2], square_vertices[0][2]],
                    'b-')
            
            ax1.scatter(square_vertices[:, 0], square_vertices[:, 1], square_vertices[:, 2], 
                       color='red', s=50)
            
            ax1.set_title('Step 1: Middle Square')
            
            # Step 2: Adding top and bottom points
            ax2 = fig.add_subplot(132, projection='3d')
            
            # Plot the middle square
            ax2.plot([square_vertices[0][0], square_vertices[1][0], square_vertices[2][0], square_vertices[3][0], square_vertices[0][0]],
                    [square_vertices[0][1], square_vertices[1][1], square_vertices[2][1], square_vertices[3][1], square_vertices[0][1]],
                    [square_vertices[0][2], square_vertices[1][2], square_vertices[2][2], square_vertices[3][2], square_vertices[0][2]],
                    'b-')
            
            # Add top and bottom points
            top_point = np.array([0, 0, r])
            bottom_point = np.array([0, 0, -r])
            
            ax2.scatter([top_point[0], bottom_point[0]], 
                       [top_point[1], bottom_point[1]], 
                       [top_point[2], bottom_point[2]], 
                       color='green', s=100)
            
            # Draw lines from middle square to top and bottom
            for vertex in square_vertices:
                ax2.plot([vertex[0], top_point[0]], [vertex[1], top_point[1]], [vertex[2], top_point[2]], 'g--')
                ax2.plot([vertex[0], bottom_point[0]], [vertex[1], bottom_point[1]], [vertex[2], bottom_point[2]], 'g--')
            
            ax2.set_title('Step 2: Adding Top and Bottom Points')
            
            # Step 3: Complete octahedron
            ax3 = fig.add_subplot(133, projection='3d')
            self.plot_solid('octahedron', ax=ax3)
            ax3.set_title('Step 3: Complete Octahedron')
            
        elif solid_name == 'dodecahedron':
            # Step 1: Starting with a cube
            ax1 = fig.add_subplot(131, projection='3d')
            self.plot_solid('cube', ax=ax1, color='lightgray', alpha=0.3)
            ax1.set_title('Step 1: Start with a Cube')
            
            # Step 2: Golden rectangles on each face
            ax2 = fig.add_subplot(132, projection='3d')
            
            # Plot the cube
            self.plot_solid('cube', ax=ax2, color='lightgray', alpha=0.2)
            
            # Approximate golden rectangles on cube faces
            # This is simplified for visualization
            phi = self.phi
            r = self.radius / np.sqrt(3)  # Adjusted for cube
            
            # Vertices for new points in the dodecahedron
            new_points = []
            
            # For each face of the cube, we add a point
            # These points will form part of the dodecahedron
            new_points.extend([
                [r*phi, 0, 0],
                [-r*phi, 0, 0],
                [0, r*phi, 0],
                [0, -r*phi, 0],
                [0, 0, r*phi],
                [0, 0, -r*phi]
            ])
            
            # Plot these new points
            new_points = np.array(new_points)
            ax2.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], 
                       color='green', s=50)
            
            ax2.set_title('Step 2: Golden Rectangles on Cube Faces')
            
            # Step 3: Complete dodecahedron
            ax3 = fig.add_subplot(133, projection='3d')
            self.plot_solid('dodecahedron', ax=ax3)
            ax3.set_title('Step 3: Complete Dodecahedron')
            
        elif solid_name == 'icosahedron':
            # Step 1: Three golden rectangles in perpendicular planes
            ax1 = fig.add_subplot(131, projection='3d')
            
            # Create three golden rectangles in perpendicular planes
            phi = self.phi
            r = self.radius
            
            # Vertices for the golden rectangles
            rect1 = np.array([  # In xy-plane
                [r, 0, 0],
                [0, r*phi, 0],
                [-r, 0, 0],
                [0, -r*phi, 0]
            ])
            
            rect2 = np.array([  # In xz-plane
                [r, 0, 0],
                [0, 0, r*phi],
                [-r, 0, 0],
                [0, 0, -r*phi]
            ])
            
            rect3 = np.array([  # In yz-plane
                [0, r, 0],
                [0, 0, r*phi],
                [0, -r, 0],
                [0, 0, -r*phi]
            ])
            
            # Plot the golden rectangles
            for rect in [rect1, rect2, rect3]:
                ax1.plot([rect[0][0], rect[1][0], rect[2][0], rect[3][0], rect[0][0]],
                        [rect[0][1], rect[1][1], rect[2][1], rect[3][1], rect[0][1]],
                        [rect[0][2], rect[1][2], rect[2][2], rect[3][2], rect[0][2]],
                        'b-')
            
            all_points = np.vstack([rect1, rect2, rect3])
            ax1.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], 
                       color='red', s=50)
            
            ax1.set_title('Step 1: Three Golden Rectangles')
            
            # Step 2: Identifying the 12 vertices from the rectangles
            ax2 = fig.add_subplot(132, projection='3d')
            
            # Identify the 12 unique vertices
            unique_vertices = []
            for point in all_points:
                # Check if this point is close to any vertex we've already found
                new_point = True
                for existing in unique_vertices:
                    if np.linalg.norm(point - existing) < 0.1:
                        new_point = False
                        break
                if new_point:
                    unique_vertices.append(point)
            
            unique_vertices = np.array(unique_vertices)
            
            # Plot these vertices
            ax2.scatter(unique_vertices[:, 0], unique_vertices[:, 1], unique_vertices[:, 2], 
                       color='red', s=100)
            
            # Draw some connecting lines to suggest the icosahedron
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Around equator
                         (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),  # To top
                         (6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]: # To bottom
                # Only for illustration; the actual icosahedron has more complex connectivity
                if i < len(unique_vertices) and j < len(unique_vertices):
                    ax2.plot([unique_vertices[i][0], unique_vertices[j][0]],
                            [unique_vertices[i][1], unique_vertices[j][1]],
                            [unique_vertices[i][2], unique_vertices[j][2]], 'g--')
            
            ax2.set_title('Step 2: Identifying 12 Vertices')
            
            # Step 3: Complete icosahedron
            ax3 = fig.add_subplot(133, projection='3d')
            self.plot_solid('icosahedron', ax=ax3)
            ax3.set_title('Step 3: Complete Icosahedron')
        
        plt.suptitle(f'Ancient Construction of the {solid_name.capitalize()}', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def euclid_elements_book_xiii(self):
        """
        Create an educational display about Euclid's Elements Book XIII,
        which describes the construction of the Platonic solids.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "EUCLID'S ELEMENTS - BOOK XIII", fontsize=18, 
               ha='center', va='top', weight='bold')
        
        ax.text(0.5, 0.90, "The Construction of the Five Regular Solids", fontsize=14, 
               ha='center', va='top', style='italic')
        
        # Main content
        content = """
        Book XIII of Euclid's Elements, written around 300 BCE, is the culmination of his geometric treatise.
        It presents rigorous constructions of all five Platonic solids and demonstrates their mathematical properties.
        
        KEY PROPOSITIONS:
        
        • Proposition 13: Constructing a regular pyramid (tetrahedron) inscribed in a sphere.
        
        • Proposition 14: Constructing a regular octahedron inscribed in a sphere.
        
        • Proposition 15: Constructing a regular cube inscribed in a sphere.
        
        • Proposition 16: Constructing a regular icosahedron inscribed in a sphere, demonstrating
          that its edges relate to the golden ratio.
        
        • Proposition 17: Constructing a regular dodecahedron inscribed in a sphere, again involving
          the golden ratio in its proportions.
        
        • Proposition 18: Comparing the edges of the five solids when inscribed in the same sphere,
          establishing their relationships to one another.
        
        HISTORICAL SIGNIFICANCE:
        
        Euclid's systematic treatment of the Platonic solids represents the culmination of Greek geometry.
        By demonstrating that exactly five regular polyhedra exist and providing rigorous constructions
        for each, Book XIII stands as one of the greatest achievements of ancient mathematics.
        
        The connection between the Platonic solids and the golden ratio (especially in the dodecahedron
        and icosahedron) exemplifies the harmony and elegance that Greek mathematicians sought in geometry.
        
        Plato had earlier associated these five solids with the classical elements in his dialogue Timaeus:
        • Tetrahedron - Fire
        • Octahedron - Air
        • Icosahedron - Water
        • Cube - Earth
        • Dodecahedron - Cosmos (or Aether/Universe)
        
        Euclid's treatment was purely mathematical, providing the rigorous foundation for these philosophical ideas.
        """
        
        ax.text(0.5, 0.5, content, fontsize=12, ha='center', va='center',
               bbox=dict(facecolor='antiquewhite', alpha=0.8, boxstyle='round,pad=1'))
        
        # Footer
        ax.text(0.5, 0.05, "The Elements concludes with the proof that only five regular polyhedra exist.",
               fontsize=10, ha='center', va='bottom', style='italic')
        
        plt.tight_layout()
        return fig


class ArchimedeanSolids:
    """
    A class to construct and visualize Archimedean solids,
    which are the semi-regular polyhedra discovered by Archimedes.
    This is included as a historical extension of the Platonic solids.
    """
    
    def __init__(self):
        """Initialize with basic information about Archimedean solids."""
        self.info = """
        ARCHIMEDEAN SOLIDS
        
        Archimedean solids are semi-regular convex polyhedra composed of two or more types
        of regular polygons meeting in identical vertices. They are named after Archimedes,
        who provided the first known comprehensive study of these solids.
        
        There are 13 Archimedean solids:
        
        1. Truncated Tetrahedron
        2. Truncated Cube
        3. Truncated Octahedron
        4. Truncated Dodecahedron
        5. Truncated Icosahedron
        6. Cuboctahedron
        7. Icosidodecahedron
        8. Rhombicuboctahedron
        9. Rhombicosidodecahedron
        10. Truncated Cuboctahedron
        11. Truncated Icosidodecahedron
        12. Snub Cube
        13. Snub Dodecahedron
        
        HISTORICAL CONTEXT:
        
        Archimedes' original work on these solids was lost, but was referenced by Pappus of Alexandria.
        The solids were rediscovered during the Renaissance, notably by Kepler in his book Harmonices Mundi.
        
        These solids represent an important extension of the Platonic solids and demonstrate
        how truncation and other operations can generate new regular polyhedra.
        """
    
    def info_display(self):
        """Create an informational display about Archimedean solids."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        ax.text(0.5, 0.5, self.info, fontsize=12, ha='center', va='center',
               bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=1'))
        
        ax.set_title("Archimedean Solids: Historical Extension of the Platonic Solids", 
                    fontsize=14, weight='bold')
        
        return fig


class HistoricalContext:
    """
    A class to provide historical context for the ancient
    geometric construction of regular solids.
    """
    
    def __init__(self):
        """Initialize with historical information."""
        pass
    
    def create_timeline(self):
        """Create a timeline of key developments in the study of regular solids."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "TIMELINE: THE STUDY OF REGULAR SOLIDS IN ANCIENT MATHEMATICS", 
               fontsize=16, ha='center', va='top', weight='bold')
        
        # Timeline events
        events = [
            (-530, "Pythagoras establishes his school, where regular polyhedra are first studied systematically."),
            (-450, "Empedocles develops the theory of four elements, later connected to the Platonic solids."),
            (-400, "Theaetetus (a collaborator of Plato) proves that only five regular polyhedra exist."),
            (-380, "Plato writes Timaeus, associating regular solids with the classical elements."),
            (-350, "Aristotle adds aether as the fifth element, associated with the dodecahedron."),
            (-300, "Euclid writes Elements XIII, providing rigorous constructions of all five Platonic solids."),
            (-250, "Archimedes discovers the 13 semi-regular polyhedra now known as Archimedean solids."),
            (-150, "Hypsicles adds a fourteenth book to the Elements, further exploring regular solids."),
            (150, "Ptolemy studies the properties and relations of inscribed regular polyhedra."),
            (320, "Pappus of Alexandria preserves much of the knowledge about regular polyhedra in his Collection.")
        ]
        
        # Plot timeline
        timeline_y = 0.5
        ax.axhline(y=timeline_y, xmin=0.05, xmax=0.95, color='black', linewidth=2)
        
        # Add events
        min_year = min(events, key=lambda x: x[0])[0]
        max_year = max(events, key=lambda x: x[0])[0]
        year_range = max_year - min_year
        
        for year, description in events:
            # Normalize position on timeline
            x_pos = 0.05 + 0.9 * (year - min_year) / year_range
            
            # Alternate above and below the timeline
            if events.index((year, description)) % 2 == 0:
                y_pos = timeline_y + 0.1
                alignment = 'bottom'
                y_line = np.linspace(timeline_y, y_pos - 0.02, 100)
            else:
                y_pos = timeline_y - 0.1
                alignment = 'top'
                y_line = np.linspace(timeline_y, y_pos + 0.02, 100)
            
            # Draw line to event
            x_line = np.ones(100) * x_pos
            ax.plot(x_line, y_line, 'k-', alpha=0.5)
            
            # Add event marker
            ax.plot(x_pos, timeline_y, 'o', markersize=8, color='blue')
            
            # Add event text
            year_text = f"{abs(year)} BCE" if year < 0 else f"{year} CE"
            ax.text(x_pos, y_pos, f"{year_text}\n{description}", ha='center', va=alignment,
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Add explanation
        explanation = """
        The study of regular polyhedra spans over a millennium in ancient mathematics,
        from the early Pythagoreans to the late Greek mathematicians of Alexandria.
        
        The construction of these perfect geometric forms was seen as a way to understand
        the fundamental structure of the universe. The fact that exactly five regular 
        polyhedra exist was considered a profound mathematical truth with cosmic significance.
        
        This timeline highlights key developments in the understanding and construction of 
        these solids, culminating in Euclid's systematic treatment in Book XIII of the Elements.
        """
        
        ax.text(0.5, 0.12, explanation, fontsize=10, ha='center', va='center',
               bbox=dict(facecolor='lightyellow', alpha=0.7, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        return fig
    
    def platonic_philosophy(self):
        """Create a display about the philosophical significance of the Platonic solids."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "THE COSMIC SIGNIFICANCE OF REGULAR SOLIDS IN PLATONIC PHILOSOPHY", 
               fontsize=16, ha='center', va='top', weight='bold')
        
        # Content
        content = """
        PLATO'S TIMAEUS AND THE ELEMENTS
        
        In his dialogue Timaeus, Plato presents a cosmological account in which the five regular solids
        play a central role in the physical structure of the universe. Each solid is associated with
        one of the classical elements:
        
        • TETRAHEDRON - FIRE: The tetrahedron is assigned to fire because it has the fewest faces
          and sharp angles, making it the most mobile and penetrating of the solids, just as fire
          is the most mobile and penetrating of the elements.
        
        • OCTAHEDRON - AIR: The octahedron represents air because it is the second most mobile
          solid after the tetrahedron, corresponding to air's lightness and mobility.
        
        • ICOSAHEDRON - WATER: The icosahedron, with its many faces, moves less easily than the
          octahedron but more easily than the cube, corresponding to water's fluidity yet greater
          density than air.
        
        • CUBE - EARTH: The cube is associated with earth because its square faces make it the most
          stable and least mobile of the solids, just as earth is the most stable element.
        
        • DODECAHEDRON - COSMOS: The dodecahedron is special, representing the cosmos as a whole.
          Its twelve pentagonal faces were associated with the zodiac, suggesting that this shape
          encompasses the entire universe.
        
        MATHEMATICAL PERFECTION AS COSMIC TRUTH
        
        For Plato and his followers, mathematical forms represented a higher reality of unchanging
        truth. The discovery that exactly five perfect solids exist was seen as profound evidence
        of mathematical harmony underlying the physical world.
        
        The fact that these five solids could be constructed using only straightedge and compass
        (the tools of pure geometry) reinforced their special status. By understanding these
        perfect mathematical forms, the philosopher could gain insight into the fundamental
        structure of reality itself.
        
        LEGACY
        
        This Platonic vision of a cosmos structured according to perfect mathematical forms had
        a profound influence on Western thought for nearly two millennia. The connection between
        mathematical harmony and cosmic structure would later inspire astronomers like Kepler,
        who initially tried to explain the solar system using nested Platonic solids.
        """
        
        ax.text(0.5, 0.5, content, fontsize=12, ha='center', va='center',
               bbox=dict(facecolor='antiquewhite', alpha=0.8, boxstyle='round,pad=1'))
        
        plt.tight_layout()
        return fig


def main():
    """
    Main function to demonstrate the construction and properties
    of Platonic solids.
    """
    # Create the Platonic solids
    solids = PlatonicSolids()
    
    print("=== PLATONIC SOLIDS CONSTRUCTION ===")
    print("This program demonstrates the ancient methods")
    print("for constructing the five regular (Platonic) solids.")
    print("These constructions were described in Book XIII of Euclid's Elements.")
    print()
    
    # Display information about each solid
    for solid_name in ['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron']:
        print(solids.get_solid_info(solid_name))
        print()
    
    # Check the Euler characteristic for all solids
    print("=== VERIFICATION OF EULER'S FORMULA ===")
    print("Euler's formula states that for any convex polyhedron:")
    print("Vertices - Edges + Faces = 2")
    print()
    
    for solid_name in ['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron']:
        props = solids.properties[solid_name]
        print(f"{solid_name.capitalize()}: {props['vertex_count']} vertices - "
              f"{props['edge_count']} edges + {props['face_count']} faces = "
              f"{props['euler_characteristic']}")
    
    print()
    print("=== GENERATING VISUALIZATIONS ===")
    
    # Create visualizations
    print("1. Plotting all five Platonic solids...")
    all_solids_fig = solids.plot_all_solids()
    
    print("2. Plotting dual pairs...")
    dual_figs = {}
    for solid_name in ['tetrahedron', 'cube', 'dodecahedron']:
        dual_figs[solid_name] = solids.plot_dual_pair(solid_name)
    
    print("3. Plotting nested duals...")
    nested_figs = {}
    for solid_name in ['tetrahedron', 'cube', 'dodecahedron']:
        nested_figs[solid_name] = solids.plot_nested_dual(solid_name)
    
    print("4. Plotting construction steps...")
    construction_figs = {}
    for solid_name in ['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron']:
        construction_figs[solid_name] = solids.plot_construction_steps(solid_name)
    
    print("5. Creating Euclid's Elements Book XIII display...")
    euclid_fig = solids.euclid_elements_book_xiii()
    
    print("6. Creating historical context...")
    historical = HistoricalContext()
    timeline_fig = historical.create_timeline()
    philosophy_fig = historical.platonic_philosophy()
    
    print("7. Information about Archimedean solids...")
    archimedean = ArchimedeanSolids()
    archimedean_fig = archimedean.info_display()
    
    print("8. Creating interactive demonstration...")
    interactive_fig = solids.interactive_solid_explorer()
    
    print("9. Creating animations...")
    animations = {}
    for solid_name in ['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron']:
        animations[solid_name] = solids.animate_rotation(solid_name)
    
    print("\nAll visualizations created. Displaying plots...")
    plt.show()


if __name__ == "__main__":
    main()