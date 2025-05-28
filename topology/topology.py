import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TopologicalSpace:
    """
    A class to represent topological properties preserved under continuous deformations.
    These properties remain unchanged when a shape undergoes stretching, twisting, 
    crumpling, or bending without closing/opening holes, tearing, gluing, or passing 
    through itself.
    """
    
    def __init__(self, vertices, edges=None, faces=None):
        """
        Initialize a topological space with vertices, edges, and faces.
        
        Parameters:
        ----------
        vertices : ndarray of shape (n, d)
            Coordinates of n vertices in d-dimensional space
        edges : list of tuples, optional
            List of pairs of vertex indices representing edges
        faces : list of tuples, optional
            List of triplets of vertex indices representing triangular faces
        """
        self.vertices = np.array(vertices)
        
        # If edges and faces are not provided, triangulate the points
        if edges is None or faces is None:
            if self.vertices.shape[1] == 2:  # 2D points
                triangulation = Delaunay(self.vertices)
                self.faces = triangulation.simplices
                
                # Extract edges from faces
                self.edges = set()
                for face in self.faces:
                    self.edges.add(tuple(sorted([face[0], face[1]])))
                    self.edges.add(tuple(sorted([face[1], face[2]])))
                    self.edges.add(tuple(sorted([face[2], face[0]])))
                self.edges = list(self.edges)
            else:
                # For higher dimensions, we'd need more complex triangulation methods
                raise ValueError("Automatic triangulation is only supported for 2D points. Please provide edges and faces.")
        else:
            self.edges = edges
            self.faces = faces
    
    def euler_characteristic(self):
        """
        Compute the Euler characteristic: χ = V - E + F
        This is a topological invariant preserved under continuous deformations.
        
        Returns:
        -------
        int
            The Euler characteristic of the space
        """
        V = len(self.vertices)
        E = len(self.edges)
        F = len(self.faces) if self.faces is not None else 0
        
        return V - E + F
    
    def betti_numbers(self, max_dim=2):
        """
        Compute the Betti numbers up to a specified dimension.
        
        Betti numbers count the number of k-dimensional "holes" in a space:
        - β₀: number of connected components
        - β₁: number of 1D holes (like in a circle)
        - β₂: number of 2D voids (like inside a sphere)
        
        Returns:
        -------
        list
            List of Betti numbers [β₀, β₁, ..., βₙ]
        """
        betti = [0] * (max_dim + 1)
        
        # β₀: number of connected components
        G = nx.Graph()
        G.add_nodes_from(range(len(self.vertices)))
        G.add_edges_from(self.edges)
        betti[0] = nx.number_connected_components(G)
        
        # For a closed surface with genus g, we have:
        # χ = 2 - 2g
        # β₀ = 1 (connected)
        # β₁ = 2g
        # β₂ = 1 (assuming it's closed)
        if max_dim >= 1 and self.is_closed_surface():
            genus = self.genus()
            betti[1] = 2 * genus
            
            if max_dim >= 2:
                betti[2] = 1  # For a closed oriented surface
        
        return betti
    
    def is_connected(self):
        """
        Check if the space is connected.
        Connectedness is preserved under continuous deformations.
        
        Returns:
        -------
        bool
            True if connected, False otherwise
        """
        G = nx.Graph()
        G.add_nodes_from(range(len(self.vertices)))
        G.add_edges_from(self.edges)
        
        return nx.is_connected(G)
    
    def is_closed_surface(self):
        """
        Check if the space represents a closed surface.
        
        Returns:
        -------
        bool
            True if it's a closed surface, False otherwise
        """
        # Each edge should be shared by exactly two faces in a closed surface
        edge_count = {}
        for face in self.faces:
            edge1 = tuple(sorted([face[0], face[1]]))
            edge2 = tuple(sorted([face[1], face[2]]))
            edge3 = tuple(sorted([face[2], face[0]]))
            
            for edge in [edge1, edge2, edge3]:
                if edge in edge_count:
                    edge_count[edge] += 1
                else:
                    edge_count[edge] = 1
        
        # Check if all edges are shared by exactly 2 faces
        return all(count == 2 for count in edge_count.values())
    
    def genus(self):
        """
        Compute the genus of a closed surface.
        The genus (number of "handles") is a topological invariant.
        
        For a closed surface: χ = 2 - 2g, where g is the genus.
        
        Returns:
        -------
        int
            The genus of the surface
        """
        if not self.is_closed_surface():
            raise ValueError("Genus is only defined for closed surfaces.")
        
        chi = self.euler_characteristic()
        genus = (2 - chi) // 2
        
        return genus
    
    def is_orientable(self):
        """
        Check if the surface is orientable.
        Orientability is preserved under continuous deformations.
        
        Returns:
        -------
        bool
            True if orientable, False otherwise
        """
        
        # Create a graph where nodes are faces and edges connect adjacent faces
        G = nx.Graph()
        G.add_nodes_from(range(len(self.faces)))
        
        # Map edges to faces they belong to
        edge_to_faces = {}
        for i, face in enumerate(self.faces):
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            
            for edge in edges:
                if edge in edge_to_faces:
                    edge_to_faces[edge].append(i)
                else:
                    edge_to_faces[edge] = [i]
        
        # Connect adjacent faces
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                G.add_edge(faces[0], faces[1])
        
        # Try to assign orientations (colors) to faces
        try:
            nx.bipartite.color(G)
            return True
        except:
            return False  # Not bipartite, so not orientable
    
    def fundamental_group_generators(self, base_point=0):
        """
        Compute generators of the fundamental group.
        The fundamental group is a topological invariant.
        
        Returns:
        -------
        list
            List of loops representing generators of the fundamental group
        """
        # Create a graph from vertices and edges
        G = nx.Graph()
        G.add_nodes_from(range(len(self.vertices)))
        G.add_edges_from(self.edges)
        
        # Compute a spanning tree
        T = nx.minimum_spanning_tree(G)
        
        # Each edge not in the spanning tree creates a generator
        generators = []
        for edge in self.edges:
            if not T.has_edge(edge[0], edge[1]):
                # Find path in spanning tree from base_point to edge[0]
                path1 = nx.shortest_path(T, base_point, edge[0])
                
                # Edge from edge[0] to edge[1]
                path2 = [edge[1]]
                
                # Find path in spanning tree from edge[1] to base_point
                path3 = nx.shortest_path(T, edge[1], base_point)
                
                # Combine to form a loop
                loop = path1 + path2 + path3
                generators.append(loop)
        
        return generators
    
    def visualize(self, dim=3):
        """
        Visualize the topological space.
        
        Parameters:
        ----------
        dim : int, optional
            Dimension for visualization (2 or 3)
        """
        if dim == 2:
            plt.figure(figsize=(8, 8))
            
            # Plot vertices
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1], c='b', s=50)
            
            # Plot edges
            for edge in self.edges:
                plt.plot([self.vertices[edge[0], 0], self.vertices[edge[1], 0]],
                         [self.vertices[edge[0], 1], self.vertices[edge[1], 1]], 'k-')
            
            # Plot faces
            if self.faces is not None:
                for face in self.faces:
                    plt.fill([self.vertices[face[0], 0], self.vertices[face[1], 0], self.vertices[face[2], 0]],
                             [self.vertices[face[0], 1], self.vertices[face[1], 1], self.vertices[face[2], 1]],
                             alpha=0.3, c='r')
            
            plt.axis('equal')
            plt.grid(True)
            plt.show()
        
        elif dim == 3:
            if self.vertices.shape[1] < 3:
                raise ValueError("3D visualization requires 3D vertices.")
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot vertices
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], c='b', s=50)
            
            # Plot edges
            for edge in self.edges:
                ax.plot([self.vertices[edge[0], 0], self.vertices[edge[1], 0]],
                        [self.vertices[edge[0], 1], self.vertices[edge[1], 1]],
                        [self.vertices[edge[0], 2], self.vertices[edge[1], 2]], 'k-')
            
            # Plot faces
            if self.faces is not None:
                for face in self.faces:
                    verts = [
                        (self.vertices[face[0], 0], self.vertices[face[0], 1], self.vertices[face[0], 2]),
                        (self.vertices[face[1], 0], self.vertices[face[1], 1], self.vertices[face[1], 2]),
                        (self.vertices[face[2], 0], self.vertices[face[2], 1], self.vertices[face[2], 2])
                    ]
                    ax.add_collection3d(plt.matplotlib.tri.art3d.Poly3DCollection([verts], alpha=0.3, color='r'))
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        
        else:
            raise ValueError("Visualization is only supported for 2D and 3D.")


# Example shape creation functions

def create_torus(R=3, r=1, n=20, m=10):
    """Create a triangulated torus."""
    vertices = []
    for i in range(n):
        phi = 2 * np.pi * i / n
        for j in range(m):
            theta = 2 * np.pi * j / m
            x = (R + r * np.cos(theta)) * np.cos(phi)
            y = (R + r * np.cos(theta)) * np.sin(phi)
            z = r * np.sin(theta)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Create faces (triangulation)
    faces = []
    for i in range(n):
        for j in range(m):
            v0 = i * m + j
            v1 = i * m + (j + 1) % m
            v2 = ((i + 1) % n) * m + j
            v3 = ((i + 1) % n) * m + (j + 1) % m
            
            faces.append((v0, v1, v2))
            faces.append((v1, v3, v2))
    
    # Create edges from faces
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    
    return vertices, list(edges), faces

def create_sphere(radius=1, resolution=20):
    """Create a triangulated sphere."""
    vertices = []
    for i in range(resolution):
        theta = np.pi * i / (resolution - 1)
        for j in range(resolution):
            phi = 2 * np.pi * j / (resolution - 1)
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Create triangulation
    tri = Delaunay(vertices)
    faces = tri.simplices
    
    # Create edges from faces
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    
    return vertices, list(edges), faces

def create_klein_bottle(R=3, r=1, n=20, m=10):
    """Create a triangulated Klein bottle (non-orientable surface)."""
    vertices = []
    for i in range(n):
        u = i / n * 2 * np.pi
        for j in range(m):
            v = j / m * 2 * np.pi
            
            # Parametrization of Klein bottle
            x = (R + r * np.cos(v) * np.cos(u/2)) * np.cos(u)
            y = (R + r * np.cos(v) * np.cos(u/2)) * np.sin(u)
            z = r * np.sin(v) * np.cos(u/2)
            
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Create faces (triangulation)
    faces = []
    for i in range(n):
        for j in range(m):
            v0 = i * m + j
            v1 = i * m + (j + 1) % m
            v2 = ((i + 1) % n) * m + j
            v3 = ((i + 1) % n) * m + (j + 1) % m
            
            if i == n - 1:  # Special case for the twist
                v2 = ((i + 1) % n) * m + (m - j) % m
                v3 = ((i + 1) % n) * m + (m - (j + 1)) % m
            
            faces.append((v0, v1, v2))
            faces.append((v1, v3, v2))
    
    # Create edges from faces
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    
    return vertices, list(edges), faces

def create_mobius_strip(R=3, width=1, n=20, m=10):
    """Create a triangulated Möbius strip (non-orientable surface with boundary)."""
    vertices = []
    for i in range(n):
        u = i / n * 2 * np.pi
        for j in range(m):
            v = (j / (m - 1) - 0.5) * width
            
            # Parametrization of Möbius strip
            x = (R + v * np.cos(u/2)) * np.cos(u)
            y = (R + v * np.cos(u/2)) * np.sin(u)
            z = v * np.sin(u/2)
            
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Create faces (triangulation)
    faces = []
    for i in range(n-1):
        for j in range(m-1):
            v0 = i * m + j
            v1 = i * m + (j + 1)
            v2 = ((i + 1) % n) * m + j
            v3 = ((i + 1) % n) * m + (j + 1)
            
            faces.append((v0, v1, v2))
            faces.append((v1, v3, v2))
    
    # Special case for the twist
    for j in range(m-1):
        v0 = (n-1) * m + j
        v1 = (n-1) * m + (j + 1)
        v2 = (m - j - 1)
        v3 = (m - j - 2)
        
        faces.append((v0, v1, v2))
        faces.append((v1, v3, v2))
    
    # Create edges from faces
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    
    return vertices, list(edges), faces


# Example usage
if __name__ == "__main__":
    # Create different topological objects
    vertices, edges, faces = create_sphere()
    sphere = TopologicalSpace(vertices, edges, faces)
    
    vertices, edges, faces = create_torus()
    torus = TopologicalSpace(vertices, edges, faces)
    
    vertices, edges, faces = create_klein_bottle()
    klein = TopologicalSpace(vertices, edges, faces)
    
    vertices, edges, faces = create_mobius_strip()
    mobius = TopologicalSpace(vertices, edges, faces)
    
    # Print topological invariants for each shape
    print("Sphere:")
    print(f"Euler characteristic: {sphere.euler_characteristic()}")
    print(f"Betti numbers: {sphere.betti_numbers()}")
    print(f"Is orientable: {sphere.is_orientable()}")
    
    print("\nTorus:")
    print(f"Euler characteristic: {torus.euler_characteristic()}")
    print(f"Betti numbers: {torus.betti_numbers()}")
    print(f"Genus: {torus.genus()}")
    print(f"Is orientable: {torus.is_orientable()}")
    
    print("\nKlein bottle:")
    print(f"Euler characteristic: {klein.euler_characteristic()}")
    print(f"Is orientable: {klein.is_orientable()}")
    
    print("\nMöbius strip:")
    print(f"Euler characteristic: {mobius.euler_characteristic()}")
    print(f"Is orientable: {mobius.is_orientable()}")
    
    # Visualize the shapes
    sphere.visualize(dim=3)
    torus.visualize(dim=3)
    klein.visualize(dim=3)
    mobius.visualize(dim=3)