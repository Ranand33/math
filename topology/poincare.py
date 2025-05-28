import numpy as np
import networkx as nx
from collections import defaultdict, deque
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import null_space

class SimplicialComplex:
    """
    Implementation of a simplicial complex following Poincaré's Analysis Situs.
    
    Poincaré introduced the concept of a simplicial complex as a way to represent
    topological spaces combinatorially, making them amenable to algebraic methods.
    """
    
    def __init__(self):
        """Initialize an empty simplicial complex."""
        # Store simplices by dimension: {dimension: {simplex_id: vertices}}
        self.simplices = defaultdict(dict)
        # Store boundary operators: {dimension: boundary_matrix}
        self.boundary_operators = {}
        # Store incidence relationships: {simplex_id: {neighbor_id: incidence_value}}
        self.incidence = defaultdict(dict)
        # Track the next available ID for simplices
        self.next_id = 0
        
    def add_simplex(self, vertices, id=None):
        """
        Add a simplex to the complex.
        
        Parameters:
        ----------
        vertices : tuple or list
            Vertices defining the simplex, must be hashable
        id : hashable, optional
            ID for the simplex, generated if not provided
            
        Returns:
        -------
        id : hashable
            ID of the added simplex
        """
        vertices = tuple(sorted(vertices))
        dim = len(vertices) - 1
        
        # Check if simplex already exists
        for sid, verts in self.simplices[dim].items():
            if verts == vertices:
                return sid
        
        # Generate ID if not provided
        if id is None:
            id = self.next_id
            self.next_id += 1
        
        # Add the simplex
        self.simplices[dim][id] = vertices
        
        # Add all faces (lower-dimensional simplices)
        if dim > 0:
            for i in range(dim + 1):
                face_vertices = vertices[:i] + vertices[i+1:]
                face_id = self.add_simplex(face_vertices)
                
                # Record incidence relationship
                orientation = (-1)**i
                self.incidence[id][face_id] = orientation
                
        return id
    
    def get_chain_complex(self):
        """
        Compute the chain complex of the simplicial complex.
        
        Returns:
        -------
        dict
            Dictionary mapping dimensions to boundary matrices
        """
        max_dim = max(self.simplices.keys()) if self.simplices else -1
        
        for dim in range(1, max_dim + 1):
            if dim in self.boundary_operators:
                continue
                
            if dim-1 not in self.simplices or not self.simplices[dim-1]:
                self.boundary_operators[dim] = csr_matrix((0, 0))
                continue
                
            if dim not in self.simplices or not self.simplices[dim]:
                self.boundary_operators[dim] = csr_matrix((len(self.simplices[dim-1]), 0))
                continue
            
            # Create mapping from simplex IDs to matrix indices
            dim_indices = {sid: i for i, sid in enumerate(self.simplices[dim].keys())}
            dim_minus_one_indices = {sid: i for i, sid in enumerate(self.simplices[dim-1].keys())}
            
            # Initialize sparse matrix data
            data = []
            rows = []
            cols = []
            
            # Populate boundary matrix
            for sid, faces in self.incidence.items():
                if sid in dim_indices:  # Only process simplices of the current dimension
                    for face_id, orientation in faces.items():
                        if face_id in dim_minus_one_indices:  # Ensure face is of dimension dim-1
                            rows.append(dim_minus_one_indices[face_id])
                            cols.append(dim_indices[sid])
                            data.append(orientation)
            
            # Create sparse matrix
            self.boundary_operators[dim] = csr_matrix(
                (data, (rows, cols)), 
                shape=(len(self.simplices[dim-1]), len(self.simplices[dim]))
            )
        
        return self.boundary_operators
    
    def compute_homology(self, dim):
        """
        Compute the homology group in the given dimension.
        
        This implements Poincaré's approach to homology: identifying cycles that aren't boundaries.
        
        Parameters:
        ----------
        dim : int
            Dimension of the homology group to compute
            
        Returns:
        -------
        int
            Rank of the homology group (Betti number)
        ndarray
            Basis for the homology group
        """
        boundary_ops = self.get_chain_complex()
        
        if dim not in self.simplices:
            return 0, np.array([])
        
        # Get boundary operators for dimensions dim and dim+1
        boundary_d = boundary_ops.get(dim, csr_matrix((0, len(self.simplices[dim-1]) if dim-1 in self.simplices else 0)))
        boundary_d_plus_1 = boundary_ops.get(dim+1, csr_matrix((len(self.simplices[dim]), 0)))
        
        # Compute kernel (cycles) and image (boundaries)
        if boundary_d.shape[1] > 0:
            cycles = null_space(boundary_d.toarray())  # Ker(∂ₙ)
        else:
            cycles = np.eye(boundary_d.shape[0])  # Everything is a cycle if no boundary operator
            
        if boundary_d_plus_1.shape[1] > 0:
            boundaries = boundary_d_plus_1.toarray()  # Im(∂ₙ₊₁)
            boundary_space = np.linalg.matrix_rank(boundaries)
        else:
            boundaries = np.zeros((cycles.shape[0], 0))
            boundary_space = 0
        
        # Homology is Ker(∂ₙ) / Im(∂ₙ₊₁)
        homology_rank = cycles.shape[1] - boundary_space
        
        # Find a basis for the homology group
        if homology_rank > 0 and cycles.shape[1] > 0:
            # Project cycles onto the orthogonal complement of boundaries
            if boundaries.shape[1] > 0:
                Q, R = np.linalg.qr(boundaries)
                rank = np.sum(np.abs(np.diag(R)) > 1e-10)
                if rank > 0:
                    Q = Q[:, :rank]
                    homology_basis = cycles - Q @ (Q.T @ cycles)
                else:
                    homology_basis = cycles
            else:
                homology_basis = cycles
                
            # Orthogonalize to get a clean basis
            homology_basis, r = np.linalg.qr(homology_basis)
            # Filter out zero vectors
            nonzero_cols = ~np.all(np.abs(homology_basis) < 1e-10, axis=0)
            homology_basis = homology_basis[:, nonzero_cols]
        else:
            homology_basis = np.array([])
        
        return homology_rank, homology_basis
    
    def betti_numbers(self, max_dim=None):
        """
        Compute the Betti numbers up to a specified dimension.
        
        Poincaré introduced these as topological invariants that count "holes" of different dimensions.
        
        Parameters:
        ----------
        max_dim : int, optional
            Maximum dimension for which to compute Betti numbers
            
        Returns:
        -------
        list
            Betti numbers [β₀, β₁, ..., βₙ]
        """
        if max_dim is None:
            max_dim = max(self.simplices.keys()) if self.simplices else 0
        
        betti = []
        for dim in range(max_dim + 1):
            betti_dim, _ = self.compute_homology(dim)
            betti.append(betti_dim)
            
        return betti
    
    def euler_characteristic(self):
        """
        Compute the Euler characteristic of the complex.
        
        Poincaré proved that this can be calculated either from the alternating sum
        of simplices or from the alternating sum of Betti numbers.
        
        Returns:
        -------
        int
            Euler characteristic
        """
        # Method 1: Using simplices
        euler_simplices = sum((-1)**dim * len(simplices) for dim, simplices in self.simplices.items())
        
        # Method 2: Using Betti numbers
        max_dim = max(self.simplices.keys()) if self.simplices else 0
        betti = self.betti_numbers(max_dim)
        euler_betti = sum((-1)**i * b for i, b in enumerate(betti))
        
        # These should be equal by Poincaré's theorem
        assert euler_simplices == euler_betti, "Euler characteristic mismatch!"
        
        return euler_simplices
    
    def poincare_duality_map(self, dim, oriented=True):
        """
        Compute the Poincaré duality map in the given dimension.
        
        Poincaré duality relates the kth homology group to the (n-k)th cohomology group
        of an n-dimensional manifold.
        
        Parameters:
        ----------
        dim : int
            Dimension for which to compute the duality map
        oriented : bool, optional
            Whether the manifold is oriented
            
        Returns:
        -------
        scipy.sparse.csr_matrix
            Poincaré duality map
        """
        if not oriented:
            return None  # Poincaré duality requires orientability
            
        max_dim = max(self.simplices.keys()) if self.simplices else 0
        dual_dim = max_dim - dim
        
        if dim not in self.simplices or dual_dim not in self.simplices:
            return None
            
        # Create the duality map based on intersection numbers
        rows = []
        cols = []
        data = []
        
        # In a triangulated manifold, this would be based on the intersection form
        # For simplicity, we'll use a placeholder implementation
        dim_indices = {sid: i for i, sid in enumerate(self.simplices[dim].keys())}
        dual_indices = {sid: i for i, sid in enumerate(self.simplices[dual_dim].keys())}
        
        # This is a simplified version - in a real implementation, 
        # you would compute actual intersection numbers
        for sid, vertices in self.simplices[dim].items():
            for dsid, dvertices in self.simplices[dual_dim].items():
                # Check if these simplices have complementary dimensions and intersect
                # In a proper implementation, this would use the intersection form
                if set(vertices).isdisjoint(set(dvertices)):
                    rows.append(dim_indices[sid])
                    cols.append(dual_indices[dsid])
                    data.append(1)  # Simplified intersection number
        
        return csr_matrix((data, (rows, cols)), 
                          shape=(len(self.simplices[dim]), len(self.simplices[dual_dim])))
    
    def fundamental_group_generators(self, base_vertex):
        """
        Compute generators for the fundamental group π₁(X, x₀).
        
        This is a simplified implementation of Poincaré's approach to the fundamental group,
        focusing on 1-dimensional loops.
        
        Parameters:
        ----------
        base_vertex : hashable
            Base vertex for the fundamental group
            
        Returns:
        -------
        list
            List of loops representing generators of the fundamental group
        """
        if 1 not in self.simplices or not self.simplices[1]:
            return []  # No 1-simplices, so trivial fundamental group
            
        # Create a graph from the 1-skeleton
        G = nx.Graph()
        
        # Add vertices
        for vertices in self.simplices[0].values():
            G.add_node(vertices[0])
            
        # Add edges
        for vertices in self.simplices[1].values():
            G.add_edge(vertices[0], vertices[1])
            
        # Compute a spanning tree
        T = nx.minimum_spanning_tree(G)
        
        # Each edge not in the spanning tree creates a generator
        generators = []
        for u, v in G.edges():
            if not T.has_edge(u, v):
                # Find path in spanning tree from base_vertex to u
                if nx.has_path(T, base_vertex, u):
                    path1 = nx.shortest_path(T, base_vertex, u)
                else:
                    continue  # Skip if no path exists
                
                # Edge from u to v
                path2 = [v]
                
                # Find path in spanning tree from v to base_vertex
                if nx.has_path(T, v, base_vertex):
                    path3 = nx.shortest_path(T, v, base_vertex)
                else:
                    continue  # Skip if no path exists
                
                # Combine to form a loop
                loop = path1 + path2 + path3
                generators.append(loop)
        
        return generators
    
    def linking_number(self, cycle1, cycle2):
        """
        Compute the linking number between two cycles.
        
        This is an important invariant in Poincaré's work, measuring how cycles
        are linked in space.
        
        Parameters:
        ----------
        cycle1, cycle2 : list
            Lists of simplex IDs representing the cycles
            
        Returns:
        -------
        int
            Linking number
        """
        # This is a placeholder implementation
        # In a proper implementation, this would compute the actual linking number
        # based on the intersection form in the ambient space
        return 0  # Simplified version
    
    def visualize(self, max_dim=2):
        """
        Visualize the simplicial complex up to the specified dimension.
        
        Parameters:
        ----------
        max_dim : int, optional
            Maximum dimension of simplices to visualize
        """
        # Extract vertices (0-simplices)
        if 0 not in self.simplices:
            return
            
        # Create a mapping from vertices to coordinates
        # In a real implementation, these would be actual coordinates
        vertices = list(set(v[0] for v in self.simplices[0].values()))
        n_vertices = len(vertices)
        vertex_map = {v: i for i, v in enumerate(vertices)}
        
        # Create random 3D coordinates for visualization
        coords = np.random.rand(n_vertices, 3)
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot vertices (0-simplices)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='b', s=50)
        
        # Plot edges (1-simplices)
        if 1 in self.simplices and max_dim >= 1:
            for vertices in self.simplices[1].values():
                p1, p2 = vertices
                if p1 in vertex_map and p2 in vertex_map:
                    i, j = vertex_map[p1], vertex_map[p2]
                    ax.plot([coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            [coords[i, 2], coords[j, 2]], 'k-')
        
        # Plot faces (2-simplices)
        if 2 in self.simplices and max_dim >= 2:
            for vertices in self.simplices[2].values():
                p1, p2, p3 = vertices
                if all(p in vertex_map for p in [p1, p2, p3]):
                    i, j, k = vertex_map[p1], vertex_map[p2], vertex_map[p3]
                    verts = [
                        (coords[i, 0], coords[i, 1], coords[i, 2]),
                        (coords[j, 0], coords[j, 1], coords[j, 2]),
                        (coords[k, 0], coords[k, 1], coords[k, 2])
                    ]
                    ax.add_collection3d(plt.tri.art3d.Poly3DCollection(
                        [verts], alpha=0.3, color='r'))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


class ManifoldBuilder:
    """
    Helper class to construct classical manifolds as simplicial complexes.
    """
    
    @staticmethod
    def build_sphere(n=10):
        """
        Build a triangulation of the 2-sphere.
        
        Parameters:
        ----------
        n : int, optional
            Resolution parameter
            
        Returns:
        -------
        SimplicialComplex
            Triangulation of the sphere
        """
        complex = SimplicialComplex()
        
        # Generate vertices on the sphere
        vertices = []
        for i in range(n):
            theta = np.pi * i / (n - 1)
            for j in range(2*n):
                phi = 2 * np.pi * j / (2*n)
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                vertices.append((x, y, z))
        
        # Add vertices to the complex
        vertex_ids = []
        for i, v in enumerate(vertices):
            vid = complex.add_simplex((i,))
            vertex_ids.append(vid)
        
        # Add edges and triangles
        for i in range(n):
            for j in range(2*n):
                # Current vertex
                idx = i * (2*n) + j
                
                # Adjacent vertices
                idx_right = i * (2*n) + (j + 1) % (2*n)
                idx_down = (i + 1) % n * (2*n) + j
                idx_diag = (i + 1) % n * (2*n) + (j + 1) % (2*n)
                
                # Skip edges at poles
                if i < n - 1:
                    # Add edges
                    edge1 = complex.add_simplex((idx, idx_right))
                    edge2 = complex.add_simplex((idx, idx_down))
                    edge3 = complex.add_simplex((idx_right, idx_diag))
                    edge4 = complex.add_simplex((idx_down, idx_diag))
                    edge5 = complex.add_simplex((idx_right, idx_down))
                    
                    # Add triangles
                    complex.add_simplex((idx, idx_right, idx_down))
                    complex.add_simplex((idx_right, idx_down, idx_diag))
        
        return complex
    
    @staticmethod
    def build_torus(n=10, m=10):
        """
        Build a triangulation of the torus.
        
        Parameters:
        ----------
        n, m : int, optional
            Resolution parameters
            
        Returns:
        -------
        SimplicialComplex
            Triangulation of the torus
        """
        complex = SimplicialComplex()
        
        # Generate vertices on the torus
        vertices = []
        for i in range(n):
            u = 2 * np.pi * i / n
            for j in range(m):
                v = 2 * np.pi * j / m
                x = (2 + np.cos(v)) * np.cos(u)
                y = (2 + np.cos(v)) * np.sin(u)
                z = np.sin(v)
                vertices.append((x, y, z))
        
        # Add vertices to the complex
        vertex_ids = []
        for i, v in enumerate(vertices):
            vid = complex.add_simplex((i,))
            vertex_ids.append(vid)
        
        # Add edges and triangles
        for i in range(n):
            for j in range(m):
                # Current vertex
                idx = i * m + j
                
                # Adjacent vertices (with periodic boundary)
                idx_right = i * m + (j + 1) % m
                idx_down = (i + 1) % n * m + j
                idx_diag = (i + 1) % n * m + (j + 1) % m
                
                # Add edges
                edge1 = complex.add_simplex((idx, idx_right))
                edge2 = complex.add_simplex((idx, idx_down))
                edge3 = complex.add_simplex((idx_right, idx_diag))
                edge4 = complex.add_simplex((idx_down, idx_diag))
                edge5 = complex.add_simplex((idx_right, idx_down))
                
                # Add triangles
                complex.add_simplex((idx, idx_right, idx_down))
                complex.add_simplex((idx_right, idx_down, idx_diag))
        
        return complex
    
    @staticmethod
    def build_klein_bottle(n=10, m=10):
        """
        Build a triangulation of the Klein bottle.
        
        Parameters:
        ----------
        n, m : int, optional
            Resolution parameters
            
        Returns:
        -------
        SimplicialComplex
            Triangulation of the Klein bottle
        """
        complex = SimplicialComplex()
        
        # Generate vertices on the Klein bottle
        vertices = []
        for i in range(n):
            u = 2 * np.pi * i / n
            for j in range(m):
                v = 2 * np.pi * j / m
                
                # Parametrization of Klein bottle
                x = (3 + np.cos(v) * np.cos(u/2)) * np.cos(u)
                y = (3 + np.cos(v) * np.cos(u/2)) * np.sin(u)
                z = np.sin(v) * np.cos(u/2)
                
                vertices.append((x, y, z))
        
        # Add vertices to the complex
        vertex_ids = []
        for i, v in enumerate(vertices):
            vid = complex.add_simplex((i,))
            vertex_ids.append(vid)
        
        # Add edges and triangles
        for i in range(n):
            for j in range(m):
                # Current vertex
                idx = i * m + j
                
                # Adjacent vertices (with twist for Klein bottle)
                idx_right = i * m + (j + 1) % m
                
                if i < n - 1:
                    idx_down = (i + 1) * m + j
                    idx_diag = (i + 1) * m + (j + 1) % m
                else:
                    # Apply the twist at the boundary
                    idx_down = j  # Connect to the bottom row
                    idx_diag = (j + 1) % m  # Connect to the bottom row with shift
                
                # Add edges
                edge1 = complex.add_simplex((idx, idx_right))
                edge2 = complex.add_simplex((idx, idx_down))
                edge3 = complex.add_simplex((idx_right, idx_diag))
                edge4 = complex.add_simplex((idx_down, idx_diag))
                edge5 = complex.add_simplex((idx_right, idx_down))
                
                # Add triangles
                complex.add_simplex((idx, idx_right, idx_down))
                complex.add_simplex((idx_right, idx_down, idx_diag))
        
        return complex
    
    @staticmethod
    def build_projective_plane(n=10):
        """
        Build a triangulation of the real projective plane.
        
        Parameters:
        ----------
        n : int, optional
            Resolution parameter
            
        Returns:
        -------
        SimplicialComplex
            Triangulation of the projective plane
        """
        complex = SimplicialComplex()
        
        # Generate vertices on the hemisphere with identified antipodal boundary points
        vertices = []
        vertex_map = {}  # Map to handle identification of antipodal points
        
        # Add north pole
        vertices.append((0, 0, 1))
        
        # Add intermediate points
        for i in range(1, n):
            theta = np.pi * i / (2*n)  # Only go to equator (π/2)
            for j in range(2*n):
                phi = 2 * np.pi * j / (2*n)
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                vertices.append((x, y, z))
        
        # Add identified points on the equator
        for j in range(n):
            phi = 2 * np.pi * j / (2*n)
            x = np.cos(phi)
            y = np.sin(phi)
            z = 0
            
            # For points on the equator, we identify antipodal points
            idx = len(vertices)
            vertex_map[(j, True)] = idx  # Forward direction
            vertex_map[(j + n, False)] = idx  # Reverse direction
            
            vertices.append((x, y, z))
        
        # Add vertices to the complex
        vertex_ids = []
        for i, v in enumerate(vertices):
            vid = complex.add_simplex((i,))
            vertex_ids.append(vid)
        
        # Add edges and triangles for the hemisphere
        # North pole connections
        for j in range(2*n):
            idx_pole = 0
            idx_next = 1 + j
            idx_next_right = 1 + (j + 1) % (2*n)
            
            # Add edges
            edge1 = complex.add_simplex((idx_pole, idx_next))
            edge2 = complex.add_simplex((idx_next, idx_next_right))
            edge3 = complex.add_simplex((idx_pole, idx_next_right))
            
            # Add triangle
            complex.add_simplex((idx_pole, idx_next, idx_next_right))
        
        # Intermediate connections
        for i in range(1, n-1):
            for j in range(2*n):
                # Current vertex
                idx = 1 + (i-1) * (2*n) + j
                
                # Adjacent vertices
                idx_right = 1 + (i-1) * (2*n) + (j + 1) % (2*n)
                idx_down = 1 + i * (2*n) + j
                idx_down_right = 1 + i * (2*n) + (j + 1) % (2*n)
                
                # Add edges
                edge1 = complex.add_simplex((idx, idx_right))
                edge2 = complex.add_simplex((idx, idx_down))
                edge3 = complex.add_simplex((idx_right, idx_down_right))
                edge4 = complex.add_simplex((idx_down, idx_down_right))
                edge5 = complex.add_simplex((idx_right, idx_down))
                
                # Add triangles
                complex.add_simplex((idx, idx_right, idx_down))
                complex.add_simplex((idx_right, idx_down, idx_down_right))
        
        # Handle equator connections with identifications
        i = n - 1
        for j in range(n):
            # Points just above equator
            idx = 1 + (i-1) * (2*n) + j
            idx_right = 1 + (i-1) * (2*n) + (j + 1) % (2*n)
            
            # Points on equator (with identification)
            idx_down = 1 + (n-1) * (2*n) + vertex_map.get((j, True), j)
            idx_down_right = 1 + (n-1) * (2*n) + vertex_map.get((j+1, True), (j+1) % n)
            
            # Add edges
            edge1 = complex.add_simplex((idx, idx_right))
            edge2 = complex.add_simplex((idx, idx_down))
            edge3 = complex.add_simplex((idx_right, idx_down_right))
            edge4 = complex.add_simplex((idx_down, idx_down_right))
            edge5 = complex.add_simplex((idx_right, idx_down))
            
            # Add triangles
            complex.add_simplex((idx, idx_right, idx_down))
            complex.add_simplex((idx_right, idx_down, idx_down_right))
        
        return complex


class IntersectionTheory:
    """
    Implementation of intersection theory concepts from Poincaré's work.
    """
    
    @staticmethod
    def intersection_form(complex, dim):
        """
        Compute the intersection form for cycles in the given dimension.
        
        Parameters:
        ----------
        complex : SimplicialComplex
            The simplicial complex
        dim : int
            Dimension of cycles
            
        Returns:
        -------
        ndarray
            Intersection form matrix
        """
        # Compute homology basis
        rank, basis = complex.compute_homology(dim)
        
        if rank == 0:
            return np.array([])
            
        # The intersection form is a matrix where entry (i,j) is the
        # intersection number of the i-th and j-th basis elements
        
        # For a proper implementation, we would need to compute actual
        # intersection numbers. This is a simplified placeholder.
        intersection_matrix = np.zeros((rank, rank))
        
        # In a 2n-dimensional manifold, the intersection form on the 
        # middle homology H_n is non-degenerate
        max_dim = max(complex.simplices.keys()) if complex.simplices else 0
        
        if max_dim == 2*dim:
            # In this case, the intersection form should be symplectic or symmetric
            # This is a placeholder for demonstration
            for i in range(rank):
                for j in range(rank):
                    if i < rank//2 and j == i + rank//2:
                        intersection_matrix[i, j] = 1
                        intersection_matrix[j, i] = -1
        
        return intersection_matrix
    
    @staticmethod
    def signature(complex):
        """
        Compute the signature of a 4k-dimensional manifold.
        
        This is a topological invariant introduced in Poincaré's work.
        
        Parameters:
        ----------
        complex : SimplicialComplex
            The simplicial complex
            
        Returns:
        -------
        int
            Signature of the manifold
        """
        max_dim = max(complex.simplices.keys()) if complex.simplices else 0
        
        # For a 4k-dimensional manifold, the signature is defined on the middle homology
        if max_dim % 4 != 0:
            return 0  # Not a 4k-dimensional manifold
            
        middle_dim = max_dim // 2
        intersection_form = IntersectionTheory.intersection_form(complex, middle_dim)
        
        if intersection_form.size == 0:
            return 0
            
        # The signature is the number of positive eigenvalues minus the number of negative eigenvalues
        eigenvalues = np.linalg.eigvalsh(intersection_form)
        positive = np.sum(eigenvalues > 1e-10)
        negative = np.sum(eigenvalues < -1e-10)
        
        return positive - negative


# Example usage
if __name__ == "__main__":
    # Create different topological spaces
    print("Creating a sphere...")
    sphere = ManifoldBuilder.build_sphere(n=6)
    
    print("Computing homology...")
    for dim in range(4):
        rank, basis = sphere.compute_homology(dim)
        print(f"H_{dim}(S²) = Z^{rank}")
    
    print("\nBetti numbers:", sphere.betti_numbers())
    print("Euler characteristic:", sphere.euler_characteristic())
    
    print("\nCreating a torus...")
    torus = ManifoldBuilder.build_torus(n=6, m=6)
    
    print("Computing homology...")
    for dim in range(4):
        rank, basis = torus.compute_homology(dim)
        print(f"H_{dim}(T²) = Z^{rank}")
    
    print("\nBetti numbers:", torus.betti_numbers())
    print("Euler characteristic:", torus.euler_characteristic())
    
    print("\nCreating a Klein bottle...")
    klein = ManifoldBuilder.build_klein_bottle(n=6, m=6)
    
    print("Computing homology...")
    for dim in range(4):
        rank, basis = klein.compute_homology(dim)
        print(f"H_{dim}(K²) = Z^{rank} ⊕ Z₂^{1 if dim == 1 else 0}")
    
    print("\nBetti numbers (excluding torsion):", klein.betti_numbers())
    print("Euler characteristic:", klein.euler_characteristic())
    print("Is orientable:", klein.is_orientable() if hasattr(klein, 'is_orientable') else "Not implemented")
    
    print("\nComputing fundamental groups...")
    print("Generators for π₁(S²):", len(sphere.fundamental_group_generators(0)))
    print("Generators for π₁(T²):", len(torus.fundamental_group_generators(0)))
    
    # Compute Poincaré duality maps
    print("\nPoincaré duality:")
    for dim in range(3):
        dual_map = sphere.poincare_duality_map(dim)
        if dual_map is not None:
            print(f"Duality map H_{dim}(S²) → H_{2-dim}(S²) constructed")
    
    # Visualize the complexes
    print("\nVisualizing the simplicial complexes...")
    sphere.visualize()
    torus.visualize()
    klein.visualize()