"""
Ricci Flow Implementation and Analysis Tools
===========================================

This module implements numerical techniques for analyzing curvature flows,
particularly the Ricci flow on discretized manifolds.

The Ricci flow equation: ∂g/∂t = -2Ric(g)
where g is the metric tensor and Ric(g) is the Ricci curvature.

For simplicity, we'll implement this on surfaces (2D manifolds) embedded in 3D space.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class DiscretizedManifold:
    """A triangulated representation of a 2D manifold in 3D space."""
    
    def __init__(self, vertices, faces):
        """
        Initialize a discrete manifold.
        
        Parameters:
        -----------
        vertices : ndarray, shape (n_vertices, 3)
            The 3D coordinates of vertices.
        faces : ndarray, shape (n_faces, 3)
            Indices of vertices forming triangular faces.
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        
        # Compute initial metric tensors, areas, etc.
        self._compute_edge_vectors()
        self._compute_face_metrics()
        self._compute_vertex_areas()
        self._compute_laplacian()
        
    def _compute_edge_vectors(self):
        """Compute edge vectors for each face."""
        self.edge_vectors = []
        for face in self.faces:
            # Get vertices of the face
            v0, v1, v2 = self.vertices[face]
            
            # Compute edge vectors
            e01 = v1 - v0
            e12 = v2 - v1
            e20 = v0 - v2
            
            self.edge_vectors.append([e01, e12, e20])
    
    def _compute_face_metrics(self):
        """Compute metric tensors and areas for each face."""
        self.face_metrics = []
        self.face_areas = []
        
        for edges in self.edge_vectors:
            e01, e12, e20 = edges
            
            # Compute face normal
            normal = np.cross(e01, -e20)
            area = 0.5 * np.linalg.norm(normal)
            self.face_areas.append(area)
            
            # Compute the metric tensor (first fundamental form)
            metric = np.zeros((2, 2))
            metric[0, 0] = np.dot(e01, e01)  # E
            metric[0, 1] = np.dot(e01, -e20)  # F
            metric[1, 0] = metric[0, 1]       # F
            metric[1, 1] = np.dot(-e20, -e20) # G
            
            self.face_metrics.append(metric)
    
    def _compute_vertex_areas(self):
        """Compute the area associated with each vertex (Voronoi area)."""
        self.vertex_areas = np.zeros(len(self.vertices))
        
        for i, face in enumerate(self.faces):
            # Distribute 1/3 of face area to each vertex
            area = self.face_areas[i] / 3.0
            for vertex_idx in face:
                self.vertex_areas[vertex_idx] += area
    
    def _compute_laplacian(self):
        """Compute the cotangent Laplacian operator."""
        n_vertices = len(self.vertices)
        rows, cols, data = [], [], []
        
        # Iterate over faces
        for i, face in enumerate(self.faces):
            # Get vertices of the face
            i0, i1, i2 = face
            
            # Get edge vectors
            e01, e12, e20 = self.edge_vectors[i]
            
            # Compute cotangents of angles
            # cot(angle_0) where angle_0 is at vertex 0
            dot_e01_e20 = np.dot(e01, -e20)
            cross_e01_e20 = np.linalg.norm(np.cross(e01, -e20))
            cot_0 = dot_e01_e20 / cross_e01_e20 if cross_e01_e20 != 0 else 0
            
            # cot(angle_1) where angle_1 is at vertex 1
            dot_e12_e01 = np.dot(e12, -e01)
            cross_e12_e01 = np.linalg.norm(np.cross(e12, -e01))
            cot_1 = dot_e12_e01 / cross_e12_e01 if cross_e12_e01 != 0 else 0
            
            # cot(angle_2) where angle_2 is at vertex 2
            dot_e20_e12 = np.dot(e20, -e12)
            cross_e20_e12 = np.linalg.norm(np.cross(e20, -e12))
            cot_2 = dot_e20_e12 / cross_e20_e12 if cross_e20_e12 != 0 else 0
            
            # Add contributions to the Laplacian matrix
            # For edge (i0, i1)
            rows.extend([i0, i1, i0, i1])
            cols.extend([i1, i0, i0, i1])
            data.extend([cot_2, cot_2, -cot_2, -cot_2])
            
            # For edge (i1, i2)
            rows.extend([i1, i2, i1, i2])
            cols.extend([i2, i1, i1, i2])
            data.extend([cot_0, cot_0, -cot_0, -cot_0])
            
            # For edge (i2, i0)
            rows.extend([i2, i0, i2, i0])
            cols.extend([i0, i2, i2, i0])
            data.extend([cot_1, cot_1, -cot_1, -cot_1])
        
        # Construct the sparse matrix
        self.laplacian = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
        self.laplacian = self.laplacian.tocsr()
    
    def compute_mean_curvature(self):
        """Compute the mean curvature at each vertex."""
        # The mean curvature vector is approximated as H = ΔX
        # where Δ is the Laplace-Beltrami operator and X are vertex coordinates
        mean_curvature_vector = spsolve(sparse.diags(self.vertex_areas), self.laplacian @ self.vertices)
        
        # The mean curvature is the magnitude of the mean curvature vector
        mean_curvature = np.linalg.norm(mean_curvature_vector, axis=1)
        
        return mean_curvature
    
    def compute_gaussian_curvature(self):
        """Compute the Gaussian curvature at each vertex using the angle deficit."""
        n_vertices = len(self.vertices)
        gaussian_curvature = np.zeros(n_vertices)
        
        # For each vertex, compute the sum of angles of incident triangles
        for i, face in enumerate(self.faces):
            # Get vertices of the face
            v0, v1, v2 = self.vertices[face]
            
            # Compute edge vectors
            e01 = v1 - v0
            e12 = v2 - v1
            e20 = v0 - v2
            
            # Compute angles at each vertex
            e10 = -e01
            e21 = -e12
            e02 = -e20
            
            angle0 = np.arccos(np.dot(e10, e20) / (np.linalg.norm(e10) * np.linalg.norm(e20)))
            angle1 = np.arccos(np.dot(e21, e01) / (np.linalg.norm(e21) * np.linalg.norm(e01)))
            angle2 = np.arccos(np.dot(e02, e12) / (np.linalg.norm(e02) * np.linalg.norm(e12)))
            
            # Add angles to vertices
            gaussian_curvature[face[0]] += angle0
            gaussian_curvature[face[1]] += angle1
            gaussian_curvature[face[2]] += angle2
        
        # Compute the angle deficit (2π - sum of angles)
        gaussian_curvature = 2 * np.pi - gaussian_curvature
        
        # Divide by the vertex area to get the Gaussian curvature
        gaussian_curvature /= self.vertex_areas
        
        return gaussian_curvature
    
    def compute_ricci_curvature(self):
        """
        Compute the Ricci curvature at each vertex.
        
        For a surface, the Ricci curvature equals the Gaussian curvature times the metric.
        """
        # For 2D manifolds, Ricci curvature is related to Gaussian and mean curvature
        gaussian_curvature = self.compute_gaussian_curvature()
        mean_curvature = self.compute_mean_curvature()
        
        # On a surface, the Ricci curvature tensor has only one independent component
        # which is proportional to the Gaussian curvature
        ricci_curvature = gaussian_curvature
        
        return ricci_curvature


class RicciFlow:
    """Implementation of the Ricci flow on a discretized manifold."""
    
    def __init__(self, manifold, dt=0.01):
        """
        Initialize the Ricci flow.
        
        Parameters:
        -----------
        manifold : DiscretizedManifold
            The manifold to evolve.
        dt : float
            The time step for numerical integration.
        """
        self.manifold = manifold
        self.dt = dt
        self.time = 0.0
        self.history = [manifold.vertices.copy()]
        self.curvature_history = [manifold.compute_ricci_curvature()]
    
    def step(self):
        """Perform one step of the Ricci flow."""
        # Compute the Ricci curvature
        ricci_curvature = self.manifold.compute_ricci_curvature()
        
        # Update the metric via vertex positions
        # The Ricci flow is ∂g/∂t = -2Ric(g)
        # For embedded surfaces, we can approximate this by moving vertices in the normal direction
        
        # Get mean curvature vectors (approximates surface normals)
        mean_curvature_vector = spsolve(
            sparse.diags(self.manifold.vertex_areas), 
            self.manifold.laplacian @ self.manifold.vertices
        )
        
        # Normalize to get the normal directions
        norms = np.linalg.norm(mean_curvature_vector, axis=1)
        normals = np.zeros_like(mean_curvature_vector)
        mask = norms > 0
        normals[mask] = mean_curvature_vector[mask] / norms[mask, np.newaxis]
        
        # Update vertex positions according to Ricci flow
        # Move against the Ricci curvature in the normal direction
        self.manifold.vertices -= self.dt * ricci_curvature[:, np.newaxis] * normals
        
        # Update manifold properties after vertices have moved
        self.manifold._compute_edge_vectors()
        self.manifold._compute_face_metrics()
        self.manifold._compute_vertex_areas()
        self.manifold._compute_laplacian()
        
        # Update time and store history
        self.time += self.dt
        self.history.append(self.manifold.vertices.copy())
        self.curvature_history.append(ricci_curvature)
    
    def evolve(self, n_steps):
        """Evolve the manifold for n_steps time steps."""
        for _ in range(n_steps):
            self.step()
    
    def analyze_flow(self):
        """Analyze the Ricci flow evolution."""
        # Compute various metrics over time
        n_steps = len(self.history)
        times = np.linspace(0, self.time, n_steps)
        
        # Compute average Ricci curvature over time
        avg_ricci = [np.mean(curv) for curv in self.curvature_history]
        
        # Compute the standard deviation of Ricci curvature over time
        std_ricci = [np.std(curv) for curv in self.curvature_history]
        
        # Compute the minimum and maximum Ricci curvature over time
        min_ricci = [np.min(curv) for curv in self.curvature_history]
        max_ricci = [np.max(curv) for curv in self.curvature_history]
        
        # Return the analysis results
        return {
            'times': times,
            'avg_ricci': avg_ricci,
            'std_ricci': std_ricci,
            'min_ricci': min_ricci,
            'max_ricci': max_ricci
        }
    
    def visualize_flow(self, step_indices=None):
        """
        Visualize the Ricci flow evolution.
        
        Parameters:
        -----------
        step_indices : list or None
            Indices of steps to visualize. If None, visualize all steps.
        """
        if step_indices is None:
            step_indices = range(0, len(self.history), max(1, len(self.history) // 5))
        
        n_vis = len(step_indices)
        fig = plt.figure(figsize=(15, 5 * n_vis))
        
        for i, step_idx in enumerate(step_indices):
            ax = fig.add_subplot(n_vis, 1, i + 1, projection='3d')
            
            # Get vertices and faces for this step
            vertices = self.history[step_idx]
            faces = self.manifold.faces
            
            # Get Ricci curvature for coloring
            ricci_curvature = self.curvature_history[step_idx]
            
            # Plot the surface
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            ax.plot_trisurf(x, y, z, triangles=faces, cmap=cm.viridis,
                           linewidth=0.2, alpha=0.7, 
                           facecolors=cm.viridis(ricci_curvature / max(abs(ricci_curvature.max()), abs(ricci_curvature.min()))))
            
            ax.set_title(f'Step {step_idx}, Time {step_idx * self.dt:.3f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Make the plot bounds equal for consistent visualization
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
            mid_x = (x.max() + x.min()) * 0.5
            mid_y = (y.max() + y.min()) * 0.5
            mid_z = (z.max() + z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        return fig
    
    def visualize_curvature_evolution(self):
        """Visualize the evolution of Ricci curvature over time."""
        analysis = self.analyze_flow()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot average Ricci curvature with standard deviation band
        ax.plot(analysis['times'], analysis['avg_ricci'], 'b-', label='Avg Ricci Curvature')
        ax.fill_between(analysis['times'],
                       np.array(analysis['avg_ricci']) - np.array(analysis['std_ricci']),
                       np.array(analysis['avg_ricci']) + np.array(analysis['std_ricci']),
                       alpha=0.3, color='b')
        
        # Plot min and max Ricci curvature
        ax.plot(analysis['times'], analysis['min_ricci'], 'g--', label='Min Ricci Curvature')
        ax.plot(analysis['times'], analysis['max_ricci'], 'r--', label='Max Ricci Curvature')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Ricci Curvature')
        ax.set_title('Evolution of Ricci Curvature During Flow')
        ax.legend()
        ax.grid(True)
        
        return fig


# Example usage: Creating and evolving a sphere
def create_sphere(radius=1.0, resolution=20):
    """Create a discretized sphere."""
    # Create a UV sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    # Generate vertices
    vertices = []
    for i in range(resolution):
        for j in range(resolution):
            x = radius * np.sin(v[j]) * np.cos(u[i])
            y = radius * np.sin(v[j]) * np.sin(u[i])
            z = radius * np.cos(v[j])
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])
    
    faces = np.array(faces)
    
    return DiscretizedManifold(vertices, faces)


def create_torus(R=1.0, r=0.3, resolution=20):
    """Create a discretized torus."""
    # Create a parametric torus
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, 2 * np.pi, resolution)
    
    # Generate vertices
    vertices = []
    for i in range(resolution):
        for j in range(resolution):
            x = (R + r * np.cos(v[j])) * np.cos(u[i])
            y = (R + r * np.cos(v[j])) * np.sin(u[i])
            z = r * np.sin(v[j])
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])
    
    # Connect the last column to the first
    for j in range(resolution - 1):
        idx1 = (resolution - 1) * resolution + j
        idx2 = j
        faces.append([idx1, idx1 + 1, idx2])
        faces.append([idx1 + 1, idx2 + 1, idx2])
    
    # Connect the last row to the first
    for i in range(resolution - 1):
        idx1 = i * resolution + (resolution - 1)
        idx2 = (i + 1) * resolution
        faces.append([idx1, idx2, idx1 - (resolution - 1)])
        faces.append([idx2, idx2 + (resolution - 1), idx1 - (resolution - 1)])
    
    # Connect the last vertex to the first
    idx1 = (resolution - 1) * resolution + (resolution - 1)
    idx2 = 0
    faces.append([idx1, idx2, idx1 - (resolution - 1)])
    faces.append([idx2, resolution - 1, idx1 - (resolution - 1)])
    
    faces = np.array(faces)
    
    return DiscretizedManifold(vertices, faces)


def demo_ricci_flow_on_sphere():
    """Demonstrate Ricci flow on a sphere."""
    # Create a sphere
    sphere = create_sphere(radius=1.0, resolution=20)
    
    # Initialize Ricci flow
    flow = RicciFlow(sphere, dt=0.01)
    
    # Evolve for 100 steps
    flow.evolve(100)
    
    # Analyze the flow
    analysis = flow.analyze_flow()
    
    # Visualize the flow
    fig1 = flow.visualize_flow(step_indices=[0, 25, 50, 75, 99])
    fig2 = flow.visualize_curvature_evolution()
    
    plt.show()
    
    return analysis, fig1, fig2


def demo_ricci_flow_on_torus():
    """Demonstrate Ricci flow on a torus."""
    # Create a torus
    torus = create_torus(R=1.0, r=0.3, resolution=20)
    
    # Initialize Ricci flow
    flow = RicciFlow(torus, dt=0.005)
    
    # Evolve for 200 steps
    flow.evolve(200)
    
    # Analyze the flow
    analysis = flow.analyze_flow()
    
    # Visualize the flow
    fig1 = flow.visualize_flow(step_indices=[0, 50, 100, 150, 199])
    fig2 = flow.visualize_curvature_evolution()
    
    plt.show()
    
    return analysis, fig1, fig2


if __name__ == "__main__":
    # Run the sphere demo
    print("Running Ricci flow on sphere...")
    sphere_analysis, _, _ = demo_ricci_flow_on_sphere()
    
    print("\nRunning Ricci flow on torus...")
    torus_analysis, _, _ = demo_ricci_flow_on_torus()
    
    print("\nAnalysis complete!")