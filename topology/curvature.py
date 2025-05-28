"""
Advanced Curvature Flow Analysis Techniques
==========================================

This module extends the Ricci flow implementation with additional curvature
flows and analytical techniques for studying their behaviors.

Key extensions:
1. Mean curvature flow
2. Calabi flow
3. Cross curvature flow
4. Normalized flows for long-time existence
5. Analysis tools for singularity formation
6. Convergence analysis
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
from ricci import DiscretizedManifold, RicciFlow


class MeanCurvatureFlow:
    """Implementation of the mean curvature flow on a discretized manifold."""
    
    def __init__(self, manifold, dt=0.01):
        """
        Initialize the mean curvature flow.
        
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
        self.curvature_history = [manifold.compute_mean_curvature()]
    
    def step(self):
        """Perform one step of the mean curvature flow."""
        # Compute the mean curvature vector
        mean_curvature_vector = spsolve(
            sparse.diags(self.manifold.vertex_areas), 
            self.manifold.laplacian @ self.manifold.vertices
        )
        
        # The mean curvature flow is ∂X/∂t = H
        # where H is the mean curvature vector
        self.manifold.vertices -= self.dt * mean_curvature_vector
        
        # Update manifold properties after vertices have moved
        self.manifold._compute_edge_vectors()
        self.manifold._compute_face_metrics()
        self.manifold._compute_vertex_areas()
        self.manifold._compute_laplacian()
        
        # Update time and store history
        self.time += self.dt
        self.history.append(self.manifold.vertices.copy())
        self.curvature_history.append(self.manifold.compute_mean_curvature())
    
    def evolve(self, n_steps):
        """Evolve the manifold for n_steps time steps."""
        for _ in range(n_steps):
            self.step()
    
    def analyze_flow(self):
        """Analyze the mean curvature flow evolution."""
        # Compute various metrics over time
        n_steps = len(self.history)
        times = np.linspace(0, self.time, n_steps)
        
        # Compute average mean curvature over time
        avg_mean_curv = [np.mean(curv) for curv in self.curvature_history]
        
        # Compute the standard deviation of mean curvature over time
        std_mean_curv = [np.std(curv) for curv in self.curvature_history]
        
        # Compute the minimum and maximum mean curvature over time
        min_mean_curv = [np.min(curv) for curv in self.curvature_history]
        max_mean_curv = [np.max(curv) for curv in self.curvature_history]
        
        # Return the analysis results
        return {
            'times': times,
            'avg_mean_curv': avg_mean_curv,
            'std_mean_curv': std_mean_curv,
            'min_mean_curv': min_mean_curv,
            'max_mean_curv': max_mean_curv
        }
    
    def visualize_flow(self, step_indices=None):
        """
        Visualize the mean curvature flow evolution.
        
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
            
            # Get mean curvature for coloring
            mean_curvature = self.curvature_history[step_idx]
            
            # Plot the surface
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            ax.plot_trisurf(x, y, z, triangles=faces, cmap=cm.coolwarm,
                           linewidth=0.2, alpha=0.7, 
                           facecolors=cm.coolwarm(mean_curvature / max(1e-10, mean_curvature.max())))
            
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


class NormalizedRicciFlow(RicciFlow):
    """
    Implementation of the normalized Ricci flow on a discretized manifold.
    The normalization ensures long-time existence for certain manifolds.
    """
    
    def __init__(self, manifold, dt=0.01):
        """
        Initialize the normalized Ricci flow.
        
        Parameters:
        -----------
        manifold : DiscretizedManifold
            The manifold to evolve.
        dt : float
            The time step for numerical integration.
        """
        super().__init__(manifold, dt)
    
    def step(self):
        """Perform one step of the normalized Ricci flow."""
        # Compute the Ricci curvature
        ricci_curvature = self.manifold.compute_ricci_curvature()
        
        # Compute the average scalar curvature (needed for normalization)
        # For a 2D manifold, the scalar curvature equals 2 times the Gaussian curvature
        avg_scalar_curvature = 2 * np.mean(ricci_curvature)
        
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
        
        # Update vertex positions according to normalized Ricci flow
        # The normalized Ricci flow is ∂g/∂t = -2(Ric(g) - r*g)
        # where r is the average scalar curvature
        # For our discrete manifold, we approximate this by moving vertices
        self.manifold.vertices -= self.dt * (ricci_curvature[:, np.newaxis] - avg_scalar_curvature * normals)
        
        # Update manifold properties after vertices have moved
        self.manifold._compute_edge_vectors()
        self.manifold._compute_face_metrics()
        self.manifold._compute_vertex_areas()
        self.manifold._compute_laplacian()
        
        # Update time and store history
        self.time += self.dt
        self.history.append(self.manifold.vertices.copy())
        self.curvature_history.append(ricci_curvature)


class CalabiFlow:
    """
    Implementation of the Calabi flow on a discretized manifold.
    The Calabi flow evolves the metric in the direction of the gradient of
    the L² norm of the scalar curvature.
    """
    
    def __init__(self, manifold, dt=0.001):
        """
        Initialize the Calabi flow.
        
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
        self.curvature_history = [manifold.compute_gaussian_curvature()]
    
    def step(self):
        """Perform one step of the Calabi flow."""
        # Compute the Gaussian curvature
        gaussian_curvature = self.manifold.compute_gaussian_curvature()
        
        # Compute the Laplacian of the Gaussian curvature
        laplacian_K = self.manifold.laplacian @ gaussian_curvature
        
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
        
        # Update vertex positions according to Calabi flow
        # The Calabi flow is ∂g/∂t = -∆K * g
        # where ∆K is the Laplacian of the Gaussian curvature
        update = np.zeros_like(self.manifold.vertices)
        update[mask] = -laplacian_K[mask, np.newaxis] * normals[mask]
        self.manifold.vertices += self.dt * update
        
        # Update manifold properties after vertices have moved
        self.manifold._compute_edge_vectors()
        self.manifold._compute_face_metrics()
        self.manifold._compute_vertex_areas()
        self.manifold._compute_laplacian()
        
        # Update time and store history
        self.time += self.dt
        self.history.append(self.manifold.vertices.copy())
        self.curvature_history.append(self.manifold.compute_gaussian_curvature())
    
    def evolve(self, n_steps):
        """Evolve the manifold for n_steps time steps."""
        for _ in range(n_steps):
            self.step()
    
    def analyze_flow(self):
        """Analyze the Calabi flow evolution."""
        # Compute various metrics over time
        n_steps = len(self.history)
        times = np.linspace(0, self.time, n_steps)
        
        # Compute L² norm of the Gaussian curvature over time
        l2_K = [np.sqrt(np.sum(curv**2 * self.manifold.vertex_areas)) for curv in self.curvature_history]
        
        # Compute average Gaussian curvature over time
        avg_K = [np.mean(curv) for curv in self.curvature_history]
        
        # Compute the standard deviation of Gaussian curvature over time
        std_K = [np.std(curv) for curv in self.curvature_history]
        
        # Return the analysis results
        return {
            'times': times,
            'l2_K': l2_K,
            'avg_K': avg_K,
            'std_K': std_K
        }


class CrossCurvatureFlow:
    """
    Implementation of the cross curvature flow on a discretized 3-manifold.
    
    Note: This is a simplified version for 3D hypersurfaces, as the true
    cross curvature flow is defined for 3D manifolds. We approximate it
    by using the cross product of principal curvature directions.
    """
    
    def __init__(self, manifold, dt=0.001):
        """
        Initialize the cross curvature flow.
        
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
    
    def compute_principal_curvatures(self):
        """Compute principal curvatures and directions at each vertex."""
        # Compute shape operator at each vertex
        n_vertices = len(self.manifold.vertices)
        principal_curvatures = np.zeros((n_vertices, 2))
        principal_directions = np.zeros((n_vertices, 2, 3))
        
        # Get mean curvature vectors
        mean_curvature_vector = spsolve(
            sparse.diags(self.manifold.vertex_areas), 
            self.manifold.laplacian @ self.manifold.vertices
        )
        
        # Compute Gaussian curvature
        gaussian_curvature = self.manifold.compute_gaussian_curvature()
        
        # Compute mean curvature
        mean_curvature = np.linalg.norm(mean_curvature_vector, axis=1)
        
        # For each vertex, compute principal curvatures using the quadratic formula
        # κ₁, κ₂ = H ± sqrt(H² - K)
        for i in range(n_vertices):
            H = mean_curvature[i]
            K = gaussian_curvature[i]
            
            # Compute principal curvatures
            discriminant = H**2 - K
            if discriminant >= 0:
                sqrt_discriminant = np.sqrt(discriminant)
                principal_curvatures[i, 0] = H + sqrt_discriminant
                principal_curvatures[i, 1] = H - sqrt_discriminant
            else:
                # In theory, this shouldn't happen for a real surface
                # But due to numerical issues, it might
                principal_curvatures[i, 0] = H
                principal_curvatures[i, 1] = H
            
            # Compute principal directions
            # For simplicity, we just use an arbitrary tangent frame
            # In a real implementation, we would compute this properly
            normal = mean_curvature_vector[i]
            if np.linalg.norm(normal) > 1e-10:
                normal = normal / np.linalg.norm(normal)
                
                # Find an arbitrary vector not parallel to the normal
                v = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(v, normal)) > 0.9:
                    v = np.array([0.0, 1.0, 0.0])
                
                # Compute tangent vectors using cross products
                tangent1 = np.cross(normal, v)
                tangent1 = tangent1 / np.linalg.norm(tangent1)
                tangent2 = np.cross(normal, tangent1)
                tangent2 = tangent2 / np.linalg.norm(tangent2)
                
                principal_directions[i, 0] = tangent1
                principal_directions[i, 1] = tangent2
        
        return principal_curvatures, principal_directions
    
    def step(self):
        """Perform one step of the cross curvature flow."""
        # Compute principal curvatures and directions
        principal_curvatures, principal_directions = self.compute_principal_curvatures()
        
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
        
        # Compute cross curvature as the product of principal curvatures
        cross_curvature = principal_curvatures[:, 0] * principal_curvatures[:, 1]
        
        # Update vertex positions according to cross curvature flow
        # For our discrete approximation, we move vertices in the normal direction
        # proportional to the cross curvature
        self.manifold.vertices += self.dt * cross_curvature[:, np.newaxis] * normals
        
        # Update manifold properties after vertices have moved
        self.manifold._compute_edge_vectors()
        self.manifold._compute_face_metrics()
        self.manifold._compute_vertex_areas()
        self.manifold._compute_laplacian()
        
        # Update time and store history
        self.time += self.dt
        self.history.append(self.manifold.vertices.copy())
    
    def evolve(self, n_steps):
        """Evolve the manifold for n_steps time steps."""
        for _ in range(n_steps):
            self.step()


class CurvatureFlowAnalyzer:
    """Advanced analysis tools for curvature flows."""
    
    @staticmethod
    def detect_singularities(flow, threshold=1.0):
        """
        Detect potential singularity formation in a curvature flow.
        
        Parameters:
        -----------
        flow : RicciFlow, MeanCurvatureFlow, or similar
            The flow to analyze.
        threshold : float
            Threshold for singularity detection.
            
        Returns:
        --------
        singularities : list of tuples
            Each tuple contains (time_index, vertex_indices) where
            vertex_indices are the vertices with high curvature.
        """
        singularities = []
        
        # For each time step
        for t, curvature in enumerate(flow.curvature_history):
            # Identify vertices with very high curvature
            high_curv_vertices = np.where(np.abs(curvature) > threshold)[0]
            
            if len(high_curv_vertices) > 0:
                singularities.append((t, high_curv_vertices))
        
        return singularities
    
    @staticmethod
    def analyze_convergence(flow, window_size=10):
        """
        Analyze the convergence of a curvature flow.
        
        Parameters:
        -----------
        flow : RicciFlow, MeanCurvatureFlow, or similar
            The flow to analyze.
        window_size : int
            Size of the window for convergence analysis.
            
        Returns:
        --------
        convergence_rates : dict
            Dictionary with convergence metrics.
        """
        # Compute displacement of vertices over time
        displacements = []
        for i in range(1, len(flow.history)):
            disp = np.linalg.norm(flow.history[i] - flow.history[i-1], axis=1).mean()
            displacements.append(disp)
        
        # Compute rolling average of displacement
        rolling_avg = []
        for i in range(len(displacements) - window_size + 1):
            avg = np.mean(displacements[i:i+window_size])
            rolling_avg.append(avg)
        
        # Compute rate of change of displacement
        if len(rolling_avg) > 1:
            rate_of_change = [(rolling_avg[i] - rolling_avg[i-1]) / flow.dt 
                              for i in range(1, len(rolling_avg))]
        else:
            rate_of_change = []
        
        return {
            'displacements': displacements,
            'rolling_avg': rolling_avg,
            'rate_of_change': rate_of_change
        }
    
    @staticmethod
    def analyze_spectral_properties(flow, n_eigenvalues=10):
        """
        Analyze the spectral properties of the Laplacian during the flow.
        
        Parameters:
        -----------
        flow : RicciFlow, MeanCurvatureFlow, or similar
            The flow to analyze.
        n_eigenvalues : int
            Number of smallest eigenvalues to compute.
            
        Returns:
        --------
        eigenvalues : list of arrays
            Eigenvalues of the Laplacian at each time step.
        """
        eigenvalues = []
        
        # Take a subsample of time steps for efficiency
        n_steps = len(flow.history)
        sample_indices = np.linspace(0, n_steps-1, min(20, n_steps)).astype(int)
        
        for idx in sample_indices:
            # Reconstruct the manifold at this time step
            vertices = flow.history[idx]
            manifold = DiscretizedManifold(vertices, flow.manifold.faces)
            
            # Compute eigenvalues of the Laplacian
            try:
                vals, _ = eigsh(manifold.laplacian, k=min(n_eigenvalues, manifold.laplacian.shape[0]-2), 
                               which='SM', tol=1e-4)
                eigenvalues.append(vals)
            except:
                # In case of numerical issues
                eigenvalues.append(np.array([np.nan] * n_eigenvalues))
        
        return {
            'times': sample_indices * flow.dt,
            'eigenvalues': eigenvalues
        }
    
    @staticmethod
    def create_flow_animation(flow, interval=100, n_frames=None):
        """
        Create an animation of the flow evolution.
        
        Parameters:
        -----------
        flow : RicciFlow, MeanCurvatureFlow, or similar
            The flow to animate.
        interval : int
            Interval between frames in milliseconds.
        n_frames : int or None
            Number of frames to use. If None, use all frames.
            
        Returns:
        --------
        anim : matplotlib.animation.FuncAnimation
            The animation object.
        """
        if n_frames is None:
            n_frames = len(flow.history)
        
        # Sample frames evenly
        frame_indices = np.linspace(0, len(flow.history)-1, n_frames).astype(int)
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Function to update the plot for each frame
        def update(frame):
            ax.clear()
            idx = frame_indices[frame]
            vertices = flow.history[idx]
            
            # Try to get curvature data if available
            if hasattr(flow, 'curvature_history') and len(flow.curvature_history) > idx:
                curvature = flow.curvature_history[idx]
                vmin = min(0, curvature.min())
                vmax = max(0, curvature.max())
                
                # Normalize curvature for colormap
                if vmax > vmin:
                    normalized_curvature = (curvature - vmin) / (vmax - vmin)
                else:
                    normalized_curvature = np.zeros_like(curvature)
                
                # Plot surface with curvature coloring
                x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
                ax.plot_trisurf(x, y, z, triangles=flow.manifold.faces, 
                               cmap=cm.viridis, linewidth=0.2, alpha=0.7,
                               facecolors=cm.viridis(normalized_curvature))
            else:
                # Plot surface without coloring
                x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
                ax.plot_trisurf(x, y, z, triangles=flow.manifold.faces, 
                               cmap=cm.viridis, linewidth=0.2, alpha=0.7)
            
            # Set title and labels
            ax.set_title(f'Time: {idx * flow.dt:.3f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set consistent bounds
            max_range = 1.5
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
            
            return ax
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval)
        
        return anim, fig


def demo_comparison_of_flows():
    """
    Compare different curvature flows on the same initial manifold.
    """
    from ricci import create_sphere
    
    # Create an ellipsoid (stretched sphere)
    sphere = create_sphere(radius=1.0, resolution=15)
    
    # Stretch along x-axis
    sphere.vertices[:, 0] *= 1.5
    
    # Make copies for different flows
    sphere_ricci = DiscretizedManifold(sphere.vertices.copy(), sphere.faces.copy())
    sphere_mean = DiscretizedManifold(sphere.vertices.copy(), sphere.faces.copy())
    sphere_normalized = DiscretizedManifold(sphere.vertices.copy(), sphere.faces.copy())
    
    # Initialize flows
    ricci_flow = RicciFlow(sphere_ricci, dt=0.01)
    mean_flow = MeanCurvatureFlow(sphere_mean, dt=0.01)
    norm_flow = NormalizedRicciFlow(sphere_normalized, dt=0.01)
    
    # Evolve flows
    n_steps = 50
    ricci_flow.evolve(n_steps)
    mean_flow.evolve(n_steps)
    norm_flow.evolve(n_steps)
    
    # Display final shapes
    fig = plt.figure(figsize=(15, 5))
    
    # Original shape
    ax1 = fig.add_subplot(141, projection='3d')
    x, y, z = sphere.vertices[:, 0], sphere.vertices[:, 1], sphere.vertices[:, 2]
    ax1.plot_trisurf(x, y, z, triangles=sphere.faces, cmap=cm.viridis, linewidth=0.2, alpha=0.7)
    ax1.set_title('Original Ellipsoid')
    
    # Ricci flow result
    ax2 = fig.add_subplot(142, projection='3d')
    x, y, z = ricci_flow.history[-1][:, 0], ricci_flow.history[-1][:, 1], ricci_flow.history[-1][:, 2]
    ax2.plot_trisurf(x, y, z, triangles=sphere.faces, cmap=cm.viridis, linewidth=0.2, alpha=0.7)
    ax2.set_title('Ricci Flow')
    
    # Mean curvature flow result
    ax3 = fig.add_subplot(143, projection='3d')
    x, y, z = mean_flow.history[-1][:, 0], mean_flow.history[-1][:, 1], mean_flow.history[-1][:, 2]
    ax3.plot_trisurf(x, y, z, triangles=sphere.faces, cmap=cm.viridis, linewidth=0.2, alpha=0.7)
    ax3.set_title('Mean Curvature Flow')
    
    # Normalized Ricci flow result
    ax4 = fig.add_subplot(144, projection='3d')
    x, y, z = norm_flow.history[-1][:, 0], norm_flow.history[-1][:, 1], norm_flow.history[-1][:, 2]
    ax4.plot_trisurf(x, y, z, triangles=sphere.faces, cmap=cm.viridis, linewidth=0.2, alpha=0.7)
    ax4.set_title('Normalized Ricci Flow')
    
    # Set consistent view limits
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.view_init(30, 30)
    
    plt.tight_layout()
    
    # Analyze convergence
    analyzer = CurvatureFlowAnalyzer()
    ricci_convergence = analyzer.analyze_convergence(ricci_flow)
    mean_convergence = analyzer.analyze_convergence(mean_flow)
    norm_convergence = analyzer.analyze_convergence(norm_flow)
    
    # Plot convergence rates
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ricci_convergence['displacements'], 'r-', label='Ricci Flow')
    ax.plot(mean_convergence['displacements'], 'g-', label='Mean Curvature Flow')
    ax.plot(norm_convergence['displacements'], 'b-', label='Normalized Ricci Flow')
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Displacement')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True)
    
    return fig, fig2


if __name__ == "__main__":
    # Demonstrate comparison of flows
    print("Running comparison of curvature flows...")
    fig1, fig2 = demo_comparison_of_flows()
    
    # Create animator for visualization
    from ricci import create_torus
    print("\nCreating torus animation with Ricci flow...")
    torus = create_torus(R=1.0, r=0.3, resolution=20)
    flow = RicciFlow(torus, dt=0.005)
    flow.evolve(100)
    
    analyzer = CurvatureFlowAnalyzer()
    anim, _ = analyzer.create_flow_animation(flow, n_frames=20)
    
    print("\nAnalysis complete!")
    plt.show()