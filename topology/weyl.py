import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import itertools
from functools import reduce
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.linalg import expm
import warnings
from sympy import Matrix, Symbol, Function, diff, exp, symbols, simplify, Eq, solve
from sympy.tensor.tensor import tensor_indices, TensorHead, TensorSymmetry
from sympy.tensor.toperators import PartialDerivative
from sympy.printing import pprint
from mpl_toolkits.mplot3d import Axes3D

class TensorCalculus:
    """
    Class for tensor calculations in differential geometry,
    with focus on the Weyl tensor and conformal geometry.
    """
    
    def __init__(self, dimension=4, coordinates=None, metric=None):
        """
        Initialize the tensor calculus system.
        
        Parameters:
        -----------
        dimension : int
            Dimension of the manifold
        coordinates : list of sympy.Symbol, optional
            Coordinate symbols
        metric : sympy.Matrix, optional
            Metric tensor g_μν
        """
        self.dimension = dimension
        
        # Set up coordinate symbols if not provided
        if coordinates is None:
            if dimension <= 4:
                self.coordinates = sp.symbols('t x y z')[:dimension]
            else:
                self.coordinates = sp.symbols('x_0:%d' % dimension)
        else:
            self.coordinates = coordinates
            
        # Set up metric if not provided
        if metric is None:
            # Default to Minkowski metric for 4D
            if dimension == 4:
                self.metric = sp.diag(-1, 1, 1, 1)
            else:
                # Euclidean metric for other dimensions
                self.metric = sp.eye(dimension)
        else:
            if metric.shape != (dimension, dimension):
                raise ValueError(f"Metric must be a {dimension}×{dimension} matrix")
            self.metric = metric
            
        # Compute inverse metric
        self.inverse_metric = self.metric.inv()
        
        # Initialize tensors
        self.christoffel_symbols = None
        self.riemann_tensor = None
        self.ricci_tensor = None
        self.ricci_scalar = None
        self.weyl_tensor = None
        self.einstein_tensor = None
        self.schouten_tensor = None
        self.cotton_tensor = None
    
    def compute_christoffel_symbols(self):
        """
        Compute the Christoffel symbols of the second kind.
        
        Returns:
        --------
        list of lists of lists
            Christoffel symbols Γ^λ_μν
        """
        n = self.dimension
        coords = self.coordinates
        g = self.metric
        g_inv = self.inverse_metric
        
        # Initialize Christoffel symbols
        christoffel = np.zeros((n, n, n), dtype=object)
        
        # Compute partial derivatives of the metric
        partial_g = np.zeros((n, n, n), dtype=object)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    partial_g[k, i, j] = sp.diff(g[i, j], coords[k])
        
        # Compute Christoffel symbols
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    # Sum over sigma
                    for sigma in range(n):
                        term = g_inv[lam, sigma] * (partial_g[mu, sigma, nu] + 
                                                  partial_g[nu, sigma, mu] - 
                                                  partial_g[sigma, mu, nu]) / 2
                        christoffel[lam, mu, nu] += term
        
        self.christoffel_symbols = christoffel
        return christoffel
    
    def compute_riemann_tensor(self):
        """
        Compute the Riemann curvature tensor.
        
        Returns:
        --------
        list of lists of lists of lists
            Riemann tensor R^ρ_σμν
        """
        if self.christoffel_symbols is None:
            self.compute_christoffel_symbols()
            
        n = self.dimension
        coords = self.coordinates
        gamma = self.christoffel_symbols
        
        # Initialize Riemann tensor R^ρ_σμν (contravariant in first index, covariant in others)
        riemann = np.zeros((n, n, n, n), dtype=object)
        
        # Compute components
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        # First term: ∂_μ Γ^ρ_νσ
                        term1 = sp.diff(gamma[rho, nu, sigma], coords[mu])
                        
                        # Second term: ∂_ν Γ^ρ_μσ
                        term2 = sp.diff(gamma[rho, mu, sigma], coords[nu])
                        
                        # Third term: Γ^ρ_μλ Γ^λ_νσ
                        term3 = sp.S(0)
                        for lam in range(n):
                            term3 += gamma[rho, mu, lam] * gamma[lam, nu, sigma]
                        
                        # Fourth term: Γ^ρ_νλ Γ^λ_μσ
                        term4 = sp.S(0)
                        for lam in range(n):
                            term4 += gamma[rho, nu, lam] * gamma[lam, mu, sigma]
                        
                        # Combine terms
                        riemann[rho, sigma, mu, nu] = term1 - term2 + term3 - term4
        
        self.riemann_tensor = riemann
        return riemann
    
    def compute_ricci_tensor(self):
        """
        Compute the Ricci tensor by contracting the Riemann tensor.
        
        Returns:
        --------
        sympy.Matrix
            Ricci tensor R_μν
        """
        if self.riemann_tensor is None:
            self.compute_riemann_tensor()
            
        n = self.dimension
        
        # Initialize Ricci tensor
        ricci = sp.zeros(n, n)
        
        # Compute components by contraction: R_μν = R^λ_μλν
        for mu in range(n):
            for nu in range(n):
                for lam in range(n):
                    ricci[mu, nu] += self.riemann_tensor[lam, mu, lam, nu]
        
        self.ricci_tensor = ricci
        return ricci
    
    def compute_ricci_scalar(self):
        """
        Compute the Ricci scalar by contracting the Ricci tensor.
        
        Returns:
        --------
        sympy.Expr
            Ricci scalar R
        """
        if self.ricci_tensor is None:
            self.compute_ricci_tensor()
            
        n = self.dimension
        g_inv = self.inverse_metric
        ricci = self.ricci_tensor
        
        # Initialize Ricci scalar
        ricci_scalar = sp.S(0)
        
        # Compute by contraction: R = g^μν R_μν
        for mu in range(n):
            for nu in range(n):
                ricci_scalar += g_inv[mu, nu] * ricci[mu, nu]
        
        self.ricci_scalar = ricci_scalar
        return ricci_scalar
    
    def compute_weyl_tensor(self):
        """
        Compute the Weyl conformal curvature tensor.
        
        The Weyl tensor is the traceless part of the Riemann tensor and
        is invariant under conformal transformations of the metric.
        
        Returns:
        --------
        list of lists of lists of lists
            Weyl tensor C^ρ_σμν
        """
        if self.riemann_tensor is None:
            self.compute_riemann_tensor()
            
        if self.ricci_tensor is None:
            self.compute_ricci_tensor()
            
        if self.ricci_scalar is None:
            self.compute_ricci_scalar()
            
        n = self.dimension
        g = self.metric
        riemann = self.riemann_tensor
        ricci = self.ricci_tensor
        R = self.ricci_scalar
        
        # Initialize Weyl tensor
        weyl = np.zeros((n, n, n, n), dtype=object)
        
        # If dimension < 3, Weyl tensor is identically zero
        if n < 3:
            self.weyl_tensor = weyl
            return weyl
        
        # If dimension = 3, Weyl tensor is identically zero
        # but use the Cotton tensor instead (computed separately)
        if n == 3:
            self.weyl_tensor = weyl
            self.compute_cotton_tensor()
            return weyl
        
        # For dimension >= 4, compute Weyl tensor
        # C^ρ_σμν = R^ρ_σμν - 1/(n-2) * (δ^ρ_μ R_σν - δ^ρ_ν R_σμ + g_σμ R^ρ_ν - g_σν R^ρ_μ) + R/(n-1)(n-2) * (δ^ρ_μ g_σν - δ^ρ_ν g_σμ)
        
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        # Kronecker deltas
                        delta_rho_mu = 1 if rho == mu else 0
                        delta_rho_nu = 1 if rho == nu else 0
                        
                        # Ricci tensor with mixed indices
                        R_sigma_nu = sp.S(0)
                        R_sigma_mu = sp.S(0)
                        R_rho_nu = sp.S(0)
                        R_rho_mu = sp.S(0)
                        
                        for lambda_ in range(n):
                            R_sigma_nu += g[sigma, lambda_] * self.inverse_metric[lambda_, nu] * ricci[lambda_, nu]
                            R_sigma_mu += g[sigma, lambda_] * self.inverse_metric[lambda_, mu] * ricci[lambda_, mu]
                            R_rho_nu += g[rho, lambda_] * self.inverse_metric[lambda_, nu] * ricci[lambda_, nu]
                            R_rho_mu += g[rho, lambda_] * self.inverse_metric[lambda_, mu] * ricci[lambda_, mu]
                        
                        # Riemann term
                        term1 = riemann[rho, sigma, mu, nu]
                        
                        # Ricci correction terms
                        term2 = (delta_rho_mu * R_sigma_nu - delta_rho_nu * R_sigma_mu + 
                                g[sigma, mu] * R_rho_nu - g[sigma, nu] * R_rho_mu) / (n - 2)
                        
                        # Scalar curvature correction
                        term3 = R * (delta_rho_mu * g[sigma, nu] - delta_rho_nu * g[sigma, mu]) / ((n - 1) * (n - 2))
                        
                        # Combine terms
                        weyl[rho, sigma, mu, nu] = term1 - term2 + term3
        
        self.weyl_tensor = weyl
        return weyl
    
    def compute_schouten_tensor(self):
        """
        Compute the Schouten tensor.
        
        The Schouten tensor is used in the construction of the Weyl tensor
        and in conformal geometry.
        
        Returns:
        --------
        sympy.Matrix
            Schouten tensor P_μν
        """
        if self.ricci_tensor is None:
            self.compute_ricci_tensor()
            
        if self.ricci_scalar is None:
            self.compute_ricci_scalar()
            
        n = self.dimension
        ricci = self.ricci_tensor
        R = self.ricci_scalar
        
        # Initialize Schouten tensor
        schouten = sp.zeros(n, n)
        
        # Compute components: P_μν = 1/(n-2) * (R_μν - R/(2(n-1)) * g_μν)
        for mu in range(n):
            for nu in range(n):
                schouten[mu, nu] = (ricci[mu, nu] - R * self.metric[mu, nu] / (2 * (n - 1))) / (n - 2)
        
        self.schouten_tensor = schouten
        return schouten
    
    def compute_cotton_tensor(self):
        """
        Compute the Cotton tensor.
        
        The Cotton tensor is a conformally invariant tensor in 3 dimensions,
        analogous to the Weyl tensor in higher dimensions.
        
        Returns:
        --------
        list of lists of lists
            Cotton tensor C_ρμν
        """
        if self.schouten_tensor is None:
            self.compute_schouten_tensor()
            
        n = self.dimension
        coords = self.coordinates
        P = self.schouten_tensor
        
        # Initialize Cotton tensor
        cotton = np.zeros((n, n, n), dtype=object)
        
        # Compute components: C_ρμν = ∇_μ P_ρν - ∇_ν P_ρμ
        for rho in range(n):
            for mu in range(n):
                for nu in range(n):
                    # Partial derivatives
                    d_mu_P_rho_nu = sp.diff(P[rho, nu], coords[mu])
                    d_nu_P_rho_mu = sp.diff(P[rho, mu], coords[nu])
                    
                    # Christoffel terms for ∇_μ P_ρν
                    term1 = sp.S(0)
                    for sigma in range(n):
                        term1 -= self.christoffel_symbols[sigma, mu, rho] * P[sigma, nu]
                        term1 -= self.christoffel_symbols[sigma, mu, nu] * P[rho, sigma]
                    
                    # Christoffel terms for ∇_ν P_ρμ
                    term2 = sp.S(0)
                    for sigma in range(n):
                        term2 -= self.christoffel_symbols[sigma, nu, rho] * P[sigma, mu]
                        term2 -= self.christoffel_symbols[sigma, nu, mu] * P[rho, sigma]
                    
                    # Combine terms
                    cotton[rho, mu, nu] = d_mu_P_rho_nu + term1 - d_nu_P_rho_mu - term2
        
        self.cotton_tensor = cotton
        return cotton
    
    def compute_einstein_tensor(self):
        """
        Compute the Einstein tensor.
        
        Returns:
        --------
        sympy.Matrix
            Einstein tensor G_μν
        """
        if self.ricci_tensor is None:
            self.compute_ricci_tensor()
            
        if self.ricci_scalar is None:
            self.compute_ricci_scalar()
            
        n = self.dimension
        ricci = self.ricci_tensor
        R = self.ricci_scalar
        g = self.metric
        
        # Initialize Einstein tensor
        einstein = sp.zeros(n, n)
        
        # Compute components: G_μν = R_μν - 1/2 * R * g_μν
        for mu in range(n):
            for nu in range(n):
                einstein[mu, nu] = ricci[mu, nu] - (R * g[mu, nu]) / 2
        
        self.einstein_tensor = einstein
        return einstein
    
    def compute_all_tensors(self):
        """
        Compute all curvature tensors.
        
        Returns:
        --------
        dict
            Dictionary containing all computed tensors
        """
        self.compute_christoffel_symbols()
        self.compute_riemann_tensor()
        self.compute_ricci_tensor()
        self.compute_ricci_scalar()
        self.compute_weyl_tensor()
        self.compute_einstein_tensor()
        self.compute_schouten_tensor()
        
        if self.dimension == 3:
            self.compute_cotton_tensor()
        
        return {
            'christoffel': self.christoffel_symbols,
            'riemann': self.riemann_tensor,
            'ricci': self.ricci_tensor,
            'ricci_scalar': self.ricci_scalar,
            'weyl': self.weyl_tensor,
            'einstein': self.einstein_tensor,
            'schouten': self.schouten_tensor,
            'cotton': self.cotton_tensor if self.dimension == 3 else None
        }
    
    def apply_conformal_transformation(self, conformal_factor):
        """
        Apply a conformal transformation to the metric.
        
        A conformal transformation preserves angles but not distances.
        It scales the metric by a position-dependent factor:
        g_μν → Ω²(x) g_μν
        
        Parameters:
        -----------
        conformal_factor : sympy.Expr
            The conformal factor Ω²(x)
            
        Returns:
        --------
        TensorCalculus
            A new calculator with the transformed metric
        """
        n = self.dimension
        coords = self.coordinates
        
        # Create the transformed metric
        g_new = sp.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                g_new[mu, nu] = conformal_factor * self.metric[mu, nu]
        
        # Create a new calculator with the transformed metric
        transformed = TensorCalculus(n, coords, g_new)
        
        return transformed
    
    def visualize_curvature_scalar(self, expr, param_ranges, resolution=20, cmap='viridis'):
        """
        Visualize a curvature scalar on a 2D or 3D manifold.
        
        Parameters:
        -----------
        expr : sympy.Expr
            Curvature scalar expression to visualize
        param_ranges : list of tuples
            Ranges for parameters [(x_min, x_max), (y_min, y_max), ...]
        resolution : int
            Resolution of the visualization grid
        cmap : str
            Colormap for the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the visualization
        """
        n = len(param_ranges)
        if n not in [2, 3]:
            raise ValueError("Visualization only supported for 2D or 3D manifolds")
            
        # Create parameter grids
        if n == 2:
            x_range = np.linspace(param_ranges[0][0], param_ranges[0][1], resolution)
            y_range = np.linspace(param_ranges[1][0], param_ranges[1][1], resolution)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros((resolution, resolution))
            
            # Create callable function for the expression
            expr_fn = sp.lambdify(self.coordinates[:n], expr, 'numpy')
            
            # Evaluate the expression at each grid point
            for i in range(resolution):
                for j in range(resolution):
                    try:
                        Z[i, j] = float(expr_fn(X[i, j], Y[i, j]))
                    except:
                        Z[i, j] = np.nan
            
            # Create visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the scalar field as a surface
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.8)
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Add labels
            ax.set_xlabel(str(self.coordinates[0]))
            ax.set_ylabel(str(self.coordinates[1]))
            ax.set_zlabel(str(expr))
            
            return fig
            
        elif n == 3:
            # For 3D manifolds, create a 3D slice visualization
            x_range = np.linspace(param_ranges[0][0], param_ranges[0][1], resolution)
            y_range = np.linspace(param_ranges[1][0], param_ranges[1][1], resolution)
            z_val = (param_ranges[2][0] + param_ranges[2][1]) / 2  # Middle value for the slice
            
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros((resolution, resolution))
            
            # Create callable function for the expression
            expr_fn = sp.lambdify(self.coordinates[:n], expr, 'numpy')
            
            # Evaluate the expression at each grid point on the slice
            for i in range(resolution):
                for j in range(resolution):
                    try:
                        Z[i, j] = float(expr_fn(X[i, j], Y[i, j], z_val))
                    except:
                        Z[i, j] = np.nan
            
            # Create visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Plot the scalar field as a contour plot
            contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
            
            # Add colorbar
            fig.colorbar(contour, ax=ax)
            
            # Add labels
            ax.set_xlabel(str(self.coordinates[0]))
            ax.set_ylabel(str(self.coordinates[1]))
            ax.set_title(f"{expr} at {self.coordinates[2]} = {z_val}")
            
            return fig
    
    def check_weyl_conformal_invariance(self, conformal_factor):
        """
        Check the conformal invariance of the Weyl tensor.
        
        The Weyl tensor is invariant under conformal transformations
        when its indices are appropriately positioned.
        
        Parameters:
        -----------
        conformal_factor : sympy.Expr
            The conformal factor Ω²(x)
            
        Returns:
        --------
        dict
            Dictionary containing information about the conformal transformation
        """
        # Compute the Weyl tensor for the original metric
        if self.weyl_tensor is None:
            self.compute_weyl_tensor()
        
        # Apply conformal transformation
        transformed = self.apply_conformal_transformation(conformal_factor)
        
        # Compute the Weyl tensor for the transformed metric
        transformed.compute_weyl_tensor()
        
        # Extract the conformal factor as Ω (not Ω²)
        omega = sp.sqrt(conformal_factor)
        
        # Check the transformation law for the Weyl tensor:
        # C^ρ_σμν → C^ρ_σμν (when properly indexed)
        # or equivalently: C_ρσμν → Ω² C_ρσμν (all indices lowered)
        
        # We'll check a specific component as an example
        n = self.dimension
        if n < 4:
            return {
                'original_metric': self.metric,
                'transformed_metric': transformed.metric,
                'conformal_factor': conformal_factor,
                'weyl_invariance': "Weyl tensor is identically zero in dimension < 4"
            }
        
        # Check for a specific component (with all indices lowered)
        i, j, k, l = 0, 1, 2, 3  # Example indices
        
        # Original Weyl component with indices lowered
        original_weyl_lowered = sp.S(0)
        for rho in range(n):
            original_weyl_lowered += self.metric[i, rho] * self.weyl_tensor[rho, j, k, l]
        
        # Transformed Weyl component with indices lowered
        transformed_weyl_lowered = sp.S(0)
        for rho in range(n):
            transformed_weyl_lowered += transformed.metric[i, rho] * transformed.weyl_tensor[rho, j, k, l]
        
        # Expected relationship: transformed_weyl_lowered = Ω² * original_weyl_lowered
        expected = conformal_factor * original_weyl_lowered
        
        # Check if the relationship holds (subject to simplification)
        difference = sp.simplify(transformed_weyl_lowered - expected)
        invariance_holds = (difference == 0)
        
        return {
            'original_metric': self.metric,
            'transformed_metric': transformed.metric,
            'conformal_factor': conformal_factor,
            'original_weyl_sample': original_weyl_lowered,
            'transformed_weyl_sample': transformed_weyl_lowered,
            'expected_relation': expected,
            'difference': difference,
            'weyl_invariance': invariance_holds
        }


class GaugeTheory:
    """
    Implementation of gauge theory concepts in physics.
    
    Gauge theory is a framework that describes how symmetry transformations
    can vary from point to point in spacetime, leading to the emergence of
    gauge fields that mediate fundamental interactions.
    """
    
    def __init__(self, manifold_dim=4, gauge_group='U(1)'):
        """
        Initialize the gauge theory framework.
        
        Parameters:
        -----------
        manifold_dim : int
            Dimension of the base manifold (typically spacetime)
        gauge_group : str
            Gauge group name ('U(1)', 'SU(2)', 'SU(3)', etc.)
        """
        self.manifold_dim = manifold_dim
        self.gauge_group = gauge_group
        
        # Set up coordinate symbols
        if manifold_dim <= 4:
            # Use standard spacetime coordinates
            self.coordinates = sp.symbols('t x y z')[:manifold_dim]
        else:
            # Use generic coordinates for higher dimensions
            self.coordinates = sp.symbols('x_0:%d' % manifold_dim)
        
        # Setup the structure based on the gauge group
        if gauge_group == 'U(1)':
            # U(1) gauge theory (e.g., electromagnetism)
            self.group_dim = 1
            self.generators = [sp.Matrix([[0, -1], [1, 0]])]  # Corresponds to i*σ_y
            self.structure_constants = np.zeros((1, 1, 1))
            
        elif gauge_group == 'SU(2)':
            # SU(2) gauge theory (e.g., weak interaction)
            self.group_dim = 3
            # Pauli matrices as generators (divided by 2i)
            self.generators = [
                sp.Matrix([[0, 1], [1, 0]]) / 2,
                sp.Matrix([[0, -sp.I], [sp.I, 0]]) / 2,
                sp.Matrix([[1, 0], [0, -1]]) / 2
            ]
            # SU(2) structure constants (fully antisymmetric)
            self.structure_constants = np.zeros((3, 3, 3))
            self.structure_constants[0, 1, 2] = 1  # f_{012} = 1
            # Fill in using antisymmetry
            for i, j, k in itertools.permutations([0, 1, 2]):
                sgn = 1 if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else -1
                if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2)]:
                    self.structure_constants[i, j, k] = sgn
            
        elif gauge_group == 'SU(3)':
            # SU(3) gauge theory (e.g., strong interaction)
            self.group_dim = 8
            # The 8 Gell-Mann matrices as generators
            # We'll use a simplified approach
            self.generators = None  # Would need to define all 8 Gell-Mann matrices
            self.structure_constants = np.zeros((8, 8, 8))
            # Would fill in the structure constants
            
        else:
            raise ValueError(f"Gauge group {gauge_group} not implemented")
        
        # Initialize gauge field, field strength, and associated quantities
        self.gauge_potential = None
        self.field_strength = None
        self.covariant_derivative = None
        self.action = None
    
    def set_gauge_potential(self, expressions):
        """
        Set the gauge potential (connection) A_μ.
        
        Parameters:
        -----------
        expressions : list of sympy.Expr
            Components of the gauge potential for each spacetime index
            and group generator
            
        Returns:
        --------
        self
            For method chaining
        """
        n = self.manifold_dim
        g = self.group_dim
        
        # Initialize gauge potential
        if isinstance(expressions, list):
            if len(expressions) != n * g:
                raise ValueError(f"Expected {n * g} expressions for gauge potential")
            
            # Reshape into n × g array
            self.gauge_potential = np.array(expressions).reshape(n, g)
            
        elif isinstance(expressions, np.ndarray):
            if expressions.shape != (n, g):
                raise ValueError(f"Expected {n}×{g} array for gauge potential")
            
            self.gauge_potential = expressions
            
        else:
            raise ValueError("Expressions must be a list or numpy array")
        
        return self
    
    def compute_field_strength(self):
        """
        Compute the field strength tensor F_μν.
        
        For a gauge field A_μ, the field strength is given by:
        F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]
        where the last term is the Lie bracket that vanishes for abelian groups.
        
        Returns:
        --------
        numpy.ndarray
            Field strength tensor
        """
        if self.gauge_potential is None:
            raise ValueError("Gauge potential must be set first")
            
        n = self.manifold_dim
        g = self.group_dim
        A = self.gauge_potential
        coords = self.coordinates
        
        # Initialize field strength tensor
        F = np.zeros((n, n, g), dtype=object)
        
        # Compute components
        for mu in range(n):
            for nu in range(n):
                for a in range(g):
                    # Terms from partial derivatives
                    F[mu, nu, a] = sp.diff(A[nu, a], coords[mu]) - sp.diff(A[mu, a], coords[nu])
                    
                    # Add Lie bracket term for non-abelian groups
                    if g > 1:
                        # [A_μ, A_ν] = Σ_bc f^a_bc A_μ^b A_ν^c
                        for b in range(g):
                            for c in range(g):
                                F[mu, nu, a] += self.structure_constants[a, b, c] * A[mu, b] * A[nu, c]
        
        self.field_strength = F
        return F
    
    def gauge_transform(self, transformation_param):
        """
        Apply a gauge transformation to the gauge potential.
        
        For a U(1) gauge theory: A_μ → A_μ + ∂_μ λ
        For non-abelian gauge theories: A_μ → U A_μ U^† + U ∂_μ U^†
        
        Parameters:
        -----------
        transformation_param : sympy.Expr or list
            Gauge transformation parameter(s)
            
        Returns:
        --------
        numpy.ndarray
            Transformed gauge potential
        """
        if self.gauge_potential is None:
            raise ValueError("Gauge potential must be set first")
            
        n = self.manifold_dim
        g = self.group_dim
        A = self.gauge_potential
        coords = self.coordinates
        
        # Handle transformation parameter based on gauge group
        if self.gauge_group == 'U(1)':
            # For U(1), the transformation is A_μ → A_μ + ∂_μ λ
            lambda_param = transformation_param
            
            # Initialize transformed potential
            A_transformed = np.zeros((n, g), dtype=object)
            
            # Apply transformation
            for mu in range(n):
                A_transformed[mu, 0] = A[mu, 0] + sp.diff(lambda_param, coords[mu])
                
        else:
            # For non-abelian groups, need to implement the full transformation
            # This is a simplified placeholder
            A_transformed = A.copy()
        
        return A_transformed
    
    def compute_covariant_derivative(self, field, field_type='scalar'):
        """
        Compute the gauge covariant derivative of a field.
        
        Parameters:
        -----------
        field : sympy.Expr or numpy.ndarray
            The field to differentiate
        field_type : str
            Type of the field ('scalar', 'vector', etc.)
            
        Returns:
        --------
        numpy.ndarray
            Covariant derivative of the field
        """
        if self.gauge_potential is None:
            raise ValueError("Gauge potential must be set first")
            
        n = self.manifold_dim
        g = self.group_dim
        A = self.gauge_potential
        coords = self.coordinates
        
        # Handle different field types
        if field_type == 'scalar':
            # For a scalar field φ, the covariant derivative is:
            # D_μ φ = ∂_μ φ + A_μ φ  (for U(1))
            # or more generally: D_μ φ = ∂_μ φ + A_μ^a T_a φ
            
            # Initialize covariant derivative
            D_field = np.zeros(n, dtype=object)
            
            # Compute components
            for mu in range(n):
                # Partial derivative term
                D_field[mu] = sp.diff(field, coords[mu])
                
                # Gauge term
                if self.gauge_group == 'U(1)':
                    D_field[mu] += A[mu, 0] * field
                else:
                    # For non-abelian groups, need to apply generators
                    for a in range(g):
                        # This is a simplified approach
                        D_field[mu] += A[mu, a] * field
                        
        elif field_type == 'vector':
            # For a vector field V^ν, the covariant derivative is:
            # D_μ V^ν = ∂_μ V^ν + [A_μ, V^ν]
            
            # Initialize covariant derivative
            D_field = np.zeros((n, n), dtype=object)
            
            # Compute components
            for mu in range(n):
                for nu in range(n):
                    # Partial derivative term
                    D_field[mu, nu] = sp.diff(field[nu], coords[mu])
                    
                    # Gauge term
                    if self.gauge_group == 'U(1)':
                        D_field[mu, nu] += A[mu, 0] * field[nu]
                    else:
                        # For non-abelian groups, need to apply generators
                        for a in range(g):
                            # This is a simplified approach
                            D_field[mu, nu] += A[mu, a] * field[nu]
        
        else:
            raise ValueError(f"Field type {field_type} not implemented")
        
        self.covariant_derivative = D_field
        return D_field
    
    def compute_yang_mills_action(self):
        """
        Compute the Yang-Mills action for the gauge field.
        
        The Yang-Mills action is: S = -1/4 ∫ Tr(F_μν F^μν) d⁴x
        
        Returns:
        --------
        sympy.Expr
            Yang-Mills action
        """
        if self.field_strength is None:
            self.compute_field_strength()
            
        n = self.manifold_dim
        g = self.group_dim
        F = self.field_strength
        
        # For simplicity, assume Minkowski metric
        metric = sp.diag(*([[-1] + [1]*(n-1)]))
        metric_inv = metric.inv()
        
        # Initialize action
        action = sp.S(0)
        
        # Compute Tr(F_μν F^μν)
        for mu in range(n):
            for nu in range(n):
                for rho in range(n):
                    for sigma in range(n):
                        # Raise indices with metric: F^μν = g^μρ g^νσ F_ρσ
                        for a in range(g):
                            action -= metric_inv[mu, rho] * metric_inv[nu, sigma] * F[mu, nu, a] * F[rho, sigma, a] / 4
        
        self.action = action
        return action
    
    def compute_equations_of_motion(self):
        """
        Compute the equations of motion for the gauge field.
        
        The Yang-Mills equations are: D_μ F^μν = 0
        
        Returns:
        --------
        numpy.ndarray
            Equations of motion
        """
        if self.field_strength is None:
            self.compute_field_strength()
            
        n = self.manifold_dim
        g = self.group_dim
        F = self.field_strength
        coords = self.coordinates
        
        # For simplicity, assume Minkowski metric
        metric = sp.diag(*([[-1] + [1]*(n-1)]))
        metric_inv = metric.inv()
        
        # Initialize equations of motion
        eom = np.zeros((n, g), dtype=object)
        
        # Compute D_μ F^μν
        for nu in range(n):
            for a in range(g):
                # Partial derivative term
                for mu in range(n):
                    for rho in range(n):
                        for sigma in range(n):
                            term = metric_inv[mu, rho] * metric_inv[nu, sigma] * sp.diff(F[rho, sigma, a], coords[mu])
                            eom[nu, a] += term
                
                # Gauge term for non-abelian theories
                if g > 1:
                    for mu in range(n):
                        for b in range(g):
                            for rho in range(n):
                                for sigma in range(n):
                                    term = metric_inv[mu, rho] * metric_inv[nu, sigma] * self.gauge_potential[mu, b] * F[rho, sigma, a]
                                    eom[nu, a] += term
        
        return eom
    
    def electromagnetic_field_examples(self):
        """
        Generate examples of electromagnetic fields in U(1) gauge theory.
        
        Returns:
        --------
        list
            List of example electromagnetic fields
        """
        if self.gauge_group != 'U(1)':
            raise ValueError("This method is only for U(1) gauge theory")
            
        n = self.manifold_dim
        coords = self.coordinates
        
        examples = []
        
        # Example 1: Constant magnetic field in z-direction
        if n >= 3:
            # Gauge potential for B = (0, 0, B₀)
            A_1 = np.zeros((n, 1), dtype=object)
            B0 = sp.Symbol('B_0')
            
            # A = (-B₀y/2, B₀x/2, 0)
            A_1[0, 0] = 0
            A_1[1, 0] = -B0 * coords[2] / 2
            A_1[2, 0] = B0 * coords[1] / 2
            
            # Compute field strength
            self.gauge_potential = A_1
            F_1 = self.compute_field_strength()
            
            examples.append({
                'name': 'Constant magnetic field in z-direction',
                'gauge_potential': A_1,
                'field_strength': F_1
            })
        
        # Example 2: Plane wave in x-direction
        if n >= 4:
            # Gauge potential for a plane wave
            A_2 = np.zeros((n, 1), dtype=object)
            
            # A = (0, A₀ cos(kz - ωt), 0, 0)
            A0 = sp.Symbol('A_0')
            k = sp.Symbol('k')
            omega = sp.Symbol('omega')
            
            A_2[0, 0] = 0
            A_2[1, 0] = A0 * sp.cos(k * coords[2] - omega * coords[0])
            A_2[2, 0] = 0
            A_2[3, 0] = 0
            
            # Compute field strength
            self.gauge_potential = A_2
            F_2 = self.compute_field_strength()
            
            examples.append({
                'name': 'Plane wave in x-direction',
                'gauge_potential': A_2,
                'field_strength': F_2
            })
        
        # Example 3: Coulomb potential
        if n >= 4:
            # Gauge potential for Coulomb field
            A_3 = np.zeros((n, 1), dtype=object)
            
            # A = (q/r, 0, 0, 0)
            q = sp.Symbol('q')
            r = sp.sqrt(coords[1]**2 + coords[2]**2 + coords[3]**2)
            
            A_3[0, 0] = q / r
            
            # Compute field strength
            self.gauge_potential = A_3
            F_3 = self.compute_field_strength()
            
            examples.append({
                'name': 'Coulomb potential',
                'gauge_potential': A_3,
                'field_strength': F_3
            })
        
        return examples
    
    def visualize_gauge_field(self, field_component=(0, 0), domain=None, resolution=20):
        """
        Visualize a component of the gauge field in 2D or 3D.
        
        Parameters:
        -----------
        field_component : tuple
            (mu, a) indices of the gauge field component to visualize
        domain : list of tuples, optional
            [(x_min, x_max), (y_min, y_max), ...] domain limits
        resolution : int
            Resolution of the visualization grid
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the visualization
        """
        if self.gauge_potential is None:
            raise ValueError("Gauge potential must be set first")
            
        if domain is None:
            # Default domain
            domain = [(-1, 1), (-1, 1)]
            
        # Extract the component to visualize
        mu, a = field_component
        A_component = self.gauge_potential[mu, a]
        
        # Create parameter grids
        x_range = np.linspace(domain[0][0], domain[0][1], resolution)
        y_range = np.linspace(domain[1][0], domain[1][1], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros((resolution, resolution))
        
        # Create callable function for the expression
        component_fn = sp.lambdify([self.coordinates[0], self.coordinates[1]], A_component, 'numpy')
        
        # Evaluate the expression at each grid point
        for i in range(resolution):
            for j in range(resolution):
                try:
                    Z[i, j] = float(component_fn(X[i, j], Y[i, j]))
                except:
                    Z[i, j] = np.nan
        
        # Create visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the field component as a surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Add labels
        ax.set_xlabel(str(self.coordinates[0]))
        ax.set_ylabel(str(self.coordinates[1]))
        ax.set_zlabel(f'A_{mu}^{a}')
        
        mu_names = ['t', 'x', 'y', 'z']
        if mu < len(mu_names):
            mu_label = mu_names[mu]
        else:
            mu_label = f'x_{mu}'
        
        ax.set_title(f'Gauge Field Component A_{mu_label}^{a}')
        
        return fig
    
    def create_maxwell_field_strength(self, electric_field, magnetic_field):
        """
        Create the Maxwell field strength tensor from electric and magnetic fields.
        
        Parameters:
        -----------
        electric_field : list or numpy.ndarray
            Components of the electric field (E_x, E_y, E_z)
        magnetic_field : list or numpy.ndarray
            Components of the magnetic field (B_x, B_y, B_z)
            
        Returns:
        --------
        numpy.ndarray
            Field strength tensor F_μν
        """
        if self.gauge_group != 'U(1)':
            raise ValueError("This method is only for U(1) gauge theory")
            
        if self.manifold_dim < 4:
            raise ValueError("Need at least 4D spacetime for Maxwell fields")
            
        # Convert inputs to arrays
        E = np.asarray(electric_field)
        B = np.asarray(magnetic_field)
        
        if len(E) != 3 or len(B) != 3:
            raise ValueError("Electric and magnetic fields must have 3 components each")
            
        # Create field strength tensor
        F = np.zeros((4, 4, 1), dtype=object)
        
        # Electric field components: F_0i = E_i
        for i in range(3):
            F[0, i+1, 0] = E[i]
            F[i+1, 0, 0] = -E[i]
        
        # Magnetic field components: F_ij = ε_ijk B_k
        # B_x = F_23, B_y = F_31, B_z = F_12
        F[1, 2, 0] = B[2]
        F[2, 1, 0] = -B[2]
        F[2, 3, 0] = B[0]
        F[3, 2, 0] = -B[0]
        F[3, 1, 0] = B[1]
        F[1, 3, 0] = -B[1]
        
        self.field_strength = F
        return F
    
    def compute_gauge_invariant_quantity(self, field_strength=None):
        """
        Compute a gauge invariant quantity from the field strength.
        
        For U(1): F_μν F^μν (related to E² - B²)
        For non-abelian: Tr(F_μν F^μν)
        
        Parameters:
        -----------
        field_strength : numpy.ndarray, optional
            Field strength tensor to use (uses stored tensor if None)
            
        Returns:
        --------
        sympy.Expr
            Gauge invariant quantity
        """
        if field_strength is None:
            if self.field_strength is None:
                self.compute_field_strength()
            field_strength = self.field_strength
            
        n = self.manifold_dim
        g = self.group_dim
        F = field_strength
        
        # For simplicity, assume Minkowski metric
        metric = sp.diag(*([[-1] + [1]*(n-1)]))
        metric_inv = metric.inv()
        
        # Compute invariant
        invariant = sp.S(0)
        
        for mu in range(n):
            for nu in range(n):
                for rho in range(n):
                    for sigma in range(n):
                        for a in range(g):
                            invariant += metric_inv[mu, rho] * metric_inv[nu, sigma] * F[mu, nu, a] * F[rho, sigma, a]
        
        return invariant


# Helper Class for Demonstrations
class DifferentialGeometryDemos:
    """Helper class for demonstrations and examples of tensor calculus and gauge theory."""
    
    @staticmethod
    def riemann_sphere_example():
        """Demonstrate the Weyl tensor on the Riemann sphere."""
        # Create symbolic variables
        theta, phi = sp.symbols('theta phi')
        
        # Define the metric on the unit sphere
        g = sp.Matrix([
            [1, 0],
            [0, sp.sin(theta)**2]
        ])
        
        # Initialize tensor calculator
        calculator = TensorCalculus(2, [theta, phi], g)
        
        # Compute all curvature tensors
        tensors = calculator.compute_all_tensors()
        
        # Verify that the Weyl tensor vanishes in 2D
        if np.all(tensors['weyl'] == 0):
            print("Weyl tensor vanishes for the 2D sphere as expected")
        
        # Print the Ricci scalar
        ricci_scalar = sp.simplify(tensors['ricci_scalar'])
        print(f"Ricci scalar for the unit sphere: {ricci_scalar}")
        
        # Verify the Einstein tensor
        einstein = sp.simplify(tensors['einstein'])
        print("Einstein tensor for the unit sphere:")
        sp.pprint(einstein)
        
        return calculator
    
    @staticmethod
    def schwarzschild_example():
        """Demonstrate the Weyl tensor for the Schwarzschild metric."""
        # Create symbolic variables
        t, r, theta, phi = sp.symbols('t r theta phi')
        M = sp.Symbol('M', positive=True)  # Mass parameter
        
        # Schwarzschild metric
        g = sp.Matrix([
            [-(1 - 2*M/r), 0, 0, 0],
            [0, 1/(1 - 2*M/r), 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2]
        ])
        
        # Initialize tensor calculator
        calculator = TensorCalculus(4, [t, r, theta, phi], g)
        
        # Compute components of interest
        calculator.compute_ricci_tensor()
        calculator.compute_ricci_scalar()
        calculator.compute_weyl_tensor()
        
        # Check if Ricci tensor vanishes (Schwarzschild is vacuum solution)
        ricci_flat = sp.simplify(calculator.ricci_tensor) == sp.zeros(4, 4)
        print("Is Schwarzschild spacetime Ricci-flat?", ricci_flat)
        
        # Check Weyl tensor properties
        weyl = calculator.weyl_tensor
        nonzero_count = 0
        
        # Count non-zero independent components
        for i, j, k, l in itertools.product(range(4), repeat=4):
            component = sp.simplify(weyl[i, j, k, l])
            if component != 0 and i < j and k < l:  # Count only independent components
                nonzero_count += 1
        
        print(f"Number of non-zero independent Weyl tensor components: {nonzero_count}")
        
        return calculator
    
    @staticmethod
    def conformal_invariance_example():
        """Demonstrate the conformal invariance of the Weyl tensor."""
        # Create a simple 4D spacetime metric
        t, x, y, z = sp.symbols('t x y z')
        
        # Start with Minkowski metric
        g = sp.diag(-1, 1, 1, 1)
        
        # Initialize tensor calculator
        calculator = TensorCalculus(4, [t, x, y, z], g)
        
        # Define a conformal factor
        omega_squared = sp.exp(2 * sp.Function('omega')(t, x, y, z))
        
        # Check Weyl tensor transformation
        result = calculator.check_weyl_conformal_invariance(omega_squared)
        
        # Display results
        print("Original metric:")
        sp.pprint(result['original_metric'])
        
        print("\nConformally transformed metric:")
        sp.pprint(result['transformed_metric'])
        
        print("\nConformal factor:")
        sp.pprint(result['conformal_factor'])
        
        print("\nDoes the Weyl tensor transform correctly under conformal transformations?")
        print(result['weyl_invariance'])
        
        return result
    
    @staticmethod
    def electromagnetic_gauge_example():
        """Demonstrate U(1) gauge theory concepts."""
        # Create a U(1) gauge theory in 4D spacetime
        gauge_theory = GaugeTheory(4, 'U(1)')
        
        # Define coordinates
        t, x, y, z = gauge_theory.coordinates
        
        # Set up a gauge potential for a plane electromagnetic wave
        A0 = sp.Symbol('A_0')
        k = sp.Symbol('k', real=True)
        omega = sp.Symbol('omega', real=True)
        
        # A_μ = (0, A₀ cos(kz - ωt), 0, 0)
        A = np.zeros((4, 1), dtype=object)
        A[0, 0] = 0
        A[1, 0] = A0 * sp.cos(k * z - omega * t)
        A[2, 0] = 0
        A[3, 0] = 0
        
        gauge_theory.set_gauge_potential(A)
        
        # Compute the field strength tensor
        F = gauge_theory.compute_field_strength()
        
        # Extract electric and magnetic field components
        E = [
            F[0, 1, 0],  # E_x = F_01
            F[0, 2, 0],  # E_y = F_02
            F[0, 3, 0]   # E_z = F_03
        ]
        
        B = [
            F[2, 3, 0],  # B_x = F_23
            F[3, 1, 0],  # B_y = F_31
            F[1, 2, 0]   # B_z = F_12
        ]
        
        print("Electric field components:")
        for i, component in enumerate(['x', 'y', 'z']):
            print(f"E_{component} = {sp.simplify(E[i])}")
            
        print("\nMagnetic field components:")
        for i, component in enumerate(['x', 'y', 'z']):
            print(f"B_{component} = {sp.simplify(B[i])}")
        
        # Perform a gauge transformation
        lambda_param = sp.Function('lambda')(t, x, y, z)
        A_transformed = gauge_theory.gauge_transform(lambda_param)
        
        print("\nGauge transformation:")
        print(f"Original A_1 = {A[1, 0]}")
        print(f"Transformed A_1 = {sp.simplify(A_transformed[1, 0])}")
        
        # Compute a gauge invariant quantity
        invariant = gauge_theory.compute_gauge_invariant_quantity()
        print("\nGauge invariant quantity F_μν F^μν:")
        print(sp.simplify(invariant))
        
        return gauge_theory
    
    @staticmethod
    def yang_mills_example():
        """Demonstrate non-abelian SU(2) gauge theory concepts."""
        # Create an SU(2) gauge theory in 4D spacetime
        gauge_theory = GaugeTheory(4, 'SU(2)')
        
        # Define coordinates
        t, x, y, z = gauge_theory.coordinates
        
        # Set up a simple gauge potential
        g = sp.Symbol('g', real=True)  # Coupling constant
        
        # A^a_μ components
        A = np.zeros((4, 3), dtype=object)
        # A^1_μ
        A[0, 0] = 0
        A[1, 0] = 0
        A[2, 0] = 0
        A[3, 0] = 0
        
        # A^2_μ
        A[0, 1] = 0
        A[1, 1] = g * z
        A[2, 1] = 0
        A[3, 1] = 0
        
        # A^3_μ
        A[0, 2] = 0
        A[1, 2] = 0
        A[2, 2] = -g * x
        A[3, 2] = 0
        
        gauge_theory.set_gauge_potential(A)
        
        # Compute the field strength tensor
        F = gauge_theory.compute_field_strength()
        
        # Look at some components
        print("Some field strength components:")
        print(f"F^1_12 = {sp.simplify(F[1, 2, 0])}")
        print(f"F^2_13 = {sp.simplify(F[1, 3, 1])}")
        print(f"F^3_23 = {sp.simplify(F[2, 3, 2])}")
        
        # Compute Yang-Mills action
        action = gauge_theory.compute_yang_mills_action()
        print("\nYang-Mills action:")
        print(sp.simplify(action))
        
        return gauge_theory


def main():
    """Main function to demonstrate tensor calculus and gauge theory concepts."""
    print("\nDEMONSTRATING TENSOR CALCULUS AND THE WEYL TENSOR\n")
    
    # Create a demo instance
    demos = DifferentialGeometryDemos()
    
    # Example 1: Riemann sphere
    print("Example 1: Riemann Sphere")
    print("-----------------------")
    sphere_calculator = demos.riemann_sphere_example()
    
    # Example 2: Schwarzschild spacetime
    print("\nExample 2: Schwarzschild Spacetime")
    print("-------------------------------")
    schwarzschild_calculator = demos.schwarzschild_example()
    
    # Example 3: Conformal invariance
    print("\nExample 3: Conformal Invariance of the Weyl Tensor")
    print("----------------------------------------------")
    conformal_result = demos.conformal_invariance_example()
    
    print("\n\nDEMONSTRATING GAUGE THEORY\n")
    
    # Example 4: Electromagnetic U(1) gauge theory
    print("Example 4: Electromagnetic U(1) Gauge Theory")
    print("-----------------------------------------")
    em_gauge = demos.electromagnetic_gauge_example()
    
    # Example 5: Yang-Mills SU(2) gauge theory
    print("\nExample 5: Yang-Mills SU(2) Gauge Theory")
    print("-------------------------------------")
    ym_gauge = demos.yang_mills_example()
    
    # Visualizations
    print("\nCreating visualizations...")
    
    # Visualize Ricci scalar on sphere
    try:
        param_ranges = [(0.1, np.pi - 0.1), (0, 2*np.pi)]
        sphere_calculator.visualize_curvature_scalar(sphere_calculator.ricci_scalar, param_ranges)
        plt.title("Ricci Scalar on the Sphere")
        plt.savefig("ricci_scalar_sphere.png")
        
        # Visualize gauge field component
        em_gauge.visualize_gauge_field()
        plt.title("Electromagnetic Gauge Field Component")
        plt.savefig("gauge_field_component.png")
        
        print("Visualizations saved as PNG files")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()