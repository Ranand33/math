import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, Matrix, Function, sin, cos, tan, exp, log, sqrt, diff, simplify
from sympy.abc import u, v, t
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import LightSource
from IPython.display import display, Math

class DifferentialGeometry:
    """
    A class for implementing concepts of differential geometry,
    focusing on surfaces, geodesics, and intrinsic coordinate systems.
    """
    
    def __init__(self):
        """Initialize symbolic variables for calculations."""
        # Define symbolic variables
        self.u, self.v = symbols('u v', real=True)
        self.t = symbols('t', real=True)
        self.sympy_vars = {'u': self.u, 'v': self.v, 't': self.t}
        
        # For function definitions
        self.u_t = Function('u')(self.t)
        self.v_t = Function('v')(self.t)
        self.udot = diff(self.u_t, self.t)
        self.vdot = diff(self.v_t, self.t)
        self.uddot = diff(self.udot, self.t)
        self.vddot = diff(self.vdot, self.t)
    
    def create_surface(self, surface_type='sphere', params=None, u_range=None, v_range=None):
        """
        Create a parametric surface.
        
        Parameters:
        -----------
        surface_type : str
            Type of surface ('sphere', 'torus', 'saddle', 'helicoid', 'catenoid', etc.)
        params : dict
            Parameters for the surface (e.g., radius for sphere)
        u_range : tuple
            Range for the u parameter (u_min, u_max)
        v_range : tuple
            Range for the v parameter (v_min, v_max)
            
        Returns:
        --------
        dict
            Surface information including the parametrization
        """
        # Default parameter values
        if params is None:
            params = {}
        
        # Create the surface parametrization
        if surface_type == 'sphere':
            R = params.get('radius', 1.0)
            
            # Standard spherical coordinates (u=θ, v=φ)
            x_expr = R * sp.sin(self.u) * sp.cos(self.v)
            y_expr = R * sp.sin(self.u) * sp.sin(self.v)
            z_expr = R * sp.cos(self.u)
            
            if u_range is None:
                u_range = (0, np.pi)
            if v_range is None:
                v_range = (0, 2 * np.pi)
            
            name = f"Sphere (R={R})"
            
        elif surface_type == 'torus':
            R = params.get('major_radius', 2.0)
            r = params.get('minor_radius', 1.0)
            
            x_expr = (R + r * sp.cos(self.v)) * sp.cos(self.u)
            y_expr = (R + r * sp.cos(self.v)) * sp.sin(self.u)
            z_expr = r * sp.sin(self.v)
            
            if u_range is None:
                u_range = (0, 2 * np.pi)
            if v_range is None:
                v_range = (0, 2 * np.pi)
            
            name = f"Torus (R={R}, r={r})"
            
        elif surface_type == 'saddle':
            a = params.get('a', 1.0)
            b = params.get('b', 1.0)
            
            x_expr = self.u
            y_expr = self.v
            z_expr = a * self.u**2 - b * self.v**2
            
            if u_range is None:
                u_range = (-1, 1)
            if v_range is None:
                v_range = (-1, 1)
            
            name = f"Saddle (a={a}, b={b})"
            
        elif surface_type == 'helicoid':
            a = params.get('a', 1.0)
            
            x_expr = self.u * sp.cos(self.v)
            y_expr = self.u * sp.sin(self.v)
            z_expr = a * self.v
            
            if u_range is None:
                u_range = (0, 2)
            if v_range is None:
                v_range = (0, 4 * np.pi)
            
            name = f"Helicoid (a={a})"
            
        elif surface_type == 'catenoid':
            a = params.get('a', 1.0)
            
            x_expr = a * sp.cosh(self.u / a) * sp.cos(self.v)
            y_expr = a * sp.cosh(self.u / a) * sp.sin(self.v)
            z_expr = self.u
            
            if u_range is None:
                u_range = (-2, 2)
            if v_range is None:
                v_range = (0, 2 * np.pi)
            
            name = f"Catenoid (a={a})"
            
        elif surface_type == 'pseudosphere':
            a = params.get('a', 1.0)
            
            x_expr = a * sp.sin(self.u) * sp.cos(self.v)
            y_expr = a * sp.sin(self.u) * sp.sin(self.v)
            z_expr = a * (sp.log(sp.tan(self.u/2)) + sp.cos(self.u))
            
            if u_range is None:
                u_range = (0.1, np.pi - 0.1)  # Avoid singularities
            if v_range is None:
                v_range = (0, 2 * np.pi)
            
            name = f"Pseudosphere (a={a})"
            
        elif surface_type == 'monkey_saddle':
            x_expr = self.u
            y_expr = self.v
            z_expr = self.u**3 - 3 * self.u * self.v**2
            
            if u_range is None:
                u_range = (-1, 1)
            if v_range is None:
                v_range = (-1, 1)
            
            name = "Monkey Saddle"
            
        elif surface_type == 'custom':
            # Custom surface defined by given expressions
            if 'x_expr' not in params or 'y_expr' not in params or 'z_expr' not in params:
                raise ValueError("Custom surface requires x_expr, y_expr, and z_expr parameters")
            
            x_expr = params['x_expr']
            y_expr = params['y_expr']
            z_expr = params['z_expr']
            
            if u_range is None:
                u_range = (-1, 1)
            if v_range is None:
                v_range = (-1, 1)
            
            name = params.get('name', "Custom Surface")
            
        else:
            raise ValueError(f"Unknown surface type: {surface_type}")
        
        # Create lambdified functions for numerical evaluation
        x_func = sp.lambdify((self.u, self.v), x_expr, "numpy")
        y_func = sp.lambdify((self.u, self.v), y_expr, "numpy")
        z_func = sp.lambdify((self.u, self.v), z_expr, "numpy")
        
        # Return surface information
        surface = {
            'name': name,
            'type': surface_type,
            'params': params,
            'x_expr': x_expr,
            'y_expr': y_expr,
            'z_expr': z_expr,
            'x_func': x_func,
            'y_func': y_func,
            'z_func': z_func,
            'u_range': u_range,
            'v_range': v_range
        }
        
        return surface
    
    def compute_fundamental_forms(self, surface):
        """
        Compute the first and second fundamental forms for a surface.
        
        Parameters:
        -----------
        surface : dict
            Surface information including parametrization
            
        Returns:
        --------
        dict
            First and second fundamental forms and related quantities
        """
        # Extract surface parametrization
        x_expr = surface['x_expr']
        y_expr = surface['y_expr']
        z_expr = surface['z_expr']
        
        # Partial derivatives
        x_u = diff(x_expr, self.u)
        y_u = diff(y_expr, self.u)
        z_u = diff(z_expr, self.u)
        
        x_v = diff(x_expr, self.v)
        y_v = diff(y_expr, self.v)
        z_v = diff(z_expr, self.v)
        
        # Tangent vectors
        r_u = Matrix([x_u, y_u, z_u])
        r_v = Matrix([x_v, y_v, z_v])
        
        # First fundamental form coefficients
        E = r_u.dot(r_u)
        F = r_u.dot(r_v)
        G = r_v.dot(r_v)
        
        # Normal vector
        normal = r_u.cross(r_v)
        normal_magnitude = sqrt(normal.dot(normal))
        unit_normal = normal / normal_magnitude
        
        # Second derivatives
        x_uu = diff(x_u, self.u)
        y_uu = diff(y_u, self.u)
        z_uu = diff(z_u, self.u)
        
        x_uv = diff(x_u, self.v)
        y_uv = diff(y_u, self.v)
        z_uv = diff(z_u, self.v)
        
        x_vv = diff(x_v, self.v)
        y_vv = diff(y_v, self.v)
        z_vv = diff(z_v, self.v)
        
        # Second derivatives of r
        r_uu = Matrix([x_uu, y_uu, z_uu])
        r_uv = Matrix([x_uv, y_uv, z_uv])
        r_vv = Matrix([x_vv, y_vv, z_vv])
        
        # Second fundamental form coefficients
        L = unit_normal.dot(r_uu)
        M = unit_normal.dot(r_uv)
        N = unit_normal.dot(r_vv)
        
        # Compute Christoffel symbols of the first kind
        Gamma_uu_u = 0.5 * diff(E, self.u)
        Gamma_uu_v = 0.5 * diff(E, self.v)
        Gamma_uv_u = 0.5 * (diff(F, self.u) + diff(E, self.v))
        Gamma_uv_v = 0.5 * (diff(G, self.u) + diff(F, self.v))
        Gamma_vv_u = 0.5 * (diff(F, self.v) + diff(G, self.u))
        Gamma_vv_v = 0.5 * diff(G, self.v)
        
        # Compute determinant of the first fundamental form
        g_det = E*G - F**2
        
        # Compute inverse of the first fundamental form
        if g_det != 0:
            E_inv = G / g_det
            F_inv = -F / g_det
            G_inv = E / g_det
        else:
            E_inv = sp.Symbol('E_inv')
            F_inv = sp.Symbol('F_inv')
            G_inv = sp.Symbol('G_inv')
        
        # Compute Christoffel symbols of the second kind
        Gamma_uu_u_kind2 = Gamma_uu_u * E_inv + Gamma_uu_v * F_inv
        Gamma_uu_v_kind2 = Gamma_uu_u * F_inv + Gamma_uu_v * G_inv
        Gamma_uv_u_kind2 = Gamma_uv_u * E_inv + Gamma_uv_v * F_inv
        Gamma_uv_v_kind2 = Gamma_uv_u * F_inv + Gamma_uv_v * G_inv
        Gamma_vv_u_kind2 = Gamma_vv_u * E_inv + Gamma_vv_v * F_inv
        Gamma_vv_v_kind2 = Gamma_vv_u * F_inv + Gamma_vv_v * G_inv
        
        # Calculate Gaussian and mean curvature
        try:
            K = (L*N - M**2) / g_det  # Gaussian curvature
            H = (E*N - 2*F*M + G*L) / (2*g_det)  # Mean curvature
        except:
            K = sp.Symbol('K')
            H = sp.Symbol('H')
        
        # Return results
        return {
            'E': E, 'F': F, 'G': G,  # First fundamental form
            'L': L, 'M': M, 'N': N,  # Second fundamental form
            'normal': normal,
            'unit_normal': unit_normal,
            'g_det': g_det,
            'r_u': r_u, 'r_v': r_v,
            'r_uu': r_uu, 'r_uv': r_uv, 'r_vv': r_vv,
            'Gamma_uu_u': Gamma_uu_u, 'Gamma_uu_v': Gamma_uu_v,
            'Gamma_uv_u': Gamma_uv_u, 'Gamma_uv_v': Gamma_uv_v,
            'Gamma_vv_u': Gamma_vv_u, 'Gamma_vv_v': Gamma_vv_v,
            'Gamma_uu_u_kind2': Gamma_uu_u_kind2, 'Gamma_uu_v_kind2': Gamma_uu_v_kind2,
            'Gamma_uv_u_kind2': Gamma_uv_u_kind2, 'Gamma_uv_v_kind2': Gamma_uv_v_kind2,
            'Gamma_vv_u_kind2': Gamma_vv_u_kind2, 'Gamma_vv_v_kind2': Gamma_vv_v_kind2,
            'K': K, 'H': H  # Curvatures
        }
    
    def derive_geodesic_equations(self, surface, use_energy=True, symbolic=True):
        """
        Derive the geodesic equations for a surface.
        
        Parameters:
        -----------
        surface : dict
            Surface information
        use_energy : bool
            Whether to use the energy functional (True) or arc length (False)
        symbolic : bool
            Whether to return symbolic expressions or numerical functions
            
        Returns:
        --------
        dict
            Geodesic equations and related information
        """
        # Compute fundamental forms
        forms = self.compute_fundamental_forms(surface)
        
        # Extract relevant quantities
        E, F, G = forms['E'], forms['F'], forms['G']
        
        if use_energy:
            # Derive geodesic equations using the energy functional
            # E = (1/2) ∫ (E*(du/dt)^2 + 2F*(du/dt)(dv/dt) + G*(dv/dt)^2) dt
            
            # Lagrangian
            L = (E * self.udot**2 + 2*F * self.udot * self.vdot + G * self.vdot**2) / 2
            
            # Euler-Lagrange equations
            # d/dt(∂L/∂u_dot) - ∂L/∂u = 0
            # d/dt(∂L/∂v_dot) - ∂L/∂v = 0
            
            # Partial derivatives of L
            dL_dudot = diff(L, self.udot)
            dL_du = diff(L, self.u_t)
            
            dL_dvdot = diff(L, self.vdot)
            dL_dv = diff(L, self.v_t)
            
            # Time derivatives of partial derivatives
            d_dt_dL_dudot = diff(dL_dudot, self.t)
            d_dt_dL_dvdot = diff(dL_dvdot, self.t)
            
            # Euler-Lagrange equations
            eq_u = d_dt_dL_dudot - dL_du
            eq_v = d_dt_dL_dvdot - dL_dv
        else:
            # Derive geodesic equations using the arc length functional
            # L = ∫ √(E*(du/dt)^2 + 2F*(du/dt)(dv/dt) + G*(dv/dt)^2) dt
            
            # First derivatives of the fundamental form coefficients
            dE_du = diff(E, self.u)
            dE_dv = diff(E, self.v)
            dF_du = diff(F, self.u)
            dF_dv = diff(F, self.v)
            dG_du = diff(G, self.u)
            dG_dv = diff(G, self.v)
            
            # Substitute into the geodesic equations
            eq_u = (
                self.uddot + 
                (dE_du*self.udot**2)/2 + 
                dF_du*self.udot*self.vdot + 
                (dG_du*self.vdot**2)/2 - 
                (dE_dv*self.udot**2)/2 - 
                dF_dv*self.udot*self.vdot - 
                (dG_dv*self.vdot**2)/2
            )
            
            eq_v = (
                self.vddot + 
                (dE_dv*self.udot**2)/2 + 
                dF_dv*self.udot*self.vdot + 
                (dG_dv*self.vdot**2)/2 - 
                (dE_du*self.udot**2)/2 - 
                dF_du*self.udot*self.vdot - 
                (dG_du*self.vdot**2)/2
            )
        
        # Simplify equations if possible
        try:
            eq_u = sp.simplify(eq_u)
            eq_v = sp.simplify(eq_v)
        except:
            pass
        
        # Get geodesic equations in terms of Christoffel symbols
        eq_u_christoffel = (
            self.uddot + 
            forms['Gamma_uu_u_kind2'] * self.udot**2 + 
            2 * forms['Gamma_uv_u_kind2'] * self.udot * self.vdot +
            forms['Gamma_vv_u_kind2'] * self.vdot**2
        )
        
        eq_v_christoffel = (
            self.vddot + 
            forms['Gamma_uu_v_kind2'] * self.udot**2 + 
            2 * forms['Gamma_uv_v_kind2'] * self.udot * self.vdot +
            forms['Gamma_vv_v_kind2'] * self.vdot**2
        )
        
        # Function to replace symbolic derivatives with array elements for numerical solving
        def create_numerical_equations(surface):
            # Extract fundamental form coefficients as functions
            E_func = sp.lambdify((self.u, self.v), forms['E'], "numpy")
            F_func = sp.lambdify((self.u, self.v), forms['F'], "numpy")
            G_func = sp.lambdify((self.u, self.v), forms['G'], "numpy")
            
            # Derivatives of fundamental form coefficients
            dE_du_func = sp.lambdify((self.u, self.v), diff(forms['E'], self.u), "numpy")
            dE_dv_func = sp.lambdify((self.u, self.v), diff(forms['E'], self.v), "numpy")
            dF_du_func = sp.lambdify((self.u, self.v), diff(forms['F'], self.u), "numpy")
            dF_dv_func = sp.lambdify((self.u, self.v), diff(forms['F'], self.v), "numpy")
            dG_du_func = sp.lambdify((self.u, self.v), diff(forms['G'], self.u), "numpy")
            dG_dv_func = sp.lambdify((self.u, self.v), diff(forms['G'], self.v), "numpy")
            
            # Christoffel symbols
            Gamma_uu_u_func = sp.lambdify((self.u, self.v), forms['Gamma_uu_u_kind2'], "numpy")
            Gamma_uu_v_func = sp.lambdify((self.u, self.v), forms['Gamma_uu_v_kind2'], "numpy")
            Gamma_uv_u_func = sp.lambdify((self.u, self.v), forms['Gamma_uv_u_kind2'], "numpy")
            Gamma_uv_v_func = sp.lambdify((self.u, self.v), forms['Gamma_uv_v_kind2'], "numpy")
            Gamma_vv_u_func = sp.lambdify((self.u, self.v), forms['Gamma_vv_u_kind2'], "numpy")
            Gamma_vv_v_func = sp.lambdify((self.u, self.v), forms['Gamma_vv_v_kind2'], "numpy")
            
            # Define the geodesic equations as a system of first-order ODEs
            def geodesic_system(t, y):
                u, v, u_dot, v_dot = y
                
                # Use Christoffel symbols for the equations
                try:
                    Gamma_uu_u = Gamma_uu_u_func(u, v)
                    Gamma_uu_v = Gamma_uu_v_func(u, v)
                    Gamma_uv_u = Gamma_uv_u_func(u, v)
                    Gamma_uv_v = Gamma_uv_v_func(u, v)
                    Gamma_vv_u = Gamma_vv_u_func(u, v)
                    Gamma_vv_v = Gamma_vv_v_func(u, v)
                    
                    u_ddot = -(Gamma_uu_u * u_dot**2 + 2 * Gamma_uv_u * u_dot * v_dot + Gamma_vv_u * v_dot**2)
                    v_ddot = -(Gamma_uu_v * u_dot**2 + 2 * Gamma_uv_v * u_dot * v_dot + Gamma_vv_v * v_dot**2)
                except:
                    # If Christoffel symbols have singularities, use an alternative approach
                    # with the derivatives of the metric coefficients
                    E_val = E_func(u, v)
                    F_val = F_func(u, v)
                    G_val = G_func(u, v)
                    
                    dE_du = dE_du_func(u, v)
                    dE_dv = dE_dv_func(u, v)
                    dF_du = dF_du_func(u, v)
                    dF_dv = dF_dv_func(u, v)
                    dG_du = dG_du_func(u, v)
                    dG_dv = dG_dv_func(u, v)
                    
                    # Compute determinant and inverse components
                    g_det = E_val * G_val - F_val**2
                    if abs(g_det) < 1e-10:
                        # Handle near-singular metric
                        return np.array([u_dot, v_dot, 0, 0])
                    
                    E_inv = G_val / g_det
                    F_inv = -F_val / g_det
                    G_inv = E_val / g_det
                    
                    # First kind Christoffel symbols
                    Gamma_uu_u = 0.5 * dE_du
                    Gamma_uu_v = 0.5 * dE_dv
                    Gamma_uv_u = 0.5 * (dF_du + dE_dv)
                    Gamma_uv_v = 0.5 * (dG_du + dF_dv)
                    Gamma_vv_u = 0.5 * (dF_dv + dG_du)
                    Gamma_vv_v = 0.5 * dG_dv
                    
                    # Second kind Christoffel symbols
                    Gamma_uu_u_2 = Gamma_uu_u * E_inv + Gamma_uu_v * F_inv
                    Gamma_uu_v_2 = Gamma_uu_u * F_inv + Gamma_uu_v * G_inv
                    Gamma_uv_u_2 = Gamma_uv_u * E_inv + Gamma_uv_v * F_inv
                    Gamma_uv_v_2 = Gamma_uv_u * F_inv + Gamma_uv_v * G_inv
                    Gamma_vv_u_2 = Gamma_vv_u * E_inv + Gamma_vv_v * F_inv
                    Gamma_vv_v_2 = Gamma_vv_u * F_inv + Gamma_vv_v * G_inv
                    
                    u_ddot = -(Gamma_uu_u_2 * u_dot**2 + 2 * Gamma_uv_u_2 * u_dot * v_dot + Gamma_vv_u_2 * v_dot**2)
                    v_ddot = -(Gamma_uu_v_2 * u_dot**2 + 2 * Gamma_uv_v_2 * u_dot * v_dot + Gamma_vv_v_2 * v_dot**2)
                
                return np.array([u_dot, v_dot, u_ddot, v_ddot])
            
            return geodesic_system
        
        # Create numerical functions for solving the geodesic equations
        geodesic_system = create_numerical_equations(surface)
        
        # Return results
        return {
            'eq_u': eq_u,
            'eq_v': eq_v,
            'eq_u_christoffel': eq_u_christoffel,
            'eq_v_christoffel': eq_v_christoffel,
            'geodesic_system': geodesic_system
        }
    
    def solve_geodesic(self, surface, initial_conditions, t_span=(0, 10), t_eval=None):
        """
        Solve the geodesic equations numerically.
        
        Parameters:
        -----------
        surface : dict
            Surface information
        initial_conditions : tuple
            (u0, v0, u_dot0, v_dot0) - Initial position and velocity
        t_span : tuple
            (t_min, t_max) - Time interval for integration
        t_eval : array_like, optional
            Times at which to evaluate the solution
            
        Returns:
        --------
        dict
            Solution information including the geodesic curve
        """
        # Derive geodesic equations
        eqs = self.derive_geodesic_equations(surface, symbolic=False)
        geodesic_system = eqs['geodesic_system']
        
        # Solve ODE system
        sol = solve_ivp(
            geodesic_system,
            t_span,
            initial_conditions,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )
        
        # Extract solution
        t = sol.t
        u = sol.y[0]
        v = sol.y[1]
        u_dot = sol.y[2]
        v_dot = sol.y[3]
        
        # Calculate 3D coordinates
        x = surface['x_func'](u, v)
        y = surface['y_func'](u, v)
        z = surface['z_func'](u, v)
        
        # Return solution
        return {
            't': t,
            'u': u,
            'v': v,
            'u_dot': u_dot,
            'v_dot': v_dot,
            'x': x,
            'y': y,
            'z': z,
            'sol': sol
        }
    
    def plot_surface(self, surface, ax=None, density=50, alpha=0.7, cmap='viridis', 
                     show_wireframe=False, colorize_by='z'):
        """
        Plot a 3D surface.
        
        Parameters:
        -----------
        surface : dict
            Surface information
        ax : matplotlib.axes.Axes, optional
            3D Axes to plot on
        density : int
            Number of points in each dimension
        alpha : float
            Transparency of the surface
        cmap : str
            Colormap for the surface
        show_wireframe : bool
            Whether to show a wireframe
        colorize_by : str
            How to colorize the surface ('z', 'gaussian_curvature', 'mean_curvature')
            
        Returns:
        --------
        matplotlib.axes.Axes
            The 3D axes with the plot
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Extract surface information
        u_range = surface['u_range']
        v_range = surface['v_range']
        
        # Create a grid
        u_values = np.linspace(u_range[0], u_range[1], density)
        v_values = np.linspace(v_range[0], v_range[1], density)
        u_grid, v_grid = np.meshgrid(u_values, v_values)
        
        # Compute surface coordinates
        x = surface['x_func'](u_grid, v_grid)
        y = surface['y_func'](u_grid, v_grid)
        z = surface['z_func'](u_grid, v_grid)
        
        # Handle NaN values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(mask):
            print("Warning: No valid points to plot.")
            return ax
        
        # Determine colorization
        if colorize_by == 'z':
            colors = z
        elif colorize_by in ['gaussian_curvature', 'mean_curvature']:
            # Compute curvature
            forms = self.compute_fundamental_forms(surface)
            
            if colorize_by == 'gaussian_curvature':
                K_expr = forms['K']
                
                # Create a function for Gaussian curvature
                K_func = sp.lambdify((self.u, self.v), K_expr, "numpy")
                
                # Compute curvature values
                try:
                    colors = K_func(u_grid, v_grid)
                except:
                    # Fall back to z if there's an error
                    print("Warning: Could not compute Gaussian curvature. Using z-coordinate instead.")
                    colors = z
            else:  # mean_curvature
                H_expr = forms['H']
                
                # Create a function for mean curvature
                H_func = sp.lambdify((self.u, self.v), H_expr, "numpy")
                
                # Compute curvature values
                try:
                    colors = H_func(u_grid, v_grid)
                except:
                    # Fall back to z if there's an error
                    print("Warning: Could not compute mean curvature. Using z-coordinate instead.")
                    colors = z
        else:
            # Default to z
            colors = z
        
        # Plot the surface
        surf = ax.plot_surface(x, y, z, cmap=cmap, alpha=alpha, shade=True, 
                             facecolors=cm.get_cmap(cmap)(colors/np.nanmax(colors)))
        
        if show_wireframe:
            ax.plot_wireframe(x, y, z, color='black', alpha=0.2, linewidth=0.5)
        
        # Set title and labels
        ax.set_title(surface['name'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add colorbar
        if colorize_by != 'z':
            fig = ax.figure
            cb = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
            cb.set_label('Gaussian Curvature' if colorize_by == 'gaussian_curvature' else 'Mean Curvature')
        
        return ax
    
    def plot_geodesic(self, surface, geodesic, ax=None, color='red', linewidth=2, show_velocity=False):
        """
        Plot a geodesic curve on a surface.
        
        Parameters:
        -----------
        surface : dict
            Surface information
        geodesic : dict
            Geodesic curve information
        ax : matplotlib.axes.Axes, optional
            3D Axes to plot on
        color : str
            Color of the geodesic curve
        linewidth : float
            Width of the geodesic curve
        show_velocity : bool
            Whether to show velocity vectors along the curve
            
        Returns:
        --------
        matplotlib.axes.Axes
            The 3D axes with the plot
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            self.plot_surface(surface, ax=ax, alpha=0.3)
        
        # Extract geodesic coordinates
        x = geodesic['x']
        y = geodesic['y']
        z = geodesic['z']
        
        # Plot the geodesic curve
        ax.plot(x, y, z, color=color, linewidth=linewidth, label='Geodesic')
        
        # Mark start and end points
        ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start')
        ax.scatter(x[-1], y[-1], z[-1], color='purple', s=50, label='End')
        
        if show_velocity:
            # Extract velocity information
            u_dot = geodesic['u_dot']
            v_dot = geodesic['v_dot']
            u = geodesic['u']
            v = geodesic['v']
            
            # Compute tangent vectors in 3D space
            forms = self.compute_fundamental_forms(surface)
            r_u_expr = forms['r_u']
            r_v_expr = forms['r_v']
            
            # Create functions for the tangent vectors
            r_u_func_x = sp.lambdify((self.u, self.v), r_u_expr[0], "numpy")
            r_u_func_y = sp.lambdify((self.u, self.v), r_u_expr[1], "numpy")
            r_u_func_z = sp.lambdify((self.u, self.v), r_u_expr[2], "numpy")
            
            r_v_func_x = sp.lambdify((self.u, self.v), r_v_expr[0], "numpy")
            r_v_func_y = sp.lambdify((self.u, self.v), r_v_expr[1], "numpy")
            r_v_func_z = sp.lambdify((self.u, self.v), r_v_expr[2], "numpy")
            
            # Sample points along the geodesic for velocity vectors
            n_points = len(x)
            step = max(1, n_points // 20)  # Show about 20 velocity vectors
            
            for i in range(0, n_points, step):
                # Compute tangent vectors at this point
                r_u_x = r_u_func_x(u[i], v[i])
                r_u_y = r_u_func_y(u[i], v[i])
                r_u_z = r_u_func_z(u[i], v[i])
                
                r_v_x = r_v_func_x(u[i], v[i])
                r_v_y = r_v_func_y(u[i], v[i])
                r_v_z = r_v_func_z(u[i], v[i])
                
                # Compute velocity vector in 3D space
                vel_x = u_dot[i] * r_u_x + v_dot[i] * r_v_x
                vel_y = u_dot[i] * r_u_y + v_dot[i] * r_v_y
                vel_z = u_dot[i] * r_u_z + v_dot[i] * r_v_z
                
                # Normalize velocity
                vel_norm = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
                if vel_norm > 1e-10:  # Avoid division by zero
                    vel_x /= vel_norm
                    vel_y /= vel_norm
                    vel_z /= vel_norm
                
                # Scale for visualization
                scale = 0.2
                
                # Plot velocity vector
                ax.quiver(x[i], y[i], z[i], vel_x * scale, vel_y * scale, vel_z * scale,
                       color='orange', arrow_length_ratio=0.3)
        
        ax.legend()
        return ax
    
    def intrinsic_coordinates(self, surface, coordinate_type='gaussian', base_point=None, params=None):
        """
        Generate intrinsic coordinate systems on a surface.
        
        Parameters:
        -----------
        surface : dict
            Surface information
        coordinate_type : str
            Type of intrinsic coordinates ('gaussian', 'geodesic', 'isothermal', 'normal')
        base_point : tuple, optional
            Base point (u0, v0) for coordinate system
        params : dict, optional
            Additional parameters for the coordinate system
            
        Returns:
        --------
        dict
            Coordinate system information
        """
        if params is None:
            params = {}
        
        # Default base point at the center of the parameter domain
        if base_point is None:
            u_range = surface['u_range']
            v_range = surface['v_range']
            base_point = ((u_range[0] + u_range[1]) / 2, (v_range[0] + v_range[1]) / 2)
        
        u0, v0 = base_point
        
        # Compute fundamental forms at the base point
        forms = self.compute_fundamental_forms(surface)
        
        if coordinate_type == 'gaussian':
            # Gaussian (general) coordinates are just the default parametrization
            coordinate_system = {
                'name': 'Gaussian Coordinates',
                'type': 'gaussian',
                'base_point': base_point,
                'description': (
                    "Gaussian coordinates are general curvilinear coordinates on a surface. "
                    "In these coordinates, the metric tensor g_ij has components that generally "
                    "depend on position, and the coordinate curves need not be orthogonal."
                ),
                'coordinates': lambda u, v: (u, v),
                'inverse': lambda x, y: (x, y),
                'metric': lambda u, v: np.array(
                    [float(forms['E'].subs([(self.u, u), (self.v, v)])), 
                     float(forms['F'].subs([(self.u, u), (self.v, v)]))],
                    [float(forms['F'].subs([(self.u, u), (self.v, v)])), 
                     float(forms['G'].subs([(self.u, u), (self.v, v)]))])
            }
            
        elif coordinate_type == 'geodesic':
            # Geodesic coordinates, where one family of coordinate curves are geodesics
            # perpendicular to the other family
            
            # To implement this properly, we'd need to solve geodesic equations
            # This is a simplified version where we construct approximate geodesic coordinates
            
            # Compute tangent vectors at the base point
            E0 = float(forms['E'].subs([(self.u, u0), (self.v, v0)]))
            F0 = float(forms['F'].subs([(self.u, u0), (self.v, v0)]))
            G0 = float(forms['G'].subs([(self.u, u0), (self.v, v0)]))
            
            # Compute unit vectors for a local orthonormal frame
            # e1 is along the u-direction, e2 is perpendicular to e1
            e1_u = 1.0 / np.sqrt(E0)
            e1_v = 0.0
            
            e2_u = -F0 / (np.sqrt(E0) * np.sqrt(E0 * G0 - F0**2))
            e2_v = np.sqrt(E0) / np.sqrt(E0 * G0 - F0**2)
            
            # Define the coordinate transformation (approximate)
            def coordinates(u, v):
                # Compute displacement from base point
                du = u - u0
                dv = v - v0
                
                # Project onto the local frame
                x = du * e1_u + dv * e1_v
                y = du * e2_u + dv * e2_v
                
                return x, y
            
            # Define the inverse transformation (approximate)
            def inverse(x, y):
                # Compute the displacement in the original coordinates
                du = x * e1_u + y * e2_u
                dv = x * e1_v + y * e2_v
                
                # Compute the original coordinates
                u = u0 + du
                v = v0 + dv
                
                return u, v
            
            # Define the metric tensor in geodesic coordinates
            def metric(x, y):
                # Convert to original coordinates
                u, v = inverse(x, y)
                
                # Compute metric components in original coordinates
                E = float(forms['E'].subs([(self.u, u), (self.v, v)]))
                F = float(forms['F'].subs([(self.u, u), (self.v, v)]))
                G = float(forms['G'].subs([(self.u, u), (self.v, v)]))
                
                # Apply coordinate transformation to get metric in geodesic coordinates
                # This is an approximation - in true geodesic coordinates, g11 = 1 and g12 = 0
                g11 = E * e1_u**2 + 2*F * e1_u * e1_v + G * e1_v**2
                g12 = E * e1_u * e2_u + F * (e1_u * e2_v + e1_v * e2_u) + G * e1_v * e2_v
                g22 = E * e2_u**2 + 2*F * e2_u * e2_v + G * e2_v**2
                
                return np.array([[g11, g12], [g12, g22]])
            
            coordinate_system = {
                'name': 'Geodesic Coordinates',
                'type': 'geodesic',
                'base_point': base_point,
                'description': (
                    "Geodesic coordinates are constructed so that one family of coordinate curves "
                    "are geodesics that are perpendicular to the other family of coordinate curves. "
                    "At the base point, the metric has the form g11 = 1, g12 = 0."
                ),
                'coordinates': coordinates,
                'inverse': inverse,
                'metric': metric,
                'e1': (e1_u, e1_v),
                'e2': (e2_u, e2_v)
            }
            
        elif coordinate_type == 'isothermal' or coordinate_type == 'conformal':
            # Isothermal (conformal) coordinates, where the metric is conformal to the Euclidean metric
            # g_ij = λ(u,v) * δ_ij
            
            # This is a complex topic that requires solving elliptic PDEs
            # Here we'll implement a simplified version based on Gauss's approach
            
            # Define the coordinate transformation (approximate)
            def coordinates(u, v):
                # Compute fundamental form coefficients
                E = float(forms['E'].subs([(self.u, u), (self.v, v)]))
                F = float(forms['F'].subs([(self.u, u), (self.v, v)]))
                G = float(forms['G'].subs([(self.u, u), (self.v, v)]))
                
                # Compute conformal factor (approximate)
                lambda_val = np.sqrt(np.sqrt(E * G - F**2))
                
                # Define conformal coordinates
                x = lambda_val * (u - u0)
                y = lambda_val * (v - v0)
                
                return x, y
            
            # Define the inverse transformation (approximate)
            def inverse(x, y):
                # Compute original coordinates (approximate)
                u = u0 + x / np.sqrt(E0 * G0 - F0**2)
                v = v0 + y / np.sqrt(E0 * G0 - F0**2)
                
                return u, v
            
            # Define the metric tensor in isothermal coordinates
            def metric(x, y):
                # Convert to original coordinates
                u, v = inverse(x, y)
                
                # Compute conformal factor
                E = float(forms['E'].subs([(self.u, u), (self.v, v)]))
                F = float(forms['F'].subs([(self.u, u), (self.v, v)]))
                G = float(forms['G'].subs([(self.u, u), (self.v, v)]))
                lambda_val = np.sqrt(np.sqrt(E * G - F**2))
                
                # In isothermal coordinates, the metric is proportional to the identity
                return lambda_val**2 * np.eye(2)
            
            coordinate_system = {
                'name': 'Isothermal (Conformal) Coordinates',
                'type': 'isothermal',
                'base_point': base_point,
                'description': (
                    "Isothermal coordinates are a special type of coordinates where "
                    "the metric tensor is conformal to the Euclidean metric: g_ij = λ(u,v) * δ_ij. "
                    "These coordinates preserve angles and are useful in complex analysis."
                ),
                'coordinates': coordinates,
                'inverse': inverse,
                'metric': metric
            }
            
        elif coordinate_type == 'normal' or coordinate_type == 'geodesic_polar':
            # Geodesic polar (normal) coordinates, centered at the base point
            
            # In geodesic polar coordinates, the radial curves are geodesics
            # and the angular curves are equidistant from the base point
            
            # This requires solving geodesic equations
            # This is a simplified version for demonstration
            
            # Define the coordinate transformation (approximate)
            def coordinates(u, v):
                # Compute displacement from base point
                du = u - u0
                dv = v - v0
                
                # Compute distance from base point
                E = float(forms['E'].subs([(self.u, u0), (self.v, v0)]))
                F = float(forms['F'].subs([(self.u, u0), (self.v, v0)]))
                G = float(forms['G'].subs([(self.u, u0), (self.v, v0)]))
                
                # Compute Cartesian coordinates locally
                x_local = np.sqrt(E) * du
                y_local = F * du / np.sqrt(E) + np.sqrt(G - F**2 / E) * dv
                
                # Convert to polar coordinates
                r = np.sqrt(x_local**2 + y_local**2)
                theta = np.arctan2(y_local, x_local)
                
                return r, theta
            
            # Define the inverse transformation (approximate)
            def inverse(r, theta):
                # Convert to local Cartesian coordinates
                x_local = r * np.cos(theta)
                y_local = r * np.sin(theta)
                
                # Convert to original coordinates
                E = float(forms['E'].subs([(self.u, u0), (self.v, v0)]))
                F = float(forms['F'].subs([(self.u, u0), (self.v, v0)]))
                G = float(forms['G'].subs([(self.u, u0), (self.v, v0)]))
                
                du = x_local / np.sqrt(E)
                dv = (y_local - F * du / np.sqrt(E)) / np.sqrt(G - F**2 / E)
                
                u = u0 + du
                v = v0 + dv
                
                return u, v
            
            # Define the metric tensor in normal coordinates
            def metric(r, theta):
                # In normal coordinates, the metric has a specific form
                # g_rr = 1, g_rθ = 0, g_θθ = r^2 * k(r,θ)
                
                # Convert to original coordinates
                u, v = inverse(r, theta)
                
                # Compute metric components in original coordinates
                E = float(forms['E'].subs([(self.u, u), (self.v, v)]))
                F = float(forms['F'].subs([(self.u, u), (self.v, v)]))
                G = float(forms['G'].subs([(self.u, u), (self.v, v)]))
                
                # For a proper implementation, we would compute the actual metric
                # Here, we'll use an approximate form
                g_rr = 1.0
                g_rtheta = 0.0
                g_thetatheta = r**2
                
                return np.array([[g_rr, g_rtheta], [g_rtheta, g_thetatheta]])
            
            coordinate_system = {
                'name': 'Geodesic Polar (Normal) Coordinates',
                'type': 'normal',
                'base_point': base_point,
                'description': (
                    "Geodesic polar coordinates are centered at a point, with radial curves being "
                    "geodesics emanating from the center, and angular curves being equidistant from "
                    "the center. At the center, the metric has the Euclidean form, and nearby the metric "
                    "is g_rr = 1, g_rθ = 0, g_θθ = r^2 * k(r,θ)."
                ),
                'coordinates': coordinates,
                'inverse': inverse,
                'metric': metric
            }
            
        else:
            raise ValueError(f"Unknown coordinate type: {coordinate_type}")
        
        return coordinate_system
    
    def plot_coordinate_system(self, surface, coordinate_system, ax=None, density=10, 
                              color1='red', color2='blue', linewidth=1):
        """
        Plot a coordinate system on a surface.
        
        Parameters:
        -----------
        surface : dict
            Surface information
        coordinate_system : dict
            Coordinate system information
        ax : matplotlib.axes.Axes, optional
            3D Axes to plot on
        density : int
            Number of coordinate lines to show
        color1, color2 : str
            Colors for the two families of coordinate curves
        linewidth : float
            Width of coordinate lines
            
        Returns:
        --------
        matplotlib.axes.Axes
            The 3D axes with the plot
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            self.plot_surface(surface, ax=ax, alpha=0.3)
        
        # Extract coordinate system information
        base_point = coordinate_system['base_point']
        inverse = coordinate_system['inverse']
        coords_type = coordinate_system['type']
        
        # Define the range of coordinate values
        if coords_type in ['gaussian', 'geodesic']:
            # Use the surface parameter range
            u_range = surface['u_range']
            v_range = surface['v_range']
            
            # First family of coordinate curves (constant v)
            u_values = np.linspace(u_range[0], u_range[1], density)
            v_values = np.linspace(v_range[0], v_range[1], density)
            
            for v_val in v_values:
                u_curve = u_values
                v_curve = np.full_like(u_curve, v_val)
                
                # Convert to 3D coordinates
                x = surface['x_func'](u_curve, v_curve)
                y = surface['y_func'](u_curve, v_curve)
                z = surface['z_func'](u_curve, v_curve)
                
                # Plot the curve
                ax.plot(x, y, z, color=color1, linewidth=linewidth)
            
            # Second family of coordinate curves (constant u)
            for u_val in u_values:
                u_curve = np.full_like(v_values, u_val)
                v_curve = v_values
                
                # Convert to 3D coordinates
                x = surface['x_func'](u_curve, v_curve)
                y = surface['y_func'](u_curve, v_curve)
                z = surface['z_func'](u_curve, v_curve)
                
                # Plot the curve
                ax.plot(x, y, z, color=color2, linewidth=linewidth)
                
        elif coords_type == 'isothermal':
            # Define a range of isothermal coordinates
            x_range = (-1, 1)
            y_range = (-1, 1)
            
            x_values = np.linspace(x_range[0], x_range[1], density)
            y_values = np.linspace(y_range[0], y_range[1], density)
            
            # First family (constant y)
            for y_val in y_values:
                points_3d = []
                for x_val in x_values:
                    try:
                        u, v = inverse(x_val, y_val)
                        if (u >= surface['u_range'][0] and u <= surface['u_range'][1] and
                            v >= surface['v_range'][0] and v <= surface['v_range'][1]):
                            x = surface['x_func'](u, v)
                            y = surface['y_func'](u, v)
                            z = surface['z_func'](u, v)
                            points_3d.append((x, y, z))
                    except:
                        continue
                
                if points_3d:
                    points_3d = np.array(points_3d)
                    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                          color=color1, linewidth=linewidth)
            
            # Second family (constant x)
            for x_val in x_values:
                points_3d = []
                for y_val in y_values:
                    try:
                        u, v = inverse(x_val, y_val)
                        if (u >= surface['u_range'][0] and u <= surface['u_range'][1] and
                            v >= surface['v_range'][0] and v <= surface['v_range'][1]):
                            x = surface['x_func'](u, v)
                            y = surface['y_func'](u, v)
                            z = surface['z_func'](u, v)
                            points_3d.append((x, y, z))
                    except:
                        continue
                
                if points_3d:
                    points_3d = np.array(points_3d)
                    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                          color=color2, linewidth=linewidth)
                    
        elif coords_type == 'normal':
            # Define a range of polar coordinates
            r_max = 1.0
            r_values = np.linspace(0, r_max, density)
            theta_values = np.linspace(0, 2*np.pi, density)
            
            # First family (constant theta - radial geodesics)
            for theta_val in theta_values:
                points_3d = []
                for r_val in r_values:
                    try:
                        u, v = inverse(r_val, theta_val)
                        if (u >= surface['u_range'][0] and u <= surface['u_range'][1] and
                            v >= surface['v_range'][0] and v <= surface['v_range'][1]):
                            x = surface['x_func'](u, v)
                            y = surface['y_func'](u, v)
                            z = surface['z_func'](u, v)
                            points_3d.append((x, y, z))
                    except:
                        continue
                
                if points_3d:
                    points_3d = np.array(points_3d)
                    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                          color=color1, linewidth=linewidth)
            
            # Second family (constant r - curves equidistant from the base point)
            for r_val in r_values[1:]:  # Skip r=0
                points_3d = []
                for theta_val in theta_values:
                    try:
                        u, v = inverse(r_val, theta_val)
                        if (u >= surface['u_range'][0] and u <= surface['u_range'][1] and
                            v >= surface['v_range'][0] and v <= surface['v_range'][1]):
                            x = surface['x_func'](u, v)
                            y = surface['y_func'](u, v)
                            z = surface['z_func'](u, v)
                            points_3d.append((x, y, z))
                    except:
                        continue
                
                if points_3d:
                    points_3d = np.array(points_3d)
                    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                          color=color2, linewidth=linewidth)
        
        # Plot the base point
        u0, v0 = base_point
        x0 = surface['x_func'](u0, v0)
        y0 = surface['y_func'](u0, v0)
        z0 = surface['z_func'](u0, v0)
        ax.scatter([x0], [y0], [z0], color='green', s=50, label='Base Point')
        
        # Add title and legend
        ax.set_title(f"{coordinate_system['name']} on {surface['name']}")
        ax.legend()
        
        return ax
    
    def display_geodesic_equation(self, display_method='text'):
        """
        Display the general form of the geodesic equation.
        
        Parameters:
        -----------
        display_method : str
            How to display the equation ('text', 'latex')
        """
        if display_method == 'text':
            equation = (
                "Geodesic Equation (Second Order Form):\n"
                "d²u/dt² + Γ¹₁₁(du/dt)² + 2Γ¹₁₂(du/dt)(dv/dt) + Γ¹₂₂(dv/dt)² = 0\n"
                "d²v/dt² + Γ²₁₁(du/dt)² + 2Γ²₁₂(du/dt)(dv/dt) + Γ²₂₂(dv/dt)² = 0\n\n"
                "where Γⁱⱼₖ are the Christoffel symbols of the second kind:\n"
                "Γⁱⱼₖ = (1/2) gⁱˡ(∂gₗⱼ/∂xᵏ + ∂gₗₖ/∂xʲ - ∂gⱼₖ/∂xˡ)\n\n"
                "For a surface with first fundamental form:\n"
                "ds² = E du² + 2F du dv + G dv²\n\n"
                "The Christoffel symbols are:\n"
                "Γ¹₁₁ = (G·E_u - 2F·F_u + F·E_v) / (2(EG-F²))\n"
                "Γ¹₁₂ = (G·E_v - F·G_u) / (2(EG-F²))\n"
                "Γ¹₂₂ = (2G·F_v - G·G_u - F·G_v) / (2(EG-F²))\n"
                "Γ²₁₁ = (2E·F_u - E·E_v - F·E_u) / (2(EG-F²))\n"
                "Γ²₁₂ = (E·G_u - F·E_v) / (2(EG-F²))\n"
                "Γ²₂₂ = (E·G_v - 2F·F_v + F·G_u) / (2(EG-F²))\n\n"
                "where E_u = ∂E/∂u, etc."
            )
            print(equation)
        elif display_method == 'latex':
            # Define symbols for LaTeX display
            u, v, t = sp.symbols('u v t')
            u_dot = sp.Function('\\dot{u}')(t)
            v_dot = sp.Function('\\dot{v}')(t)
            u_ddot = sp.Function('\\ddot{u}')(t)
            v_ddot = sp.Function('\\ddot{v}')(t)
            
            Gamma_uu_u = sp.Symbol('\\Gamma^1_{11}')
            Gamma_uu_v = sp.Symbol('\\Gamma^2_{11}')
            Gamma_uv_u = sp.Symbol('\\Gamma^1_{12}')
            Gamma_uv_v = sp.Symbol('\\Gamma^2_{12}')
            Gamma_vv_u = sp.Symbol('\\Gamma^1_{22}')
            Gamma_vv_v = sp.Symbol('\\Gamma^2_{22}')
            
            # Geodesic equations
            eq_u = u_ddot + Gamma_uu_u * u_dot**2 + 2 * Gamma_uv_u * u_dot * v_dot + Gamma_vv_u * v_dot**2
            eq_v = v_ddot + Gamma_uu_v * u_dot**2 + 2 * Gamma_uv_v * u_dot * v_dot + Gamma_vv_v * v_dot**2
            
            # Display equation
            print("Geodesic Equation (Second Order Form):")
            display(Math(sp.latex(eq_u) + " = 0"))
            display(Math(sp.latex(eq_v) + " = 0"))
            
            print("\nwhere Γⁱⱼₖ are the Christoffel symbols of the second kind.")
            
            # Display first fundamental form
            print("\nFor a surface with first fundamental form:")
            E, F, G = sp.symbols('E F G')
            ds2 = E * sp.diff(u, t)**2 + 2*F * sp.diff(u, t) * sp.diff(v, t) + G * sp.diff(v, t)**2
            display(Math("ds^2 = " + sp.latex(ds2)))
            
            # Display Christoffel symbols
            print("\nThe Christoffel symbols are:")
            E_u, E_v = sp.symbols('E_u E_v')
            F_u, F_v = sp.symbols('F_u F_v')
            G_u, G_v = sp.symbols('G_u G_v')
            
            Gamma_uu_u_expr = sp.Symbol('\\Gamma^1_{11} = \\frac{G\\cdot E_u - 2F\\cdot F_u + F\\cdot E_v}{2(EG-F^2)}')
            display(Math(sp.latex(Gamma_uu_u_expr)))
            
            Gamma_uu_v_expr = sp.Symbol('\\Gamma^2_{11} = \\frac{2E\\cdot F_u - E\\cdot E_v - F\\cdot E_u}{2(EG-F^2)}')
            display(Math(sp.latex(Gamma_uu_v_expr)))
            
            Gamma_uv_u_expr = sp.Symbol('\\Gamma^1_{12} = \\frac{G\\cdot E_v - F\\cdot G_u}{2(EG-F^2)}')
            display(Math(sp.latex(Gamma_uv_u_expr)))
            
            Gamma_uv_v_expr = sp.Symbol('\\Gamma^2_{12} = \\frac{E\\cdot G_u - F\\cdot E_v}{2(EG-F^2)}')
            display(Math(sp.latex(Gamma_uv_v_expr)))
            
            Gamma_vv_u_expr = sp.Symbol('\\Gamma^1_{22} = \\frac{2G\\cdot F_v - G\\cdot G_u - F\\cdot G_v}{2(EG-F^2)}')
            display(Math(sp.latex(Gamma_vv_u_expr)))
            
            Gamma_vv_v_expr = sp.Symbol('\\Gamma^2_{22} = \\frac{E\\cdot G_v - 2F\\cdot F_v + F\\cdot G_u}{2(EG-F^2)}')
            display(Math(sp.latex(Gamma_vv_v_expr)))
            
            print("\nwhere E_u = ∂E/∂u, etc.")
    
    def demonstrate_geodesic_deviation(self, surface, base_point, direction, num_geodesics=5, 
                                      t_span=(0, 5), t_eval=None):
        """
        Demonstrate geodesic deviation by showing nearby geodesics.
        
        Parameters:
        -----------
        surface : dict
            Surface information
        base_point : tuple
            Initial point (u0, v0)
        direction : tuple
            Initial direction (u_dot0, v_dot0)
        num_geodesics : int
            Number of geodesics to show
        t_span : tuple
            Time span for integration
        t_eval : array_like, optional
            Times at which to evaluate the solution
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure showing the geodesics
        """
        # Create a figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        self.plot_surface(surface, ax=ax, alpha=0.3)
        
        # Extract base point and direction
        u0, v0 = base_point
        u_dot0, v_dot0 = direction
        
        # Normalize the direction
        norm = np.sqrt(u_dot0**2 + v_dot0**2)
        u_dot0 /= norm
        v_dot0 /= norm
        
        # Compute perpendicular direction
        u_perp0 = -v_dot0
        v_perp0 = u_dot0
        
        # Compute geodesics
        geodesics = []
        deviation_angles = np.linspace(-0.1, 0.1, num_geodesics)
        
        for angle in deviation_angles:
            # Compute initial direction with deviation
            u_dot = u_dot0 + angle * u_perp0
            v_dot = v_dot0 + angle * v_perp0
            
            # Normalize
            norm = np.sqrt(u_dot**2 + v_dot**2)
            u_dot /= norm
            v_dot /= norm
            
            # Initial conditions
            initial_conditions = (u0, v0, u_dot, v_dot)
            
            # Solve geodesic equation
            geodesic = self.solve_geodesic(surface, initial_conditions, t_span, t_eval)
            geodesics.append(geodesic)
            
            # Plot the geodesic
            color = plt.cm.viridis(0.1 + 0.8 * (angle + 0.1) / 0.2)  # Color based on deviation
            self.plot_geodesic(surface, geodesic, ax=ax, color=color, linewidth=1.5, show_velocity=False)
        
        # Add legend and title
        ax.set_title('Geodesic Deviation on ' + surface['name'])
        
        # Create a custom legend for the deviation angles
        from matplotlib.lines import Line2D
        legend_elements = []
        for i, angle in enumerate(deviation_angles):
            color = plt.cm.viridis(0.1 + 0.8 * (angle + 0.1) / 0.2)
            legend_elements.append(Line2D([0], [0], color=color, lw=1.5, 
                                        label=f'Deviation {angle:.3f}'))
        
        ax.legend(handles=legend_elements, loc='best')
        
        return fig
    
    def analyze_curvature_effect(self, surface, base_point, size=1.0, density=20):
        """
        Analyze and visualize the effect of curvature on geodesics.
        
        Parameters:
        -----------
        surface : dict
            Surface information
        base_point : tuple
            Base point (u0, v0)
        size : float
            Size of the geodesic circle
        density : int
            Number of geodesics to compute
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure showing the geodesics
        """
        # Create a figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        self.plot_surface(surface, ax=ax, alpha=0.3, colorize_by='gaussian_curvature')
        
        # Extract base point
        u0, v0 = base_point
        
        # Compute geodesics in different directions
        angles = np.linspace(0, 2*np.pi, density)
        geodesics = []
        
        for angle in angles:
            # Initial direction
            u_dot0 = np.cos(angle)
            v_dot0 = np.sin(angle)
            
            # Initial conditions
            initial_conditions = (u0, v0, u_dot0, v_dot0)
            
            # Solve geodesic equation
            t_span = (0, size)
            geodesic = self.solve_geodesic(surface, initial_conditions, t_span)
            geodesics.append(geodesic)
            
            # Plot the geodesic
            color = plt.cm.hsv(angle / (2*np.pi))
            self.plot_geodesic(surface, geodesic, ax=ax, color=color, linewidth=1.5, show_velocity=False)
        
        # Compute a reference Euclidean circle
        # This requires mapping from the surface to 3D space
        # For simplicity, we'll use the tangent plane at the base point
        
        # Compute the tangent vectors at the base point
        forms = self.compute_fundamental_forms(surface)
        
        # Get the normal vector
        normal = forms['unit_normal'].subs([(self.u, u0), (self.v, v0)])
        normal = np.array([float(normal[0]), float(normal[1]), float(normal[2])])
        
        # Get tangent vectors
        r_u = forms['r_u'].subs([(self.u, u0), (self.v, v0)])
        r_v = forms['r_v'].subs([(self.u, u0), (self.v, v0)])
        
        r_u = np.array([float(r_u[0]), float(r_u[1]), float(r_u[2])])
        r_v = np.array([float(r_v[0]), float(r_v[1]), float(r_v[2])])
        
        # Normalize tangent vectors
        r_u_norm = np.linalg.norm(r_u)
        r_v_norm = np.linalg.norm(r_v)
        
        if r_u_norm > 1e-10:
            r_u /= r_u_norm
        if r_v_norm > 1e-10:
            r_v /= r_v_norm
        
        # Make the second vector orthogonal to the first
        r_v = r_v - np.dot(r_v, r_u) * r_u
        r_v_norm = np.linalg.norm(r_v)
        if r_v_norm > 1e-10:
            r_v /= r_v_norm
        
        # Get the base point in 3D
        x0 = surface['x_func'](u0, v0)
        y0 = surface['y_func'](u0, v0)
        z0 = surface['z_func'](u0, v0)
        base_point_3d = np.array([x0, y0, z0])
        
        # Create a circle in the tangent plane
        circle_angles = np.linspace(0, 2*np.pi, 100)
        circle_points = []
        
        for theta in circle_angles:
            # Point on the circle in the tangent plane
            point = base_point_3d + size * (np.cos(theta) * r_u + np.sin(theta) * r_v)
            circle_points.append(point)
        
        circle_points = np.array(circle_points)
        
        # Plot the tangent plane circle
        ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
              'k--', linewidth=1, label='Euclidean Circle')
        
        # Mark the endpoint of each geodesic
        endpoints = np.array([[g['x'][-1], g['y'][-1], g['z'][-1]] for g in geodesics])
        ax.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
                c='red', s=20, label='Geodesic Endpoints')
        
        # Mark the base point
        ax.scatter([x0], [y0], [z0], color='green', s=100, label='Base Point')
        
        # Add title and legend
        ax.set_title('Effect of Curvature on Geodesics: ' + surface['name'])
        ax.legend()
        
        # Calculate Gaussian curvature at the base point
        K = forms['K'].subs([(self.u, u0), (self.v, v0)])
        K_value = float(K)
        
        # Add a text box with curvature information
        info_text = f"Gaussian Curvature at Base Point: {K_value:.3f}"
        ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        return fig

def main():
    """Run demonstrations of the differential geometry concepts."""
    # Create the differential geometry object
    dg = DifferentialGeometry()
    
    print("DIFFERENTIAL GEOMETRY AND GEODESICS")
    print("===================================")
    
    # Display the general form of the geodesic equation
    print("\n1. Geodesic Equation - Analytical Form")
    print("---------------------------------------")
    dg.display_geodesic_equation()
    
    # Create some surfaces
    print("\n2. Creating Surfaces")
    print("-------------------")
    
    # Create a sphere
    sphere = dg.create_surface('sphere', {'radius': 1.0})
    print(f"Created: {sphere['name']}")
    
    # Create a torus
    torus = dg.create_surface('torus', {'major_radius': 2.0, 'minor_radius': 0.5})
    print(f"Created: {torus['name']}")
    
    # Create a saddle
    saddle = dg.create_surface('saddle', {'a': 1.0, 'b': 1.0})
    print(f"Created: {saddle['name']}")
    
    # Computing fundamental forms
    print("\n3. Computing Fundamental Forms")
    print("-----------------------------")
    sphere_forms = dg.compute_fundamental_forms(sphere)
    print(f"First Fundamental Form for Sphere:\n  E = {sphere_forms['E']}\n  F = {sphere_forms['F']}\n  G = {sphere_forms['G']}")
    print(f"Gaussian Curvature: K = {sphere_forms['K']}")
    
    # Derive geodesic equations
    print("\n4. Deriving Geodesic Equations")
    print("-----------------------------")
    sphere_eqs = dg.derive_geodesic_equations(sphere)
    print("Geodesic equations for the sphere:")
    print(f"  {sphere_eqs['eq_u_christoffel']} = 0")
    print(f"  {sphere_eqs['eq_v_christoffel']} = 0")
    
    # Solve geodesics
    print("\n5. Solving Geodesic Equations")
    print("----------------------------")
    
    # Geodesic on a sphere (a great circle)
    initial_conditions = (np.pi/4, 0, 0, 1)  # (u0, v0, u_dot0, v_dot0)
    sphere_geodesic = dg.solve_geodesic(sphere, initial_conditions, t_span=(0, 2*np.pi))
    print(f"Solved geodesic on {sphere['name']}")
    
    # Intrinsic coordinate systems
    print("\n6. Intrinsic Coordinate Systems")
    print("------------------------------")
    
    # Create different coordinate systems
    gaussian_coords = dg.intrinsic_coordinates(sphere, 'gaussian')
    print(f"Created: {gaussian_coords['name']}")
    
    geodesic_coords = dg.intrinsic_coordinates(sphere, 'geodesic')
    print(f"Created: {geodesic_coords['name']}")
    
    isothermal_coords = dg.intrinsic_coordinates(sphere, 'isothermal')
    print(f"Created: {isothermal_coords['name']}")
    
    normal_coords = dg.intrinsic_coordinates(sphere, 'normal')
    print(f"Created: {normal_coords['name']}")
    
    # Visualizations
    print("\n7. Visualizations")
    print("---------------")
    print("Generating plots...")
    
    # Plot the sphere and a geodesic
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    dg.plot_surface(sphere, ax=ax1)
    dg.plot_geodesic(sphere, sphere_geodesic, ax=ax1)
    plt.title('Geodesic on a Sphere (Great Circle)')
    
    # Plot different coordinate systems
    fig2, axs = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': '3d'})
    axs = axs.flatten()
    
    dg.plot_coordinate_system(sphere, gaussian_coords, ax=axs[0])
    dg.plot_coordinate_system(sphere, geodesic_coords, ax=axs[1])
    dg.plot_coordinate_system(sphere, isothermal_coords, ax=axs[2])
    dg.plot_coordinate_system(sphere, normal_coords, ax=axs[3])
    
    plt.tight_layout()
    
    # Demonstrate geodesic deviation
    fig3 = dg.demonstrate_geodesic_deviation(sphere, (np.pi/4, 0), (0, 1), num_geodesics=7)
    
    # Analyze the effect of curvature
    fig4 = dg.analyze_curvature_effect(torus, (0, 0), size=0.5)
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()