import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, eigs
from scipy.linalg import null_space, svd
import networkx as nx
from sympy import symbols, diff, Matrix, simplify, solve, Poly
from sympy.abc import x, y, z, w
from itertools import combinations, product
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from tqdm import tqdm


##############################
# 1. CALABI-YAU MANIFOLDS
##############################

class CalabiYauManifold:
    """
    Class for studying the topology of Calabi-Yau manifolds.
    
    Calabi-Yau manifolds are complex Kähler manifolds with vanishing first Chern class.
    They play a central role in string theory as they allow for compactifications 
    that preserve supersymmetry in four dimensions.
    
    In string theory, we typically consider Calabi-Yau threefolds, which are 
    complex 3-dimensional (real 6-dimensional) manifolds.
    """
    
    def __init__(self, name, ambient_space=None, defining_equations=None, hodge_numbers=None):
        """
        Initialize a Calabi-Yau manifold.
        
        Parameters:
        ----------
        name : str
            Name or description of the Calabi-Yau manifold
        ambient_space : str, optional
            Description of the ambient space (e.g., 'P^4', 'P^2 × P^2')
        defining_equations : list, optional
            List of defining equations for the manifold
        hodge_numbers : dict, optional
            Dictionary of Hodge numbers {(p,q): h^{p,q}}
        """
        self.name = name
        self.ambient_space = ambient_space
        self.defining_equations = defining_equations
        self.hodge_numbers = hodge_numbers or {}
        
        # Computed properties
        self._hilbert_polynomial = None
        self._euler_characteristic = None
        self._betti_numbers = None
    
    def compute_hodge_diamond(self, dimension=3):
        """
        Compute the Hodge diamond for a Calabi-Yau manifold.
        
        For a Calabi-Yau threefold, the Hodge diamond has the form:
                        1
                      0   0
                    0   h^{1,1}  0
                  0   h^{2,1}  h^{2,1}  0
                0   h^{1,1}  0   0   0
              0   0   0   0   0   0
            1   0   0   0   0   0   1
            
        Where h^{1,1} and h^{2,1} are the independent Hodge numbers.
        
        Returns:
        -------
        dict
            Complete dictionary of Hodge numbers
        """
        # Start with known Hodge numbers
        complete_hodge = self.hodge_numbers.copy()
        
        # For Calabi-Yau manifolds of dimension n:
        # h^{0,0} = h^{n,n} = 1
        complete_hodge[(0,0)] = 1
        complete_hodge[(dimension,dimension)] = 1
        
        # h^{p,0} = h^{0,p} = 0 for 0 < p < n
        for p in range(1, dimension):
            complete_hodge[(p,0)] = 0
            complete_hodge[(0,p)] = 0
        
        # Serre duality: h^{p,q} = h^{n-p,n-q}
        for p in range(dimension+1):
            for q in range(dimension+1):
                if (p,q) in complete_hodge:
                    complete_hodge[(dimension-p,dimension-q)] = complete_hodge[(p,q)]
        
        # Set remaining Hodge numbers to 0 for a complete diamond
        for p in range(dimension+1):
            for q in range(dimension+1):
                if (p,q) not in complete_hodge:
                    complete_hodge[(p,q)] = 0
        
        self.hodge_numbers = complete_hodge
        return complete_hodge
    
    def display_hodge_diamond(self, dimension=3):
        """
        Display the Hodge diamond in a visually pleasing format.
        
        Returns:
        -------
        str
            String representation of the Hodge diamond
        """
        self.compute_hodge_diamond(dimension)
        
        diamond = ""
        for p in range(dimension + 1):
            line = " " * (dimension - p) * 4
            for q in range(dimension + 1):
                if 0 <= p <= dimension and 0 <= q <= dimension:
                    line += f"{self.hodge_numbers.get((p,q), 0):^8}"
            diamond += line + "\n"
        
        return diamond
    
    def compute_euler_characteristic(self):
        """
        Compute the Euler characteristic using the Hodge numbers.
        
        For a complex manifold, the Euler characteristic is:
        χ = ∑_{p,q} (-1)^{p+q} h^{p,q}
        
        Returns:
        -------
        int
            Euler characteristic
        """
        if not self.hodge_numbers:
            raise ValueError("Hodge numbers must be computed first")
        
        euler = 0
        for (p, q), h_pq in self.hodge_numbers.items():
            euler += (-1) ** (p + q) * h_pq
        
        self._euler_characteristic = euler
        return euler
    
    def compute_betti_numbers(self):
        """
        Compute the Betti numbers from the Hodge numbers.
        
        For a complex manifold, the Betti numbers are:
        b_k = ∑_{p+q=k} h^{p,q}
        
        Returns:
        -------
        dict
            Dictionary of Betti numbers {k: b_k}
        """
        if not self.hodge_numbers:
            raise ValueError("Hodge numbers must be computed first")
        
        betti = {}
        dimension = max(p+q for p, q in self.hodge_numbers.keys()) // 2
        
        for k in range(2*dimension + 1):
            betti[k] = sum(h_pq for (p, q), h_pq in self.hodge_numbers.items() if p + q == k)
        
        self._betti_numbers = betti
        return betti
    
    def mirror_manifold(self):
        """
        Compute the mirror Calabi-Yau manifold.
        
        Mirror symmetry is a duality in string theory that relates pairs 
        of Calabi-Yau manifolds by exchanging complex structure and Kähler
        moduli. In terms of Hodge numbers, it exchanges h^{1,1} and h^{n-1,1}.
        
        Returns:
        -------
        CalabiYauManifold
            Mirror Calabi-Yau manifold
        """
        if not self.hodge_numbers:
            raise ValueError("Hodge numbers must be computed first")
        
        # Determine the dimension
        dimension = max(p for p, _ in self.hodge_numbers.keys())
        
        # Create mirrored Hodge numbers by exchanging h^{1,1} and h^{n-1,1}
        mirror_hodge = {}
        mirror_hodge[(1,1)] = self.hodge_numbers.get((dimension-1, 1), 0)
        mirror_hodge[(dimension-1,1)] = self.hodge_numbers.get((1, 1), 0)
        
        return CalabiYauManifold(
            name=f"Mirror of {self.name}",
            ambient_space=f"Mirror of {self.ambient_space}" if self.ambient_space else None,
            hodge_numbers=mirror_hodge
        )
    
    def compute_topological_invariants(self):
        """
        Compute various topological invariants of the Calabi-Yau manifold.
        
        Returns:
        -------
        dict
            Dictionary of topological invariants
        """
        if not self.hodge_numbers:
            self.compute_hodge_diamond()
        
        dimension = max(p for p, _ in self.hodge_numbers.keys())
        
        # Compute Euler characteristic
        euler = self.compute_euler_characteristic()
        
        # Compute Betti numbers
        betti = self.compute_betti_numbers()
        
        # For threefolds, compute additional invariants
        if dimension == 3:
            h11 = self.hodge_numbers.get((1,1), 0)
            h21 = self.hodge_numbers.get((2,1), 0)
            
            # Number of Kähler moduli
            kahler_moduli = h11
            
            # Number of complex structure moduli
            complex_moduli = h21
            
            # For string theory: number of generations of particles
            generations = abs(euler) // 2
            
            return {
                "euler_characteristic": euler,
                "betti_numbers": betti,
                "h11": h11,
                "h21": h21,
                "kahler_moduli": kahler_moduli,
                "complex_moduli": complex_moduli,
                "generations": generations
            }
        else:
            return {
                "euler_characteristic": euler,
                "betti_numbers": betti
            }


class CalabiYauCatalog:
    """
    Catalog of well-known Calabi-Yau manifolds and methods for classifying them.
    """
    
    @staticmethod
    def quintic_threefold():
        """
        Create the quintic threefold, the most studied Calabi-Yau manifold.
        It is defined as a hypersurface of degree 5 in P^4.
        
        Returns:
        -------
        CalabiYauManifold
            The quintic threefold
        """
        # Define variables symbolically
        z0, z1, z2, z3, z4 = symbols('z0 z1 z2 z3 z4')
        
        # Generic quintic polynomial
        equation = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 - 5*z0*z1*z2*z3*z4
        
        # Known Hodge numbers
        hodge_numbers = {(1,1): 1, (2,1): 101}
        
        return CalabiYauManifold(
            name="Quintic threefold",
            ambient_space="P^4",
            defining_equations=[equation],
            hodge_numbers=hodge_numbers
        )
    
    @staticmethod
    def bicubic_threefold():
        """
        Create the bicubic Calabi-Yau threefold, defined as the intersection
        of two cubics in P^5.
        
        Returns:
        -------
        CalabiYauManifold
            The bicubic threefold
        """
        # Known Hodge numbers
        hodge_numbers = {(1,1): 2, (2,1): 83}
        
        return CalabiYauManifold(
            name="Bicubic threefold",
            ambient_space="P^5",
            hodge_numbers=hodge_numbers
        )
    
    @staticmethod
    def create_complete_intersection(ambient_dimension, degrees):
        """
        Create a Calabi-Yau manifold as a complete intersection in a 
        projective space.
        
        Parameters:
        ----------
        ambient_dimension : int
            Dimension of the ambient projective space
        degrees : list
            List of degrees of the defining equations
        
        Returns:
        -------
        CalabiYauManifold
            The complete intersection Calabi-Yau
        """
        # For a Calabi-Yau in P^n, the sum of degrees must equal n+1
        if sum(degrees) != ambient_dimension + 1:
            raise ValueError(f"Sum of degrees must be {ambient_dimension+1} for a Calabi-Yau in P^{ambient_dimension}")
        
        # Compute the dimension of the manifold
        manifold_dimension = ambient_dimension - len(degrees)
        
        # Create name and description
        degree_str = "×".join([str(d) for d in degrees])
        name = f"CICY[{ambient_dimension},{degree_str}]"
        ambient_space = f"P^{ambient_dimension}"
        
        return CalabiYauManifold(
            name=name,
            ambient_space=ambient_space
        )
    
    @staticmethod
    def visualize_hodge_numbers(calabi_yaus, dimension=3):
        """
        Visualize the Hodge numbers of a collection of Calabi-Yau manifolds.
        
        Parameters:
        ----------
        calabi_yaus : list
            List of CalabiYauManifold objects
        dimension : int, optional
            Dimension of the Calabi-Yau manifolds
        """
        if dimension != 3:
            raise ValueError("Only visualization for threefolds is implemented")
        
        h11 = []
        h21 = []
        labels = []
        
        for cy in calabi_yaus:
            # Ensure Hodge numbers are computed
            cy.compute_hodge_diamond(dimension)
            h11.append(cy.hodge_numbers.get((1,1), 0))
            h21.append(cy.hodge_numbers.get((2,1), 0))
            labels.append(cy.name)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(h11, h21, s=50, alpha=0.7)
        
        # Add mirror symmetry line
        max_val = max(max(h11), max(h21)) + 10
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        plt.xlabel('$h^{1,1}$ (Kähler moduli)')
        plt.ylabel('$h^{2,1}$ (Complex structure moduli)')
        plt.title('Hodge Numbers of Calabi-Yau Threefolds')
        plt.grid(True, alpha=0.3)
        
        # Annotate points
        for i, label in enumerate(labels):
            plt.annotate(label, (h11[i], h21[i]), 
                         xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def generate_kreuzer_skarke_sample(n_samples=10):
        """
        Generate a sample of Calabi-Yau manifolds from the Kreuzer-Skarke database.
        
        The Kreuzer-Skarke database contains around 500 million reflexive polytopes
        in 4D, which correspond to Calabi-Yau threefolds.
        
        Parameters:
        ----------
        n_samples : int, optional
            Number of random samples to generate
        
        Returns:
        -------
        list
            List of CalabiYauManifold objects
        """
        # This is a simplified approximation of the distribution of Hodge numbers
        # in the Kreuzer-Skarke database
        samples = []
        
        # Generate some random Hodge numbers with a distribution similar to the actual database
        for i in range(n_samples):
            # Generate h11 with a tendency toward smaller values
            h11 = int(np.random.exponential(scale=30)) + 1
            
            # Generate h21 with a tendency toward smaller values
            h21 = int(np.random.exponential(scale=30)) + 1
            
            # Create a Calabi-Yau with these Hodge numbers
            cy = CalabiYauManifold(
                name=f"KS-sample-{i}",
                hodge_numbers={(1,1): h11, (2,1): h21}
            )
            samples.append(cy)
        
        return samples


##############################
# 2. TOPOLOGICAL QUANTUM FIELD THEORY (TQFT)
##############################

class TQFT:
    """
    Base class for topological quantum field theories.
    
    A TQFT is a quantum field theory that is invariant under diffeomorphisms,
    meaning it only depends on the topology of the spacetime manifold.
    """
    
    def __init__(self, name):
        """
        Initialize a TQFT.
        
        Parameters:
        ----------
        name : str
            Name of the TQFT
        """
        self.name = name
    
    def partition_function(self, manifold):
        """
        Compute the partition function of the TQFT on a given manifold.
        
        The partition function Z(M) of a TQFT on a manifold M is a topological
        invariant of M.
        
        Parameters:
        ----------
        manifold : object
            Manifold on which to compute the partition function
        
        Returns:
        -------
        float or complex
            Value of the partition function
        """
        raise NotImplementedError("Subclass must implement partition_function")
    
    def observables(self, manifold, *ops):
        """
        Compute the expectation values of observables in the TQFT.
        
        Parameters:
        ----------
        manifold : object
            Manifold on which to compute the observables
        *ops : objects
            Operators or observables to compute
        
        Returns:
        -------
        list
            List of expectation values
        """
        raise NotImplementedError("Subclass must implement observables")


class ChernSimonsTQFT(TQFT):
    """
    Implementation of the Chern-Simons TQFT.
    
    Chern-Simons theory is a 3D TQFT that describes the topological properties
    of knots and links. It is also related to the Jones polynomial in knot theory.
    """
    
    def __init__(self, gauge_group="SU(2)", coupling=None):
        """
        Initialize Chern-Simons TQFT.
        
        Parameters:
        ----------
        gauge_group : str, optional
            Name of the gauge group
        coupling : float, optional
            Coupling constant (level) of the theory
        """
        super().__init__(f"Chern-Simons {gauge_group}")
        self.gauge_group = gauge_group
        self.coupling = coupling
        
        # Cache for computed values
        self._cached_results = {}
    
    def partition_function(self, manifold, level=None):
        """
        Compute the Chern-Simons partition function.
        
        For a 3-manifold M, the Chern-Simons partition function is related
        to topological invariants such as the Reshetikhin-Turaev invariant.
        
        Parameters:
        ----------
        manifold : object
            A 3-manifold
        level : int, optional
            Level of the Chern-Simons theory
        
        Returns:
        -------
        complex
            Partition function value (a topological invariant)
        """
        k = level if level is not None else self.coupling
        if k is None:
            raise ValueError("Coupling constant (level) must be specified")
        
        # Cache key
        cache_key = (manifold.name if hasattr(manifold, 'name') else str(manifold), k)
        if cache_key in self._cached_results:
            return self._cached_results[cache_key]
        
        # Example computation for simple cases
        if hasattr(manifold, 'euler_characteristic'):
            euler = manifold.euler_characteristic
            
            # For a 3-sphere S³
            if euler == 2:
                # Witten's formula for the partition function on S³
                result = np.sqrt(2 / (k + 2)) * np.sin(np.pi / (k + 2))
            else:
                # Placeholder for other manifolds
                result = np.exp(1j * np.pi * k * euler / 4)
        else:
            # Default approximation for general manifolds
            result = np.exp(1j * np.pi / 8)
        
        self._cached_results[cache_key] = result
        return result
    
    def wilson_loop_expectation(self, knot, representation="fundamental", level=None):
        """
        Compute the expectation value of a Wilson loop operator.
        
        In Chern-Simons theory, the expectation value of a Wilson loop
        along a knot K is related to the Jones polynomial of K.
        
        Parameters:
        ----------
        knot : object
            Description of the knot
        representation : str, optional
            Representation of the gauge group
        level : int, optional
            Level of the Chern-Simons theory
        
        Returns:
        -------
        complex
            Wilson loop expectation value (related to knot invariants)
        """
        k = level if level is not None else self.coupling
        if k is None:
            raise ValueError("Coupling constant (level) must be specified")
        
        # Cache key
        cache_key = (str(knot), representation, k)
        if cache_key in self._cached_results:
            return self._cached_results[cache_key]
        
        # Simple knot invariants (placeholder implementation)
        if hasattr(knot, 'crossing_number'):
            # For the unknot
            if knot.crossing_number == 0:
                if representation == "fundamental":
                    # Jones polynomial for the unknot is 1
                    result = 1.0
                else:
                    # Dimension of the representation
                    result = 2.0  # dim of adjoint for SU(2)
            else:
                # Simplified Jones polynomial for non-trivial knots
                q = np.exp(2 * np.pi * 1j / (k + 2))
                result = (q + 1/q) * (-1) ** knot.crossing_number
        else:
            # Default value for unknown knots
            result = 1.0
        
        self._cached_results[cache_key] = result
        return result
    
    def observables(self, manifold, *knots):
        """
        Compute observables in Chern-Simons theory.
        
        Parameters:
        ----------
        manifold : object
            3-manifold
        *knots : objects
            Knots or links in the manifold
        
        Returns:
        -------
        list
            List of Wilson loop expectation values
        """
        return [self.wilson_loop_expectation(knot) for knot in knots]


class WittenTQFT(TQFT):
    """
    Implementation of the Witten-type TQFT used in string theory.
    
    These TQFTs arise from the topological twisting of supersymmetric theories
    and are related to the topological string theory.
    """
    
    def __init__(self, twist_type="A"):
        """
        Initialize a Witten-type TQFT.
        
        Parameters:
        ----------
        twist_type : str, optional
            Type of topological twisting ("A" or "B")
        """
        super().__init__(f"Witten {twist_type}-model")
        self.twist_type = twist_type
        
        if twist_type not in ["A", "B"]:
            raise ValueError("Twist type must be either 'A' or 'B'")
    
    def partition_function(self, calabi_yau):
        """
        Compute the partition function of the topological string on a Calabi-Yau.
        
        Parameters:
        ----------
        calabi_yau : CalabiYauManifold
            The Calabi-Yau manifold
        
        Returns:
        -------
        complex
            Partition function value
        """
        # Ensure Hodge numbers are computed
        if not calabi_yau.hodge_numbers:
            calabi_yau.compute_hodge_diamond()
        
        # The partition function depends on the type of twist
        if self.twist_type == "A":
            # A-model depends on Kähler moduli (h^{1,1})
            h11 = calabi_yau.hodge_numbers.get((1,1), 0)
            return np.exp(-h11 / 12)
        else:  # B-model
            # B-model depends on complex structure moduli (h^{2,1})
            h21 = calabi_yau.hodge_numbers.get((2,1), 0)
            return np.exp(-h21 / 12)
    
    def compute_gromov_witten_invariants(self, calabi_yau, degree_max=3):
        """
        Compute Gromov-Witten invariants for a Calabi-Yau threefold.
        
        Gromov-Witten invariants count holomorphic maps from Riemann surfaces
        to the Calabi-Yau and are important in topological string theory.
        
        Parameters:
        ----------
        calabi_yau : CalabiYauManifold
            The Calabi-Yau manifold
        degree_max : int, optional
            Maximum degree for the invariants
        
        Returns:
        -------
        dict
            Dictionary of Gromov-Witten invariants
        """
        # This is a simplified model for illustrative purposes
        # Real computation requires more sophisticated methods
        
        if not calabi_yau.hodge_numbers:
            calabi_yau.compute_hodge_diamond()
        
        h11 = calabi_yau.hodge_numbers.get((1,1), 0)
        
        # Example: the GW invariants for the quintic threefold
        # are known for small degrees
        gw_invariants = {}
        
        if calabi_yau.name == "Quintic threefold":
            # Known values for the quintic
            gw_invariants = {
                1: 2875,
                2: 609250,
                3: 317206375
            }
        else:
            # Approximate values for other Calabi-Yau manifolds
            for d in range(1, degree_max + 1):
                # This is not accurate but illustrates the concept
                gw_invariants[d] = int(h11 * (5**d) / d**3)
        
        return gw_invariants
    
    def observables(self, calabi_yau, *ops):
        """
        Compute observables in the topological string theory.
        
        Parameters:
        ----------
        calabi_yau : CalabiYauManifold
            The Calabi-Yau manifold
        *ops : objects
            Operators or observables to compute
        
        Returns:
        -------
        list
            List of expectation values
        """
        # Placeholder for computing topological string observables
        return [1.0 for _ in ops]


class ReshetikhinTuraevTQFT(TQFT):
    """
    Implementation of the Reshetikhin-Turaev TQFT.
    
    This TQFT is based on quantum groups and provides invariants of
    3-manifolds and links within them.
    """
    
    def __init__(self, quantum_group="U_q(sl_2)"):
        """
        Initialize Reshetikhin-Turaev TQFT.
        
        Parameters:
        ----------
        quantum_group : str, optional
            Quantum group used in the construction
        """
        super().__init__(f"Reshetikhin-Turaev ({quantum_group})")
        self.quantum_group = quantum_group
    
    def partition_function(self, manifold, level=3):
        """
        Compute the Reshetikhin-Turaev invariant of a 3-manifold.
        
        Parameters:
        ----------
        manifold : object
            A 3-manifold
        level : int, optional
            Level parameter for the quantum group
        
        Returns:
        -------
        complex
            The Reshetikhin-Turaev invariant
        """
        # This is a simplified implementation
        q = np.exp(np.pi * 1j / level)
        
        # For S³, the invariant is 1/√(2sin(π/r))
        # For other manifolds, we need surgery presentation
        if hasattr(manifold, 'euler_characteristic') and manifold.euler_characteristic == 2:
            return 1 / np.sqrt(2 * np.sin(np.pi / level))
        else:
            # Placeholder for other manifolds
            return np.exp(1j * np.pi / 4)
    
    def compute_invariant_of_link(self, link, level=3):
        """
        Compute the quantum invariant of a link using the Reshetikhin-Turaev construction.
        
        Parameters:
        ----------
        link : object
            A link in a 3-manifold
        level : int, optional
            Level parameter for the quantum group
        
        Returns:
        -------
        complex
            The quantum invariant of the link
        """
        # Simplified implementation for illustration
        q = np.exp(np.pi * 1j / level)
        
        # For the unknot in the fundamental representation, the invariant is (q + q⁻¹)
        if hasattr(link, 'components'):
            # Multi-component link
            result = (q + 1/q) ** len(link.components)
        else:
            # Single knot
            result = q + 1/q
        
        return result
    
    def observables(self, manifold, *links):
        """
        Compute observables in the Reshetikhin-Turaev TQFT.
        
        Parameters:
        ----------
        manifold : object
            A 3-manifold
        *links : objects
            Links in the manifold
        
        Returns:
        -------
        list
            List of quantum invariants of the links
        """
        return [self.compute_invariant_of_link(link) for link in links]


##############################
# 3. COSMOLOGICAL TOPOLOGY
##############################

class CosmologicalModel:
    """
    Base class for cosmological models describing the overall shape and topology of the universe.
    """
    
    def __init__(self, name, spatial_curvature=0):
        """
        Initialize a cosmological model.
        
        Parameters:
        ----------
        name : str
            Name of the cosmological model
        spatial_curvature : float, optional
            Spatial curvature parameter (Ω_k)
        """
        self.name = name
        self.spatial_curvature = spatial_curvature
        
        # Parameters from observational cosmology
        self.omega_matter = 0.3  # Matter density parameter
        self.omega_lambda = 0.7  # Dark energy density parameter
        self.hubble_parameter = 70.0  # km/s/Mpc
        
        # Computed properties
        self._scale_factor = None
        self._topology = None
    
    def scale_factor(self, t):
        """
        Compute the scale factor a(t) of the universe at time t.
        
        The scale factor describes how distances in the universe change with time.
        
        Parameters:
        ----------
        t : float or array
            Time(s) at which to compute the scale factor
        
        Returns:
        -------
        float or array
            Scale factor at the given time(s)
        """
        raise NotImplementedError("Subclass must implement scale_factor")
    
    def friedmann_equation(self, a):
        """
        Evaluate the Friedmann equation for a given scale factor.
        
        The Friedmann equation relates the expansion rate to the energy content.
        
        Parameters:
        ----------
        a : float or array
            Scale factor(s)
        
        Returns:
        -------
        float or array
            Value of the Friedmann equation
        """
        H0 = self.hubble_parameter / 100  # Dimensionless Hubble parameter
        
        # Friedmann equation: (H/H0)² = Ω_m/a³ + Ω_Λ + Ω_k/a²
        return H0**2 * (self.omega_matter / a**3 + self.omega_lambda + self.spatial_curvature / a**2)
    
    def topology(self):
        """
        Determine the possible topology of the universe based on the model.
        
        Returns:
        -------
        str
            Description of the topology
        """
        if self._topology is not None:
            return self._topology
        
        # Topology depends on the spatial curvature
        if abs(self.spatial_curvature) < 1e-5:
            # Flat universe - could be R³ or a 3-torus, etc.
            self._topology = "Flat (R³ or compact flat manifold)"
        elif self.spatial_curvature < 0:
            # Open universe - could be hyperbolic
            self._topology = "Open (hyperbolic or compact hyperbolic manifold)"
        else:
            # Closed universe - could be a 3-sphere or other compact manifolds
            self._topology = "Closed (S³ or other spherical space form)"
        
        return self._topology
    
    def visualize_expansion(self, t_min=0, t_max=20, n_points=100):
        """
        Visualize the expansion history of the universe.
        
        Parameters:
        ----------
        t_min, t_max : float, optional
            Time range to visualize (in Gyr)
        n_points : int, optional
            Number of points for plotting
        """
        times = np.linspace(t_min, t_max, n_points)
        scale_factors = np.array([self.scale_factor(t) for t in times])
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, scale_factors)
        plt.xlabel('Time (Gyr)')
        plt.ylabel('Scale Factor a(t)')
        plt.title(f'Expansion History - {self.name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class FLRWModel(CosmologicalModel):
    """
    Implementation of the Friedmann-Lemaître-Robertson-Walker (FLRW) cosmological model.
    
    The FLRW model is a solution to Einstein's field equations that describes an 
    expanding, homogeneous, and isotropic universe.
    """
    
    def __init__(self, omega_matter=0.3, omega_lambda=0.7, omega_radiation=0.0):
        """
        Initialize a FLRW model.
        
        Parameters:
        ----------
        omega_matter : float, optional
            Matter density parameter
        omega_lambda : float, optional
            Dark energy density parameter
        omega_radiation : float, optional
            Radiation density parameter
        """
        # Calculate spatial curvature from the density parameters
        omega_k = 1.0 - omega_matter - omega_lambda - omega_radiation
        
        super().__init__("FLRW Model", omega_k)
        self.omega_matter = omega_matter
        self.omega_lambda = omega_lambda
        self.omega_radiation = omega_radiation
    
    def scale_factor(self, t):
        """
        Compute the scale factor a(t) for the FLRW model.
        
        For a ΛCDM universe with matter and dark energy, this requires
        numerical integration of the Friedmann equation.
        
        Parameters:
        ----------
        t : float or array
            Time(s) in Gyr
        
        Returns:
        -------
        float or array
            Scale factor at the given time(s)
        """
        if np.isscalar(t):
            # Convert time to dimensionless units
            H0 = self.hubble_parameter
            t_H0 = t * 1e9 * H0 / (978 * 1e9)  # Convert Gyr to 1/H0
            
            # For a flat universe with matter and dark energy,
            # there's an analytical solution
            if abs(self.spatial_curvature) < 1e-5 and self.omega_radiation < 1e-5:
                a_t = (self.omega_matter / self.omega_lambda) ** (1/3) * \
                      np.sinh(1.5 * np.sqrt(self.omega_lambda) * t_H0) ** (2/3)
                return a_t
            else:
                # For other cases, we would use numerical integration
                # This is a simplified approximation
                a_t = (1 + self.hubble_parameter * t / 3e3) ** (2/3)
                return a_t
        else:
            # Handle array input recursively
            return np.array([self.scale_factor(t_i) for t_i in t])
    
    def cosmic_topology(self):
        """
        Analyze the possible cosmic topology based on the FLRW parameters.
        
        Returns:
        -------
        dict
            Information about the possible cosmic topology
        """
        # Basic topology based on spatial curvature
        base_topology = self.topology()
        
        # More detailed analysis
        if abs(self.spatial_curvature) < 1e-5:
            # Flat universe with different possible topologies
            possible_topologies = [
                "R³ (infinite flat space)",
                "T³ (3-torus)",
                "T² × R (cylinder)",
                "Klein bottle × R",
                "Half-turn space",
                "Quarter-turn space",
                "Third-turn space",
                "Sixth-turn space",
                "Hantzsche-Wendt space"
            ]
        elif self.spatial_curvature < 0:
            # Hyperbolic universe with different possible topologies
            possible_topologies = [
                "H³ (infinite hyperbolic space)",
                "Compact hyperbolic manifolds (there are infinitely many)"
            ]
        else:
            # Spherical universe with different possible topologies
            possible_topologies = [
                "S³ (3-sphere)",
                "RP³ (real projective space)",
                "L(p,q) (lens spaces)",
                "Poincaré dodecahedral space",
                "Quaternionic space",
                "Octahedral space",
                "Truncated cube space"
            ]
        
        return {
            "base_topology": base_topology,
            "possible_topologies": possible_topologies
        }
    
    def cosmic_horizon(self, t=13.8):
        """
        Compute the cosmic particle horizon at time t.
        
        The particle horizon is the maximum distance from which light could
        have reached us since the beginning of the universe.
        
        Parameters:
        ----------
        t : float, optional
            Time in Gyr (default is the current age of the universe)
        
        Returns:
        -------
        float
            Particle horizon in Gpc
        """
        # Simplified calculation for illustration
        c = 299792.458  # Speed of light in km/s
        H0 = self.hubble_parameter  # km/s/Mpc
        
        # For a flat universe dominated by matter
        if abs(self.spatial_curvature) < 1e-5 and self.omega_lambda < 1e-5:
            horizon = 3 * c * t * 1e9 / (1e9 * H0)  # Mpc
        else:
            # More general case (approximate)
            horizon = 3000 * c / H0  # Mpc
        
        return horizon / 1000  # Gpc


class CosmicTopology:
    """
    Class for analyzing and visualizing cosmic topology.
    """
    
    @staticmethod
    def analyze_cmb_for_topology(omega_matter=0.3, omega_lambda=0.7):
        """
        Analyze the constraints on cosmic topology from CMB observations.
        
        Parameters:
        ----------
        omega_matter : float, optional
            Matter density parameter
        omega_lambda : float, optional
            Dark energy density parameter
        
        Returns:
        -------
        dict
            Information about the constraints on cosmic topology
        """
        # Calculate the curvature
        omega_k = 1.0 - omega_matter - omega_lambda
        
        # Size of the observable universe (in Gpc)
        observable_universe_radius = 14.0  # Approximately
        
        # CMB constraints on the size of the fundamental domain
        if abs(omega_k) < 1e-2:
            # Flat universe
            min_domain_size = 20.0  # Minimum size in Gpc based on CMB data
            conclusion = "No evidence for compact topology has been found in CMB data."
        elif omega_k < 0:
            # Hyperbolic universe
            min_domain_size = 15.0
            conclusion = "Hyperbolic compact topologies are less constrained by CMB data."
        else:
            # Spherical universe
            min_domain_size = 25.0
            conclusion = "Spherical compact topologies are strongly constrained by CMB data."
        
        return {
            "omega_k": omega_k,
            "observable_universe_radius": observable_universe_radius,
            "min_domain_size": min_domain_size,
            "conclusion": conclusion
        }
    
    @staticmethod
    def visualize_topology_implications():
        """
        Visualize the implications of different cosmic topologies on observations.
        """
        # Create a figure for visualization
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Flat torus topology (T^3) - repeated patterns
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        # Create a 3D grid of points
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        z = np.linspace(-1, 1, 5)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Plot points
        ax1.scatter(X, Y, Z, c='blue', alpha=0.5)
        ax1.set_title('Toroidal Universe (T³)\nRepeated Patterns')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 2. Cosmic microwave background for different topologies
        ax2 = fig.add_subplot(2, 2, 2)
        
        # Generate a simulated CMB-like pattern
        nx, ny = 500, 500
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Standard CMB (infinite flat space)
        Z = np.sin(10*X) * np.cos(10*Y) + np.random.randn(nx, ny) * 0.2
        mask = R <= 1.0
        Z = np.where(mask, Z, np.nan)
        
        # Plot CMB-like pattern
        im = ax2.imshow(Z, cmap='coolwarm', extent=[-1, 1, -1, 1])
        ax2.set_title('CMB in Infinite Flat Space\nNo Repeated Patterns')
        plt.colorbar(im, ax=ax2)
        
        # 3. CMB in a toroidal universe (with matching circles)
        ax3 = fig.add_subplot(2, 2, 3)
        
        # Generate a simulated CMB with matching patterns
        Z_torus = np.zeros((nx, ny))
        for i in range(3):
            for j in range(3):
                # Create a repeated pattern with some randomness
                pattern = np.sin(10*(X + 2*i)) * np.cos(10*(Y + 2*j)) + np.random.randn(nx, ny) * 0.1
                # Fade with distance from center
                weight = np.exp(-((i-1)**2 + (j-1)**2) / 1.0)
                Z_torus += pattern * weight
        
        mask = R <= 1.0
        Z_torus = np.where(mask, Z_torus, np.nan)
        
        # Plot toroidal CMB-like pattern
        im = ax3.imshow(Z_torus, cmap='coolwarm', extent=[-1, 1, -1, 1])
        ax3.set_title('CMB in Toroidal Universe\nWith Matching Patterns')
        plt.colorbar(im, ax=ax3)
        
        # 4. Visualization of cosmic horizons
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Create a circle representing the observable universe
        theta = np.linspace(0, 2*np.pi, 100)
        r_observable = 1.0
        x_obs = r_observable * np.cos(theta)
        y_obs = r_observable * np.sin(theta)
        
        ax4.plot(x_obs, y_obs, 'b-', label='Observable Universe')
        
        # Create circles representing different topological scales
        r_topology_small = 0.6
        x_top_small = r_topology_small * np.cos(theta)
        y_top_small = r_topology_small * np.sin(theta)
        
        r_topology_large = 1.5
        x_top_large = r_topology_large * np.cos(theta)
        y_top_large = r_topology_large * np.sin(theta)
        
        ax4.plot(x_top_small, y_top_small, 'r--', label='Detectable Topology')
        ax4.plot(x_top_large, y_top_large, 'g-.', label='Undetectable Topology')
        
        ax4.set_title('Cosmic Horizons and Topology Scales')
        ax4.legend()
        ax4.set_aspect('equal')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def cosmic_topology_summary():
        """
        Provide a summary of the current understanding of cosmic topology.
        
        Returns:
        -------
        dict
            Summary of cosmic topology
        """
        return {
            "current_constraints": """
                Current observations from the Cosmic Microwave Background (CMB) and Large Scale Structure (LSS)
                of the universe strongly suggest that the observable universe is nearly flat (Ω_k ≈ 0).
                This means that the three-dimensional space of our universe is well described by Euclidean geometry.
                
                No definitive evidence has been found for a non-trivial topology (i.e., a compact universe
                smaller than the observable universe). The CMB data from the Planck satellite puts a lower
                bound on the size of the fundamental domain of a compact topology to be at least 1.9 times
                the diameter of the observable universe.
            """,
            
            "possible_topologies": {
                "flat": [
                    "R³ (infinite flat space)",
                    "T³ (3-torus)",
                    "Other compact flat manifolds (17 possibilities)"
                ],
                "positive_curvature": [
                    "S³ (3-sphere)",
                    "RP³ (real projective space)",
                    "Lens spaces and other spherical space forms"
                ],
                "negative_curvature": [
                    "H³ (infinite hyperbolic space)",
                    "Compact hyperbolic manifolds (infinitely many)"
                ]
            },
            
            "observational_signatures": """
                If the universe has a compact topology with a fundamental domain smaller than the observable
                universe, we would expect to see:
                
                1. Matched circles in the CMB: the same patterns of temperature fluctuations appearing in
                   multiple directions in the sky.
                
                2. Suppression of low-multipole modes in the CMB power spectrum.
                
                3. Periodic patterns in the large-scale structure of the universe.
                
                So far, none of these signatures have been convincingly detected.
            """,
            
            "future_prospects": """
                Future observations with more sensitive CMB experiments and larger galaxy surveys may provide
                better constraints on the topology of the universe. Gravitational wave astronomy might also
                provide new ways to probe cosmic topology through the detection of topological defects or
                the stochastic gravitational wave background.
            """
        }


# Example usage and visualization

if __name__ == "__main__":
    # 1. Calabi-Yau Manifolds in String Theory
    print("Creating Calabi-Yau manifolds...")
    
    # Create the quintic threefold
    quintic = CalabiYauCatalog.quintic_threefold()
    
    # Compute and display Hodge numbers
    print("\nHodge diamond for the quintic threefold:")
    print(quintic.display_hodge_diamond())
    
    # Compute topological invariants
    invariants = quintic.compute_topological_invariants()
    print("\nTopological invariants of the quintic:")
    for key, value in invariants.items():
        print(f"  {key}: {value}")
    
    # Create mirror manifold
    mirror_quintic = quintic.mirror_manifold()
    print("\nHodge diamond for the mirror quintic:")
    print(mirror_quintic.display_hodge_diamond())
    
    # Generate a sample of Calabi-Yau manifolds
    print("\nGenerating sample of Calabi-Yau manifolds...")
    cy_sample = [quintic, mirror_quintic]
    cy_sample.extend(CalabiYauCatalog.generate_kreuzer_skarke_sample(5))
    
    # Visualize Hodge numbers
    print("Visualizing Hodge numbers...")
    CalabiYauCatalog.visualize_hodge_numbers(cy_sample)
    
    # 2. Topological Quantum Field Theory
    print("\nCreating TQFTs...")
    
    # Chern-Simons theory
    cs = ChernSimonsTQFT(coupling=3)
    
    # Example knot
    class Knot:
        def __init__(self, name, crossing_number):
            self.name = name
            self.crossing_number = crossing_number
    
    # Create some knots
    unknot = Knot("Unknot", 0)
    trefoil = Knot("Trefoil", 3)
    
    # Compute Wilson loop expectation values
    print("\nWilson loop expectation values:")
    print(f"  Unknot: {cs.wilson_loop_expectation(unknot)}")
    print(f"  Trefoil: {cs.wilson_loop_expectation(trefoil)}")
    
    # Witten TQFT
    witten_a = WittenTQFT(twist_type="A")
    witten_b = WittenTQFT(twist_type="B")
    
    # Compute partition functions
    print("\nTopological string partition functions:")
    print(f"  A-model on quintic: {witten_a.partition_function(quintic)}")
    print(f"  B-model on quintic: {witten_b.partition_function(quintic)}")
    
    # Compute Gromov-Witten invariants
    gw_invariants = witten_a.compute_gromov_witten_invariants(quintic)
    print("\nGromov-Witten invariants for the quintic:")
    for degree, count in gw_invariants.items():
        print(f"  Degree {degree}: {count}")
    
    # 3. Cosmological Topology
    print("\nCosmological models and cosmic topology...")
    
    # Create FLRW model
    flrw = FLRWModel(omega_matter=0.3, omega_lambda=0.7)
    
    # Analyze cosmic topology
    topology_info = flrw.cosmic_topology()
    print("\nCosmic topology based on FLRW parameters:")
    print(f"  Base topology: {topology_info['base_topology']}")
    print("  Possible topologies:")
    for topology in topology_info['possible_topologies']:
        print(f"    - {topology}")
    
    # Visualize expansion history
    print("\nVisualizing universe expansion...")
    flrw.visualize_expansion()
    
    # Analyze CMB constraints on topology
    cmb_analysis = CosmicTopology.analyze_cmb_for_topology()
    print("\nCMB constraints on cosmic topology:")
    print(f"  Curvature parameter: {cmb_analysis['omega_k']}")
    print(f"  Observable universe radius: {cmb_analysis['observable_universe_radius']} Gpc")
    print(f"  Minimum size of fundamental domain: {cmb_analysis['min_domain_size']} Gpc")
    print(f"  Conclusion: {cmb_analysis['conclusion']}")
    
    # Visualize topology implications
    print("\nVisualizing implications of different cosmic topologies...")
    CosmicTopology.visualize_topology_implications()
    
    # Summary of cosmic topology
    topology_summary = CosmicTopology.cosmic_topology_summary()
    print("\nSummary of our current understanding of cosmic topology:")
    print(topology_summary["current_constraints"])
    
    print("\nPossible topologies for the universe:")
    for curvature, topologies in topology_summary["possible_topologies"].items():
        print(f"  {curvature.replace('_', ' ').title()} universe:")
        for topology in topologies:
            print(f"    - {topology}")
    
    print("\nObservational signatures of non-trivial topology:")
    print(topology_summary["observational_signatures"])
    
    print("\nFuture prospects for detecting cosmic topology:")
    print(topology_summary["future_prospects"])