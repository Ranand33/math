import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import null_space, det, eigvals
from sympy import symbols, Matrix, diff, simplify, solve, Poly, latex
from itertools import combinations, product, permutations
import networkx as nx
from collections import defaultdict, deque
import random
from tqdm import tqdm

#############################################
# PART 1: SMOOTH MANIFOLD FOUNDATIONS
#############################################

class SmoothManifold:
    """
    Base class for smooth manifolds with classification capabilities.
    
    A smooth manifold is a topological manifold with a smooth structure, 
    which allows for calculus to be performed.
    """
    
    def __init__(self, dimension, name=None):
        """
        Initialize a smooth manifold.
        
        Parameters:
        ----------
        dimension : int
            The dimension of the manifold
        name : str, optional
            Name or description of the manifold
        """
        self.dimension = dimension
        self.name = name or f"{dimension}-manifold"
        self.charts = []
        self.transition_maps = {}
        self.tangent_bundle = None
        self.boundary = None
        
        # Topological invariants
        self._euler_characteristic = None
        self._homology = None
        self._signature = None
        self._characteristic_classes = {}
        self._additional_invariants = {}
    
    def add_chart(self, domain, coordinates, inverse=None):
        """
        Add a coordinate chart to the manifold.
        
        Parameters:
        ----------
        domain : object
            Domain of the chart in the manifold
        coordinates : callable
            Coordinate function mapping points to R^n
        inverse : callable, optional
            Inverse function mapping R^n back to the manifold
            
        Returns:
        -------
        int
            Index of the added chart
        """
        chart_id = len(self.charts)
        self.charts.append({
            'id': chart_id,
            'domain': domain,
            'coordinates': coordinates,
            'inverse': inverse
        })
        return chart_id
    
    def add_transition_map(self, chart_id1, chart_id2, transition_map):
        """
        Add a transition map between two charts.
        
        Parameters:
        ----------
        chart_id1, chart_id2 : int
            Indices of the charts
        transition_map : callable
            Smooth function mapping between chart coordinates
        """
        self.transition_maps[(chart_id1, chart_id2)] = transition_map
    
    def is_orientable(self):
        """
        Determine if the manifold is orientable.
        
        Returns:
        -------
        bool
            True if the manifold is orientable, False otherwise
        """
        # This is a placeholder; actual implementation would depend on the manifold
        # For known manifolds, we could return the correct result
        if hasattr(self, '_orientable'):
            return self._orientable
        
        # For general manifolds, would need to check if transition maps preserve orientation
        # Simplified version: check if we know this is a non-orientable manifold
        non_orientable = ['Möbius strip', 'Klein bottle', 'Real projective plane', 'RP^2']
        if any(name in self.name for name in non_orientable):
            self._orientable = False
            return False
        
        # Default to orientable for most standard manifolds
        self._orientable = True
        return True
    
    def euler_characteristic(self):
        """
        Compute the Euler characteristic of the manifold.
        
        The Euler characteristic is a topological invariant defined as the alternating
        sum of the number of k-cells in a cell decomposition of the manifold.
        
        Returns:
        -------
        int
            The Euler characteristic
        """
        if self._euler_characteristic is not None:
            return self._euler_characteristic
        
        # For common manifolds, return known values
        if 'sphere' in self.name.lower() or 'S^' in self.name:
            # Euler characteristic of S^n is 2 for even n, 0 for odd n
            self._euler_characteristic = 2 if self.dimension % 2 == 0 else 0
        elif 'torus' in self.name.lower() or 'T^' in self.name:
            # Euler characteristic of T^n is 0
            self._euler_characteristic = 0
        elif 'projective' in self.name.lower() or 'RP^' in self.name:
            # Euler characteristic of RP^n is 1 for even n, 0 for odd n
            self._euler_characteristic = 1 if self.dimension % 2 == 0 else 0
        elif 'CP^' in self.name:
            # Euler characteristic of CP^n is n+1
            n = int(self.name.split('^')[1]) if '^' in self.name else self.dimension // 2
            self._euler_characteristic = n + 1
        else:
            # For a general manifold, would compute from a cell decomposition
            self._euler_characteristic = None
        
        return self._euler_characteristic
    
    def compute_homology(self):
        """
        Compute the homology groups of the manifold.
        
        Homology groups are algebraic invariants that detect 'holes' of different dimensions.
        
        Returns:
        -------
        dict
            Dictionary of homology groups {dimension: (rank, torsion)}
        """
        if self._homology is not None:
            return self._homology
        
        # For some common manifolds, return known homology
        homology = {}
        
        if 'sphere' in self.name.lower() or 'S^' in self.name:
            n = int(self.name.split('^')[1]) if '^' in self.name else self.dimension
            # Homology of sphere S^n: H_0 = Z, H_n = Z, H_k = 0 for 0 < k < n
            homology[0] = (1, [])  # H_0 = Z (rank 1, no torsion)
            homology[n] = (1, [])  # H_n = Z
            for k in range(1, n):
                homology[k] = (0, [])  # H_k = 0
                
        elif 'torus' in self.name.lower() or 'T^' in self.name:
            n = int(self.name.split('^')[1]) if '^' in self.name else self.dimension
            # Homology of n-torus: H_k = Z^(n choose k)
            for k in range(n+1):
                homology[k] = (self._binomial(n, k), [])
                
        elif 'projective' in self.name.lower() or 'RP^' in self.name:
            n = int(self.name.split('^')[1]) if '^' in self.name else self.dimension
            # Homology of RP^n
            homology[0] = (1, [])  # H_0 = Z
            for k in range(1, n):
                if k % 2 == 0:
                    homology[k] = (0, [])  # H_k = 0 for even k < n
                else:
                    homology[k] = (0, [2])  # H_k = Z_2 for odd k < n
            if n % 2 == 0:
                homology[n] = (0, [])  # H_n = 0 for even n
            else:
                homology[n] = (1, [])  # H_n = Z for odd n
                
        elif 'CP^' in self.name:
            n = int(self.name.split('^')[1]) if '^' in self.name else self.dimension // 2
            # Homology of CP^n: H_2k = Z for 0 ≤ k ≤ n, H_odd = 0
            for k in range(n+1):
                homology[2*k] = (1, [])  # H_2k = Z
                if 2*k+1 <= 2*n:
                    homology[2*k+1] = (0, [])  # H_2k+1 = 0
        
        else:
            # For general manifolds, this would use cellular homology algorithms
            homology = None
        
        self._homology = homology
        return homology

    def _binomial(self, n, k):
        """Helper method to compute binomial coefficients."""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        return self._binomial(n-1, k-1) + self._binomial(n-1, k)
    
    def signature(self):
        """
        Compute the signature of the manifold (if dimension is divisible by 4).
        
        The signature is a topological invariant defined as the signature of the 
        intersection form on the middle-dimensional homology.
        
        Returns:
        -------
        int or None
            The signature, or None if not applicable
        """
        if self._signature is not None:
            return self._signature
        
        # Signature only defined for 4k-dimensional manifolds
        if self.dimension % 4 != 0:
            return None
        
        # For common manifolds, return known values
        if 'sphere' in self.name.lower() or 'S^' in self.name:
            # Signature of S^4k is 0
            self._signature = 0
        elif 'K3' in self.name:
            # K3 surface has signature -16
            self._signature = -16
        elif 'E8' in self.name:
            # E8 manifold has signature 8
            self._signature = 8
        elif 'torus' in self.name.lower() or 'T^' in self.name:
            # Signature of T^4k is 0
            self._signature = 0
        elif 'CP^' in self.name:
            # Signature of CP^2k is 1
            self._signature = 1
        else:
            # For general manifolds, would compute from intersection form
            self._signature = None
        
        return self._signature
    
    def compute_characteristic_classes(self):
        """
        Compute characteristic classes for the manifold.
        
        Characteristic classes are cohomology classes that measure the twisting
        of vector bundles over the manifold.
        
        Returns:
        -------
        dict
            Dictionary of characteristic classes
        """
        # This would compute Stiefel-Whitney, Chern, Pontryagin, and Euler classes
        # Simplified implementation for common examples
        
        result = {}
        
        if 'sphere' in self.name.lower() or 'S^' in self.name:
            # Spheres have trivial tangent bundle except for even dimensions
            if self.dimension % 2 == 0:
                # Euler class is non-trivial for even-dimensional spheres
                result['euler_class'] = 'non-zero'
            else:
                result['euler_class'] = 'zero'
            
            # All Stiefel-Whitney classes vanish except w_n for even n
            result['stiefel_whitney'] = {
                'w_n': 'non-zero' if self.dimension % 2 == 0 else 'zero'
            }
            
            # Pontryagin classes vanish for spheres
            result['pontryagin'] = {'all': 'zero'}
            
        elif 'CP^' in self.name:
            n = int(self.name.split('^')[1]) if '^' in self.name else self.dimension // 2
            # CP^n has non-trivial Chern classes
            result['chern'] = {f'c_{i}': 'non-zero' for i in range(1, n+1)}
            
            # Pontryagin classes are related to Chern classes
            result['pontryagin'] = {f'p_{i}': 'non-zero' for i in range(1, n//2+1)}
            
        elif 'torus' in self.name.lower() or 'T^' in self.name:
            # Tori have trivial tangent bundle
            result['stiefel_whitney'] = {'all': 'zero'}
            result['pontryagin'] = {'all': 'zero'}
            result['euler_class'] = 'zero'
            
        self._characteristic_classes = result
        return result
    
    def is_diffeomorphic_to(self, other_manifold):
        """
        Check if this manifold is diffeomorphic to another manifold.
        
        Parameters:
        ----------
        other_manifold : SmoothManifold
            Another smooth manifold
            
        Returns:
        -------
        bool or None
            True if diffeomorphic, False if not, None if unknown
        """
        # First check dimensions
        if self.dimension != other_manifold.dimension:
            return False
        
        # Check easy invariants
        if self.euler_characteristic() != other_manifold.euler_characteristic():
            return False
        
        if self.is_orientable() != other_manifold.is_orientable():
            return False
        
        # For dimensions 1 and 2, we can fully classify
        if self.dimension == 1:
            # 1-manifolds are classified by orientability and boundary
            return (self.is_orientable() == other_manifold.is_orientable() and
                    (self.boundary is None) == (other_manifold.boundary is None))
            
        elif self.dimension == 2:
            # 2-manifolds are classified by genus, orientability, and boundary
            # This is a simplification - we would need to compute genus
            if 'sphere' in self.name.lower() and 'sphere' in other_manifold.name.lower():
                return True
            elif 'torus' in self.name.lower() and 'torus' in other_manifold.name.lower():
                return True
            elif 'projective' in self.name.lower() and 'projective' in other_manifold.name.lower():
                return True
            elif 'Klein' in self.name and 'Klein' in other_manifold.name:
                return True
            
        # For dimension 3, we would need more sophisticated methods
        # For dimension 4 and higher, this is an extremely difficult problem
        
        # If we can't determine, return None
        return None
    
    def classify(self):
        """
        Attempt to classify the manifold up to diffeomorphism.
        
        Returns:
        -------
        str
            Classification information
        """
        # Classification depends heavily on dimension
        if self.dimension == 1:
            if self.boundary is None:
                if self.is_orientable():
                    return "Circle S¹ (unique closed orientable 1-manifold)"
                else:
                    return "Möbius band with boundary removed (unique closed non-orientable 1-manifold)"
            else:
                return "Interval [0,1] (unique compact 1-manifold with boundary)"
                
        elif self.dimension == 2:
            if self.boundary is None:
                # Closed surfaces are classified by genus and orientability
                if self.is_orientable():
                    # Compute genus from Euler characteristic: χ = 2 - 2g
                    euler = self.euler_characteristic()
                    if euler is None:
                        return "Orientable surface of unknown genus"
                    
                    genus = (2 - euler) // 2
                    if genus == 0:
                        return "Sphere S² (genus 0 orientable surface)"
                    elif genus == 1:
                        return "Torus T² (genus 1 orientable surface)"
                    else:
                        return f"Connected sum of {genus} tori (genus {genus} orientable surface)"
                else:
                    # Non-orientable surfaces characterized by number of projective planes
                    euler = self.euler_characteristic()
                    if euler is None:
                        return "Non-orientable surface of unknown type"
                    
                    # For non-orientable surfaces: χ = 2 - k where k is # of RP² factors
                    k = 2 - euler
                    if k == 1:
                        return "Real projective plane RP² (non-orientable surface with 1 cross-cap)"
                    elif k == 2:
                        return "Klein bottle (non-orientable surface with 2 cross-caps)"
                    else:
                        return f"Connected sum of {k} real projective planes (non-orientable surface with {k} cross-caps)"
            else:
                # Surfaces with boundary are classified by genus, orientability, and # of boundary components
                return "Surface with boundary (need more information to classify completely)"
                
        elif self.dimension == 3:
            # 3-manifold classification is much more complex
            # We would need to check if it's prime, Seifert-fibered, hyperbolic, etc.
            if 'sphere' in self.name.lower() or 'S^3' in self.name:
                return "3-sphere S³"
            elif 'torus' in self.name.lower() or 'T^3' in self.name:
                return "3-torus T³"
            else:
                return "3-manifold (classification requires advanced techniques: Geometrization Theorem)"
                
        elif self.dimension == 4:
            # 4-manifold classification is an active area of research with no complete solution
            if 'sphere' in self.name.lower() or 'S^4' in self.name:
                return "4-sphere S⁴"
            elif 'K3' in self.name:
                return "K3 surface"
            elif 'E8' in self.name:
                return "E8 manifold"
            elif 'CP^2' in self.name:
                return "Complex projective plane CP²"
            else:
                return "4-manifold (classification is open problem in mathematics)"
                
        else:
            # High-dimensional manifolds (dim ≥ 5) are classified through surgery theory
            # but this is a very complex topic
            return f"{self.dimension}-manifold (high-dimensional classification requires surgery theory)"
    
    def construct_from_surgery(self, base_manifold, surgery_data):
        """
        Construct a new manifold by performing surgery on a base manifold.
        
        Parameters:
        ----------
        base_manifold : SmoothManifold
            The base manifold on which to perform surgery
        surgery_data : list
            List of surgery instructions (spheres and framings)
            
        Returns:
        -------
        SmoothManifold
            The resulting manifold after surgery
        """
        # This is a placeholder for a complex operation
        # In reality, this would modify the manifold structure
        
        # Copy basic properties from the base manifold
        self.dimension = base_manifold.dimension
        
        # Surgery doesn't change dimension, but changes topology
        if base_manifold.name == "S³" and len(surgery_data) == 1:
            # Special case: surgery on unknot in S³ with framing n
            framing = surgery_data[0].get('framing', 0)
            if framing == 0:
                self.name = "S¹ × S²"
            elif framing == 1:
                self.name = "S³"  # Surgery on S³ with +1 framing gives back S³
            elif framing == -1:
                self.name = "S³"  # Surgery on S³ with -1 framing gives back S³
            else:
                self.name = f"Lens space L({abs(framing)},1)"
        else:
            self.name = f"Result of surgery on {base_manifold.name}"
        
        # Reset computed invariants
        self._euler_characteristic = None
        self._homology = None
        self._signature = None
        
        return self
    
    def connected_sum(self, other_manifold):
        """
        Construct the connected sum of this manifold with another.
        
        Parameters:
        ----------
        other_manifold : SmoothManifold
            Another manifold of the same dimension
            
        Returns:
        -------
        SmoothManifold
            The connected sum
        """
        if self.dimension != other_manifold.dimension:
            raise ValueError("Connected sum requires manifolds of the same dimension")
        
        # Create the connected sum
        result = SmoothManifold(self.dimension)
        
        # Handle special cases
        if 'sphere' in self.name.lower() or 'S^' in self.name:
            # M # S^n ≅ M for any n-manifold M
            result.name = other_manifold.name
        elif 'sphere' in other_manifold.name.lower() or 'S^' in other_manifold.name:
            result.name = self.name
        else:
            # Generic case
            result.name = f"{self.name} # {other_manifold.name}"
        
        # Connected sum of orientable manifolds is orientable
        result._orientable = self.is_orientable() and other_manifold.is_orientable()
        
        # Euler characteristic is additive minus 2
        if self.euler_characteristic() is not None and other_manifold.euler_characteristic() is not None:
            result._euler_characteristic = (
                self.euler_characteristic() + 
                other_manifold.euler_characteristic() - 2
            )
        
        return result
    
    def is_cobordant_to(self, other_manifold):
        """
        Check if this manifold is cobordant to another manifold.
        
        Two n-manifolds are cobordant if there exists an (n+1)-manifold
        whose boundary is the disjoint union of the two manifolds.
        
        Parameters:
        ----------
        other_manifold : SmoothManifold
            Another smooth manifold
            
        Returns:
        -------
        bool or None
            True if cobordant, False if not, None if unknown
        """
        # First check dimensions
        if self.dimension != other_manifold.dimension:
            return False
        
        # For oriented manifolds, check if Stiefel-Whitney numbers match
        if self.is_orientable() and other_manifold.is_orientable():
            # In reality, we would compute and compare Stiefel-Whitney numbers
            # This is a placeholder implementation
            # Two oriented manifolds are cobordant iff they have the same Pontryagin numbers
            return None  # Cannot determine without computing Pontryagin numbers
        
        # For unoriented manifolds, check if Stiefel-Whitney numbers match
        # This is a placeholder implementation
        return None  # Cannot determine without computing Stiefel-Whitney numbers
    
    def visualize(self, method='wireframe'):
        """
        Visualize the manifold if possible.
        
        Parameters:
        ----------
        method : str, optional
            Visualization method ('wireframe', 'surface', 'embedding')
        """
        # Only implemented for some specific manifolds and low dimensions
        fig = plt.figure(figsize=(10, 8))
        
        if self.dimension == 1:
            if 'circle' in self.name.lower() or 'S^1' in self.name:
                # Visualize a circle
                theta = np.linspace(0, 2*np.pi, 100)
                x = np.cos(theta)
                y = np.sin(theta)
                plt.plot(x, y)
                plt.axis('equal')
                plt.title('Circle S¹')
                
        elif self.dimension == 2:
            if 'sphere' in self.name.lower() or 'S^2' in self.name:
                # Visualize a sphere
                ax = fig.add_subplot(111, projection='3d')
                u = np.linspace(0, 2*np.pi, 30)
                v = np.linspace(0, np.pi, 30)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
                
                if method == 'wireframe':
                    ax.plot_wireframe(x, y, z, color='blue', alpha=0.5)
                else:
                    ax.plot_surface(x, y, z, color='blue', alpha=0.5)
                
                ax.set_title('2-Sphere S²')
                
            elif 'torus' in self.name.lower() or 'T^2' in self.name:
                # Visualize a torus
                ax = fig.add_subplot(111, projection='3d')
                u = np.linspace(0, 2*np.pi, 30)
                v = np.linspace(0, 2*np.pi, 30)
                u, v = np.meshgrid(u, v)
                
                R = 3  # Major radius
                r = 1  # Minor radius
                
                x = (R + r*np.cos(v)) * np.cos(u)
                y = (R + r*np.cos(v)) * np.sin(u)
                z = r * np.sin(v)
                
                if method == 'wireframe':
                    ax.plot_wireframe(x, y, z, color='blue', alpha=0.5)
                else:
                    ax.plot_surface(x, y, z, color='blue', alpha=0.5)
                
                ax.set_title('2-Torus T²')
                
            elif 'projective' in self.name.lower() or 'RP^2' in self.name:
                # Visualize Roman surface (a model of RP²)
                ax = fig.add_subplot(111, projection='3d')
                u = np.linspace(0, np.pi, 30)
                v = np.linspace(0, 2*np.pi, 30)
                u, v = np.meshgrid(u, v)
                
                # Parametrization of the Roman surface
                x = np.sin(u) * np.cos(u) * np.sin(v)
                y = np.sin(u) * np.cos(u) * np.cos(v)
                z = np.sin(u) ** 2
                
                if method == 'wireframe':
                    ax.plot_wireframe(x, y, z, color='blue', alpha=0.5)
                else:
                    ax.plot_surface(x, y, z, color='blue', alpha=0.5)
                
                ax.set_title('Real Projective Plane RP² (Roman Surface)')
                
            elif 'klein' in self.name.lower():
                # Visualize a Klein bottle
                ax = fig.add_subplot(111, projection='3d')
                u = np.linspace(0, 2*np.pi, 30)
                v = np.linspace(0, 2*np.pi, 30)
                u, v = np.meshgrid(u, v)
                
                # "Figure 8" immersion of the Klein bottle
                r = 1.5
                
                x = (r + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.cos(u)
                y = (r + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.sin(u)
                z = np.sin(u/2) * np.sin(v) + np.cos(u/2) * np.sin(2*v)
                
                if method == 'wireframe':
                    ax.plot_wireframe(x, y, z, color='blue', alpha=0.5)
                else:
                    ax.plot_surface(x, y, z, color='blue', alpha=0.5)
                
                ax.set_title('Klein Bottle')
        
        else:
            # Higher dimensional manifolds can't be visualized directly
            plt.text(0.5, 0.5, f"Cannot directly visualize {self.dimension}-dimensional manifold",
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        plt.show()


#############################################
# PART 2: MORSE THEORY AND HANDLE DECOMPOSITION
#############################################

class MorseTheory:
    """
    Implementation of Morse theory for analyzing smooth manifolds.
    
    Morse theory studies the topology of manifolds by analyzing
    critical points of smooth functions on the manifold.
    """
    
    @staticmethod
    def morse_function(manifold, function_type='standard'):
        """
        Construct a Morse function on a manifold.
        
        Parameters:
        ----------
        manifold : SmoothManifold
            The manifold on which to define the Morse function
        function_type : str, optional
            Type of Morse function to construct
            
        Returns:
        -------
        callable
            A Morse function on the manifold
        """
        # This is a simplified implementation
        # In reality, constructing Morse functions is non-trivial
        
        if function_type == 'standard':
            # Standard height function for simple manifolds
            if 'sphere' in manifold.name.lower() or 'S^' in manifold.name:
                # Height function on a sphere has two critical points (min and max)
                return lambda x: x[-1]  # Last coordinate as height
            elif 'torus' in manifold.name.lower() or 'T^' in manifold.name:
                # Height function on a torus has four critical points
                return lambda x: x[-1]  # Last coordinate as height
        
        # Default generic Morse function (placeholder)
        return lambda x: np.sum(x**2)  # Sum of squares
    
    @staticmethod
    def find_critical_points(morse_function, manifold, domain=None):
        """
        Find the critical points of a Morse function on a manifold.
        
        Parameters:
        ----------
        morse_function : callable
            A Morse function
        manifold : SmoothManifold
            The manifold
        domain : array-like, optional
            Domain on which to search for critical points
            
        Returns:
        -------
        list
            List of critical points and their indices
        """
        # This would find where the gradient vanishes
        # Simplified implementation for common cases
        
        if 'sphere' in manifold.name.lower() or 'S^' in manifold.name:
            # Standard height function on S^n has two critical points
            n = manifold.dimension
            north_pole = np.zeros(n+1)
            north_pole[-1] = 1
            south_pole = np.zeros(n+1)
            south_pole[-1] = -1
            
            return [
                {'point': north_pole, 'index': n, 'value': 1},
                {'point': south_pole, 'index': 0, 'value': -1}
            ]
            
        elif 'torus' in manifold.name.lower() or 'T^2' in manifold.name and manifold.dimension == 2:
            # Standard height function on T^2 has four critical points
            # Critical points at top, bottom, inner equator, outer equator
            R = 3  # Major radius
            r = 1  # Minor radius
            
            top = np.array([R, 0, r])
            bottom = np.array([R, 0, -r])
            inner = np.array([R-r, 0, 0])
            outer = np.array([R+r, 0, 0])
            
            return [
                {'point': top, 'index': 2, 'value': r},
                {'point': outer, 'index': 1, 'value': 0},
                {'point': inner, 'index': 1, 'value': 0},
                {'point': bottom, 'index': 0, 'value': -r}
            ]
        
        # Default placeholder
        return []
    
    @staticmethod
    def compute_morse_homology(critical_points, manifold):
        """
        Compute homology using Morse theory.
        
        Parameters:
        ----------
        critical_points : list
            List of critical points with indices
        manifold : SmoothManifold
            The manifold
            
        Returns:
        -------
        dict
            Homology groups computed via Morse theory
        """
        # Group critical points by index
        critical_by_index = defaultdict(int)
        for cp in critical_points:
            critical_by_index[cp['index']] += 1
        
        # Morse homology: rank of H_k is the number of critical points of index k
        # minus the number of gradient flow lines between index k+1 and k points
        # This is a simplified version assuming minimal gradient flows
        
        homology = {}
        for k in range(manifold.dimension + 1):
            rank = critical_by_index.get(k, 0)
            homology[k] = (rank, [])  # (rank, torsion)
            
        # Check against known results
        if manifold._homology:
            for k, (rank, torsion) in manifold._homology.items():
                if k in homology and homology[k][0] != rank:
                    print(f"Warning: Morse homology rank for H_{k} ({homology[k][0]}) "
                          f"doesn't match expected rank ({rank})")
        
        return homology
    
    @staticmethod
    def handle_decomposition(manifold, critical_points=None):
        """
        Construct a handle decomposition of the manifold from Morse function data.
        
        Parameters:
        ----------
        manifold : SmoothManifold
            The manifold
        critical_points : list, optional
            Critical points of a Morse function on the manifold
            
        Returns:
        -------
        dict
            Handle decomposition of the manifold
        """
        if critical_points is None:
            # Get a Morse function and its critical points
            morse_fn = MorseTheory.morse_function(manifold)
            critical_points = MorseTheory.find_critical_points(morse_fn, manifold)
        
        # Group handles by index
        handles = defaultdict(int)
        for cp in critical_points:
            handles[cp['index']] += 1
        
        # Create a handle decomposition description
        description = []
        
        # Start with 0-handles
        n_0_handles = handles.get(0, 0)
        if n_0_handles > 0:
            description.append(f"{n_0_handles} 0-handle(s)")
        
        # Add k-handles for k > 0
        for k in range(1, manifold.dimension + 1):
            n_k_handles = handles.get(k, 0)
            if n_k_handles > 0:
                description.append(f"{n_k_handles} {k}-handle(s)")
        
        return {
            'handles': dict(handles),
            'description': ", ".join(description)
        }
    
    @staticmethod
    def visualize_handle_decomposition(manifold):
        """
        Visualize a handle decomposition of the manifold.
        
        Parameters:
        ----------
        manifold : SmoothManifold
            The manifold to visualize
        """
        # Get handle decomposition
        morse_fn = MorseTheory.morse_function(manifold)
        critical_points = MorseTheory.find_critical_points(morse_fn, manifold)
        decomposition = MorseTheory.handle_decomposition(manifold, critical_points)
        
        # Only implemented for dimension 2 manifolds
        if manifold.dimension != 2:
            print(f"Visualization of handle decomposition not implemented for dimension {manifold.dimension}")
            print(f"Handle decomposition: {decomposition['description']}")
            return
        
        # Visualize 2-manifold handle decomposition
        fig = plt.figure(figsize=(12, 6))
        
        if 'sphere' in manifold.name.lower() or 'S^2' in manifold.name:
            # Sphere: 1 0-handle + 1 2-handle
            ax1 = fig.add_subplot(121)
            circle = plt.Circle((0, 0), 1, fill=False, color='blue')
            ax1.add_patch(circle)
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
            ax1.set_aspect('equal')
            ax1.set_title("0-handle (disk)")
            
            ax2 = fig.add_subplot(122)
            circle = plt.Circle((0, 0), 1, fill=False, color='blue')
            ax2.add_patch(circle)
            ax2.set_xlim(-1.5, 1.5)
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_aspect('equal')
            ax2.set_title("Attach 2-handle (disk) along boundary")
            
        elif 'torus' in manifold.name.lower() or 'T^2' in manifold.name:
            # Torus: 1 0-handle + 2 1-handles + 1 2-handle
            ax1 = fig.add_subplot(221)
            circle = plt.Circle((0, 0), 1, fill=False, color='blue')
            ax1.add_patch(circle)
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
            ax1.set_aspect('equal')
            ax1.set_title("0-handle (disk)")
            
            ax2 = fig.add_subplot(222)
            circle = plt.Circle((0, 0), 1, fill=False, color='blue')
            ax2.add_patch(circle)
            # Add first 1-handle
            rect1 = plt.Rectangle((-0.2, -1), 0.4, 2, color='red', alpha=0.5)
            ax2.add_patch(rect1)
            ax2.set_xlim(-1.5, 1.5)
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_aspect('equal')
            ax2.set_title("Attach first 1-handle")
            
            ax3 = fig.add_subplot(223)
            # Draw band with hole
            theta = np.linspace(0, 2*np.pi, 100)
            x_outer = 1.0 * np.cos(theta)
            y_outer = 1.0 * np.sin(theta)
            x_inner1 = 0.2 * np.cos(theta) - 0.6
            y_inner1 = 0.2 * np.sin(theta) - 0.6
            x_inner2 = 0.2 * np.cos(theta) + 0.6
            y_inner2 = 0.2 * np.sin(theta) + 0.6
            
            ax3.plot(x_outer, y_outer, 'b-')
            ax3.plot(x_inner1, y_inner1, 'b-')
            ax3.plot(x_inner2, y_inner2, 'b-')
            # Add second 1-handle
            ax3.plot([-0.8, 0.8], [-0.6, 0.6], 'r-', linewidth=4, alpha=0.5)
            ax3.set_xlim(-1.5, 1.5)
            ax3.set_ylim(-1.5, 1.5)
            ax3.set_aspect('equal')
            ax3.set_title("Attach second 1-handle")
            
            ax4 = fig.add_subplot(224)
            # Draw torus outline
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0, 2*np.pi, 20)
            u_mesh, v_mesh = np.meshgrid(u, v)
            
            R = 1.0  # Major radius
            r = 0.4  # Minor radius
            
            x = (R + r*np.cos(v_mesh)) * np.cos(u_mesh)
            y = (R + r*np.cos(v_mesh)) * np.sin(u_mesh)
            z = r * np.sin(v_mesh)
            
            ax4 = fig.add_subplot(224, projection='3d')
            ax4.plot_surface(x, y, z, color='blue', alpha=0.3)
            ax4.set_title("Final torus after 2-handle")
        
        plt.tight_layout()
        plt.show()


#############################################
# PART 3: SURGERY THEORY AND CLASSIFICATION
#############################################

class SurgeryTheory:
    """
    Implementation of surgery theory for classifying manifolds.
    
    Surgery theory is a technique used to construct and classify
    manifolds by performing operations on simpler ones.
    """
    
    @staticmethod
    def dehn_surgery(knot, framing):
        """
        Perform Dehn surgery along a knot in a 3-manifold.
        
        Parameters:
        ----------
        knot : object
            Representation of a knot
        framing : int
            Surgery framing
            
        Returns:
        -------
        SmoothManifold
            Resulting 3-manifold after surgery
        """
        # Create a new 3-manifold
        result = SmoothManifold(3)
        
        # Dehn surgery on unknot in S³ with different framings
        if knot.name == "Unknot" and hasattr(knot, 'ambient_manifold') and knot.ambient_manifold.name == "S³":
            if framing == 0:
                result.name = "S¹ × S²"
            elif framing == 1:
                result.name = "S³"  # Surgery on S³ with +1 framing gives back S³
            elif framing == -1:
                result.name = "S³"  # Surgery on S³ with -1 framing gives back S³
            else:
                result.name = f"Lens space L({abs(framing)},1)"
        elif knot.name == "Trefoil" and hasattr(knot, 'ambient_manifold') and knot.ambient_manifold.name == "S³":
            # Surgery on trefoil creates more complex 3-manifolds
            if framing == 0:
                result.name = "Brieskorn sphere Σ(2,3,6) = Seifert fibered manifold"
            else:
                result.name = f"Result of {framing}-surgery on trefoil knot"
        else:
            result.name = f"Result of {framing}-surgery on {knot.name}"
        
        # Set basic properties
        result._orientable = True  # Surgery preserves orientability
        
        return result
    
    @staticmethod
    def surgery_along_sphere(manifold, sphere_data):
        """
        Perform surgery along an embedded sphere.
        
        Parameters:
        ----------
        manifold : SmoothManifold
            The ambient manifold
        sphere_data : dict
            Data about the embedded sphere and surgery
            
        Returns:
        -------
        SmoothManifold
            Resulting manifold after surgery
        """
        # Extract sphere information
        sphere_dim = sphere_data.get('dimension', 1)
        sphere_normal_bundle = sphere_data.get('normal_bundle', 'trivial')
        framing = sphere_data.get('framing', None)
        
        # Create a new manifold
        result = SmoothManifold(manifold.dimension)
        
        # Handle special cases
        if sphere_dim == 1 and manifold.dimension == 3:
            # This is Dehn surgery in dimension 3
            knot = type('Knot', (), {'name': sphere_data.get('name', 'Unknown knot'), 
                                     'ambient_manifold': manifold})
            return SurgeryTheory.dehn_surgery(knot, framing)
        
        # Generic case
        result.name = f"Result of surgery on {sphere_dim}-sphere in {manifold.name}"
        
        # Special cases where result is known
        if (sphere_dim == 0 and manifold.dimension >= 3 and
            'sphere' in manifold.name.lower() and sphere_normal_bundle == 'trivial'):
            # Surgery on S^0 in S^n gives S^{n-1} × S^1
            n = manifold.dimension
            result.name = f"S^{sphere_dim} × S^{n-sphere_dim}"
            
        return result
    
    @staticmethod
    def compute_normal_invariants(manifold):
        """
        Compute normal invariants used in surgery classification.
        
        Parameters:
        ----------
        manifold : SmoothManifold
            The manifold
            
        Returns:
        -------
        dict
            Normal invariants
        """
        # This is a placeholder for a complex computation
        # Normal invariants are elements of [X, G/O] in surgery theory
        
        return {
            'normal_invariants': 'Requires advanced computation',
            'surgery_obstruction': None
        }
    
    @staticmethod
    def surgery_obstruction(normal_invariant, manifold):
        """
        Compute the surgery obstruction.
        
        Parameters:
        ----------
        normal_invariant : object
            Normal invariant
        manifold : SmoothManifold
            The manifold
            
        Returns:
        -------
        object
            Surgery obstruction
        """
        # This is a placeholder for a complex computation
        # The surgery obstruction determines whether a normal invariant can be realized
        # by a homotopy equivalence that is homotopic to a diffeomorphism
        
        return {
            'obstruction': 'Requires advanced computation',
            'is_zero': None
        }
    
    @staticmethod
    def visualize_surgery(manifold, surgery_data):
        """
        Visualize a surgery operation if possible.
        
        Parameters:
        ----------
        manifold : SmoothManifold
            The manifold
        surgery_data : dict
            Surgery specifications
        """
        # Only implemented for some simple 3-dimensional cases
        if manifold.dimension != 3:
            print(f"Visualization of surgery not implemented for dimension {manifold.dimension}")
            return
        
        # Extract surgery information
        surgery_type = surgery_data.get('type', 'Dehn')
        
        if surgery_type == 'Dehn':
            # Dehn surgery on a knot in S³
            knot_type = surgery_data.get('knot', 'unknot')
            framing = surgery_data.get('framing', 0)
            
            fig = plt.figure(figsize=(15, 5))
            
            # Draw the knot
            ax1 = fig.add_subplot(131, projection='3d')
            if knot_type.lower() == 'unknot':
                # Draw unknot (circle)
                theta = np.linspace(0, 2*np.pi, 100)
                x = np.cos(theta)
                y = np.sin(theta)
                z = np.zeros_like(theta)
                ax1.plot(x, y, z, 'b-', linewidth=2)
            elif knot_type.lower() == 'trefoil':
                # Draw trefoil knot
                t = np.linspace(0, 2*np.pi, 100)
                x = np.sin(t) + 2 * np.sin(2*t)
                y = np.cos(t) - 2 * np.cos(2*t)
                z = -np.sin(3*t)
                ax1.plot(x, y, z, 'b-', linewidth=2)
            else:
                ax1.text(0, 0, 0, f"Knot: {knot_type}", fontsize=12)
            
            ax1.set_title(f"Knot: {knot_type}")
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # Draw the surgery torus
            ax2 = fig.add_subplot(132, projection='3d')
            if knot_type.lower() == 'unknot':
                # Draw a torus around the unknot
                u = np.linspace(0, 2*np.pi, 20)
                v = np.linspace(0, 2*np.pi, 20)
                u, v = np.meshgrid(u, v)
                
                r = 0.2  # Small radius for surgery torus
                R = 1.0  # Major radius (from unknot)
                
                x = (R + r*np.cos(v)) * np.cos(u)
                y = (R + r*np.cos(v)) * np.sin(u)
                z = r * np.sin(v)
                
                ax2.plot_surface(x, y, z, color='red', alpha=0.3)
                
                # Draw the meridian and longitude
                # Meridian (constant u)
                u_const = 0
                v_merid = np.linspace(0, 2*np.pi, 100)
                x_merid = (R + r*np.cos(v_merid)) * np.cos(u_const)
                y_merid = (R + r*np.cos(v_merid)) * np.sin(u_const)
                z_merid = r * np.sin(v_merid)
                ax2.plot(x_merid, y_merid, z_merid, 'g-', linewidth=2, label='Meridian')
                
                # Longitude (constant v)
                v_const = 0
                u_long = np.linspace(0, 2*np.pi, 100)
                x_long = (R + r*np.cos(v_const)) * np.cos(u_long)
                y_long = (R + r*np.cos(v_const)) * np.sin(u_long)
                z_long = r * np.sin(v_const)
                ax2.plot(x_long, y_long, z_long, 'y-', linewidth=2, label='Longitude')
                
                # Draw the surgery curve based on framing
                if framing != 0:
                    # Draw (p,q) curve based on framing
                    p = abs(framing)
                    q = 1
                    t = np.linspace(0, 2*np.pi*p, 100*p)
                    x_surg = (R + r*np.cos(q*t/p)) * np.cos(t)
                    y_surg = (R + r*np.cos(q*t/p)) * np.sin(t)
                    z_surg = r * np.sin(q*t/p)
                    ax2.plot(x_surg, y_surg, z_surg, 'r-', linewidth=3, label=f'({p},{q}) Curve')
                    
            else:
                ax2.text(0, 0, 0, "Surgery torus visualization\nnot implemented for this knot", fontsize=12)
            
            ax2.set_title(f"Surgery Torus (Framing: {framing})")
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            if knot_type.lower() == 'unknot':
                ax2.legend()
            
            # Show the result
            ax3 = fig.add_subplot(133)
            
            # Determine the result
            if knot_type.lower() == 'unknot':
                if framing == 0:
                    result = "S¹ × S²"
                elif abs(framing) == 1:
                    result = "S³"
                else:
                    result = f"Lens space L({abs(framing)},1)"
            elif knot_type.lower() == 'trefoil':
                if framing == 0:
                    result = "Brieskorn sphere Σ(2,3,6)"
                else:
                    result = f"Result of {framing}-surgery on trefoil"
            else:
                result = f"Result of {framing}-surgery on {knot_type}"
            
            ax3.text(0.5, 0.5, f"Result: {result}", fontsize=14, 
                     horizontalalignment='center', verticalalignment='center')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.show()


#############################################
# PART 4: SMOOTH 4-MANIFOLD THEORY
#############################################

class SmoothFourManifold(SmoothManifold):
    """
    Specialized class for smooth 4-manifolds, which have unique properties
    and classification challenges.
    """
    
    def __init__(self, name=None):
        """
        Initialize a smooth 4-manifold.
        
        Parameters:
        ----------
        name : str, optional
            Name or description of the 4-manifold
        """
        super().__init__(4, name)
        
        # Additional invariants specific to 4-manifolds
        self._intersection_form = None
        self._kirby_diagram = None
        self._seiberg_witten_invariants = {}
        self._donaldson_polynomials = {}
    
    def intersection_form(self):
        """
        Compute the intersection form of the 4-manifold.
        
        The intersection form is a symmetric bilinear form on H²(M;Z)/torsion.
        
        Returns:
        -------
        array
            Matrix representing the intersection form
        str
            Type of the intersection form (even/odd, definite/indefinite)
        """
        if self._intersection_form is not None:
            return self._intersection_form
        
        # For common 4-manifolds, return known intersection forms
        if 'S^4' in self.name or 'sphere' in self.name.lower():
            # Intersection form is empty (0×0 matrix)
            matrix = np.array([])
            form_type = "even definite"
        elif 'CP^2' in self.name:
            # Intersection form is [1] (positive definite)
            matrix = np.array([[1]])
            form_type = "odd definite"
        elif '-CP^2' in self.name:
            # Intersection form is [-1] (negative definite)
            matrix = np.array([[-1]])
            form_type = "odd definite"
        elif 'S^2 x S^2' in self.name:
            # Intersection form is [[0,1],[1,0]] (even indefinite)
            matrix = np.array([[0, 1], [1, 0]])
            form_type = "even indefinite"
        elif 'K3' in self.name:
            # Intersection form is 2(-E8) ⊕ 3H (even indefinite)
            # Where H is the hyperbolic form and E8 is the E8 form
            # This is a 22×22 matrix, simplified here
            matrix = np.eye(22)
            matrix[:16, :16] = -1 * np.eye(16)  # -2E8 block
            # Add 3 hyperbolic forms
            for i in range(3):
                idx = 16 + 2*i
                matrix[idx, idx] = 0
                matrix[idx, idx+1] = 1
                matrix[idx+1, idx] = 1
                matrix[idx+1, idx+1] = 0
            form_type = "even indefinite"
        elif 'E8' in self.name:
            # E8 manifold has intersection form E8 (even definite)
            # E8 is an 8×8 matrix with specific entries
            matrix = np.zeros((8, 8))
            # Diagonal entries
            for i in range(8):
                matrix[i, i] = 2
            # Off-diagonal entries based on E8 Dynkin diagram
            matrix[0, 1] = matrix[1, 0] = 1
            matrix[1, 2] = matrix[2, 1] = 1
            matrix[2, 3] = matrix[3, 2] = 1
            matrix[3, 4] = matrix[4, 3] = 1
            matrix[4, 5] = matrix[5, 4] = 1
            matrix[5, 6] = matrix[6, 5] = 1
            matrix[2, 7] = matrix[7, 2] = 1
            form_type = "even definite"
        else:
            # For unknown manifolds, return None
            matrix = None
            form_type = None
        
        self._intersection_form = (matrix, form_type)
        return self._intersection_form
    
    def is_simply_connected(self):
        """
        Check if the 4-manifold is simply connected.
        
        Returns:
        -------
        bool
            True if simply connected, False otherwise
        """
        # For known manifolds, return correct value
        if 'S^4' in self.name or 'sphere' in self.name.lower():
            return True
        elif 'CP^2' in self.name:
            return True
        elif 'S^2 x S^2' in self.name:
            return True
        elif 'K3' in self.name:
            return True
        elif 'E8' in self.name:
            return True
        elif 'T^4' in self.name or 'torus' in self.name.lower():
            return False
        
        # Default to None for unknown
        return None
    
    def create_exotic_pair(self):
        """
        Create an exotic pair - another 4-manifold homeomorphic but not
        diffeomorphic to this one.
        
        Returns:
        -------
        SmoothFourManifold
            An exotic version of this manifold
        """
        # Some known exotic pairs
        if 'S^4' in self.name:
            # No known exotic S⁴ (still an open question)
            return None
        elif 'R^4' in self.name:
            # There are uncountably many exotic R⁴
            exotic = SmoothFourManifold("Exotic R⁴")
            exotic._additional_invariants["description"] = "Exotic R⁴ constructed via Casson handles"
            return exotic
        elif 'K3' in self.name:
            # K3 has infinitely many exotic smooth structures
            exotic = SmoothFourManifold("Exotic K3")
            exotic._additional_invariants["description"] = "Exotic K3 with non-trivial Seiberg-Witten invariants"
            return exotic
        elif 'CP^2' in self.name:
            # No known exotic CP²
            return None
        
        # Generic case - describe a theoretical exotic structure
        exotic = SmoothFourManifold(f"Exotic {self.name}")
        exotic._additional_invariants["description"] = (
            f"Theoretical exotic structure on {self.name}, which would be "
            "homeomorphic but not diffeomorphic to the standard structure."
        )
        return exotic
    
    def seiberg_witten_invariants(self, max_dims=3):
        """
        Compute Seiberg-Witten invariants of the 4-manifold.
        
        Parameters:
        ----------
        max_dims : int, optional
            Maximum dimension of the Seiberg-Witten moduli space to consider
            
        Returns:
        -------
        dict
            Dictionary of Seiberg-Witten invariants
        """
        if self._seiberg_witten_invariants:
            return self._seiberg_witten_invariants
        
        # For some standard manifolds, return known values
        if 'K3' in self.name:
            # K3 has a single non-zero SW invariant
            self._seiberg_witten_invariants = {
                'basic_class': 'trivial canonical class',
                'value': 1
            }
        elif 'E8' in self.name:
            self._seiberg_witten_invariants = {
                'basic_class': 'non-trivial canonical class',
                'value': 1
            }
        elif 'S^4' in self.name or 'sphere' in self.name.lower():
            # S⁴ has b⁺ = 0, so SW invariants vanish
            self._seiberg_witten_invariants = {
                'all_invariants': 0
            }
        elif 'CP^2' in self.name:
            # CP² with standard orientation has b⁺ = 1
            self._seiberg_witten_invariants = {
                'canonical_class': 'non-trivial',
                'value': 1
            }
        else:
            # Placeholder for a complex computation
            self._seiberg_witten_invariants = {
                'description': 'Computation of Seiberg-Witten invariants requires specialized techniques'
            }
        
        return self._seiberg_witten_invariants
    
    def donaldson_polynomials(self, max_degree=4):
        """
        Compute Donaldson polynomials of the 4-manifold.
        
        Parameters:
        ----------
        max_degree : int, optional
            Maximum degree of polynomials to compute
            
        Returns:
        -------
        dict
            Dictionary of Donaldson polynomials
        """
        if self._donaldson_polynomials:
            return self._donaldson_polynomials
        
        # Placeholder for a very complex computation
        self._donaldson_polynomials = {
            'description': 'Computation of Donaldson polynomials requires advanced gauge theory'
        }
        
        return self._donaldson_polynomials
    
    def kirby_diagram(self):
        """
        Construct a Kirby diagram for the 4-manifold.
        
        Returns:
        -------
        dict
            Information about the Kirby diagram
        """
        if self._kirby_diagram is not None:
            return self._kirby_diagram
        
        # For some standard manifolds, return known diagrams
        if 'S^4' in self.name or 'sphere' in self.name.lower():
            self._kirby_diagram = {
                'description': 'Empty Kirby diagram (no handles)',
                'zero_handles': 1,
                'one_handles': 0,
                'two_handles': 0,
                'three_handles': 0,
                'four_handles': 1
            }
        elif 'CP^2' in self.name:
            self._kirby_diagram = {
                'description': 'Single 2-handle attached along unknot with framing +1',
                'zero_handles': 1,
                'one_handles': 0,
                'two_handles': 1,
                'three_handles': 0,
                'four_handles': 1
            }
        elif '-CP^2' in self.name:
            self._kirby_diagram = {
                'description': 'Single 2-handle attached along unknot with framing -1',
                'zero_handles': 1,
                'one_handles': 0,
                'two_handles': 1,
                'three_handles': 0,
                'four_handles': 1
            }
        elif 'S^2 x S^2' in self.name:
            self._kirby_diagram = {
                'description': 'Two 2-handles attached along unlinked unknots with framing 0',
                'zero_handles': 1,
                'one_handles': 0,
                'two_handles': 2,
                'three_handles': 0,
                'four_handles': 1
            }
        else:
            # Generic placeholder
            self._kirby_diagram = {
                'description': f'Kirby diagram for {self.name} requires specialized construction'
            }
        
        return self._kirby_diagram
    
    def classify_by_wall_theorem(self):
        """
        Classify the 4-manifold using Wall's theorem for simply connected case.
        
        Returns:
        -------
        str
            Classification information
        """
        # Wall's theorem applies to simply connected, closed 4-manifolds
        if not self.is_simply_connected():
            return "Wall's theorem only applies to simply connected 4-manifolds"
        
        # Get the intersection form
        int_form, form_type = self.intersection_form()
        if int_form is None:
            return "Could not determine intersection form"
        
        # Wall's theorem: simply connected 4-manifolds are classified by:
        # 1. Intersection form
        # 2. Kirby-Siebenmann invariant (ignored here for simplicity)
        
        if form_type == "even definite":
            # Only known positive definite even form is empty
            if int_form.size == 0:
                return "Diffeomorphic to S⁴"
            
            # Negative definite even forms are related to E8
            if int_form.shape[0] % 8 == 0:
                k = int_form.shape[0] // 8
                return f"Homeomorphic to #_{k}(-E8)"
            else:
                return "No such 4-manifold exists (by Donaldson's theorem)"
            
        elif form_type == "odd definite":
            # Odd definite forms are determined by rank and signature
            signature = np.sum(np.sign(np.linalg.eigvals(int_form)))
            if signature > 0:
                return f"Homeomorphic to #{signature}CP²"
            elif signature < 0:
                return f"Homeomorphic to #{-signature}(-CP²)"
            else:
                return "Homeomorphic to S⁴"
            
        elif form_type == "even indefinite":
            # Even indefinite forms are classified by rank and signature
            rank = int_form.shape[0]
            signature = np.sum(np.sign(np.linalg.eigvalsh(int_form)))
            
            # Express in terms of E8 and hyperbolic form H
            pos = (rank + signature) // 2
            neg = (rank - signature) // 2
            
            if neg % 8 == 0:
                return f"Homeomorphic to {pos-neg//2}(S² × S²) # {neg//8}(-E8)"
            else:
                return f"Homeomorphic to {pos}(S² × S²) # {neg}(-CP²)"
            
        elif form_type == "odd indefinite":
            # Odd indefinite forms are classified by rank and signature
            rank = int_form.shape[0]
            signature = np.sum(np.sign(np.linalg.eigvalsh(int_form)))
            
            pos = (rank + signature) // 2
            neg = (rank - signature) // 2
            
            return f"Homeomorphic to {pos}CP² # {neg}(-CP²)"
        
        else:
            return "Could not classify (unknown intersection form type)"
    
    def visualize_kirby_diagram(self):
        """
        Visualize the Kirby diagram of the 4-manifold.
        """
        # Get Kirby diagram information
        kirby_info = self.kirby_diagram()
        
        # Set up figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        # Draw based on the manifold
        if 'S^4' in self.name or 'sphere' in self.name.lower():
            # Empty diagram
            ax.text(0.5, 0.5, "Empty Kirby diagram\n(S⁴)", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            ax.axis('off')
            
        elif 'CP^2' in self.name:
            # CP² has unknot with framing +1
            draw_knot(ax, 'unknot', framing=1)
            ax.set_title("Kirby diagram for CP²: unknot with +1 framing")
            
        elif '-CP^2' in self.name:
            # -CP² has unknot with framing -1
            draw_knot(ax, 'unknot', framing=-1)
            ax.set_title("Kirby diagram for -CP²: unknot with -1 framing")
            
        elif 'S^2 x S^2' in self.name:
            # S² × S² has two unlinked unknots with framing 0
            # First unknot
            theta1 = np.linspace(0, 2*np.pi, 100)
            x1 = 0.7 * np.cos(theta1) - 1
            y1 = 0.7 * np.sin(theta1)
            ax.plot(x1, y1, 'b-', linewidth=2)
            ax.text(-1, 0, "0", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            # Second unknot
            theta2 = np.linspace(0, 2*np.pi, 100)
            x2 = 0.7 * np.cos(theta2) + 1
            y2 = 0.7 * np.sin(theta2)
            ax.plot(x2, y2, 'r-', linewidth=2)
            ax.text(1, 0, "0", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_title("Kirby diagram for S² × S²: two 0-framed unknots")
            
        elif 'K3' in self.name:
            # K3 has a complicated Kirby diagram
            ax.text(0.5, 0.5, "K3 Surface Kirby Diagram\n(too complex to draw here)", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            ax.axis('off')
            
        else:
            # Generic message
            ax.text(0.5, 0.5, f"Kirby diagram for {self.name}\nnot implemented", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            ax.axis('off')
        
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()


# Helper function for drawing knots in Kirby diagrams
def draw_knot(ax, knot_type, framing=0):
    """Draw a knot with framing in a Kirby diagram."""
    if knot_type.lower() == 'unknot':
        # Draw circle (unknot)
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Add framing label
        ax.text(0, 0, str(framing), fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
    elif knot_type.lower() == 'trefoil':
        # Draw trefoil knot (simplified 2D projection)
        t = np.linspace(0, 2*np.pi, 1000)
        x = np.sin(t) + 2 * np.sin(2*t)
        y = np.cos(t) - 2 * np.cos(2*t)
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Add framing label
        ax.text(0, 0, str(framing), fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
    elif knot_type.lower() == 'figure-eight':
        # Draw figure-eight knot (simplified 2D projection)
        t = np.linspace(0, 2*np.pi, 1000)
        x = np.cos(t) * (2 + np.cos(2*t))
        y = np.sin(t) * (2 + np.cos(2*t))
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Add framing label
        ax.text(0, 0, str(framing), fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
    else:
        ax.text(0.5, 0.5, f"Knot type '{knot_type}' not implemented", 
                 horizontalalignment='center', verticalalignment='center')
    
    ax.set_aspect('equal')
    ax.axis('off')


#############################################
# PART 5: HIGH-DIMENSIONAL MANIFOLD CLASSIFICATION
#############################################

class HighDimensionalManifold(SmoothManifold):
    """
    Specialized class for high-dimensional manifolds (dimension ≥ 5).
    
    In high dimensions, the h-cobordism theorem and surgery theory
    provide powerful classification tools.
    """
    
    def __init__(self, dimension, name=None):
        """
        Initialize a high-dimensional manifold.
        
        Parameters:
        ----------
        dimension : int
            The dimension of the manifold (≥ 5)
        name : str, optional
            Name or description of the manifold
        """
        if dimension < 5:
            raise ValueError("HighDimensionalManifold requires dimension ≥ 5")
        
        super().__init__(dimension, name)
        
        # Additional invariants for high-dimensional manifolds
        self._whitehead_torsion = None
        self._tangential_homotopy_type = None
        self._surgery_obstructions = {}
    
    def classify_by_surgery_theory(self):
        """
        Classify the manifold using surgery theory.
        
        Returns:
        -------
        str
            Classification information
        """
        # This would implement the classification of high-dimensional manifolds
        # using the surgery exact sequence and surgery obstruction theory
        
        # Check if the manifold is simply connected
        is_simply_connected = self._additional_invariants.get('is_simply_connected')
        
        if is_simply_connected:
            # For simply connected manifolds, check if we're in the stable range
            if self.dimension >= 5:
                return (f"Simply connected {self.dimension}-manifolds in stable range "
                        f"are determined by their tangential homotopy type and surgery obstructions")
        
        # General case
        return f"Classification of {self.dimension}-manifold requires full surgery theory analysis"
    
    def h_cobordism_to(self, other_manifold):
        """
        Check if this manifold is h-cobordant to another manifold.
        
        Two manifolds are h-cobordant if there exists an h-cobordism between them.
        
        Parameters:
        ----------
        other_manifold : SmoothManifold
            Another smooth manifold
            
        Returns:
        -------
        bool or None
            True if h-cobordant, False if not, None if unknown
        """
        # First check dimensions
        if self.dimension != other_manifold.dimension:
            return False
        
        # For simply connected manifolds of dimension ≥ 5,
        # h-cobordism implies diffeomorphism (h-cobordism theorem)
        is_simply_connected = self._additional_invariants.get('is_simply_connected')
        other_simply_connected = other_manifold._additional_invariants.get('is_simply_connected')
        
        if (is_simply_connected and other_simply_connected and 
            self.dimension >= 5 and 
            self.is_diffeomorphic_to(other_manifold) is not None):
            # If we know whether they're diffeomorphic, we know if they're h-cobordant
            return self.is_diffeomorphic_to(other_manifold)
        
        # For non-simply connected manifolds, need to check Whitehead torsion
        # This is a placeholder - actual computation is complex
        return None
    
    def exotic_spheres(self):
        """
        Compute the exotic spheres in this dimension.
        
        Returns:
        -------
        dict
            Information about exotic spheres in this dimension
        """
        # Known groups of exotic spheres in different dimensions
        exotic_sphere_groups = {
            5: 0,    # No exotic 5-spheres
            6: 0,    # No exotic 6-spheres
            7: 28,   # Θ₇ ≅ Z/28Z
            8: 2,    # Θ₈ ≅ Z/2Z
            9: 8,    # Θ₉ ≅ Z/8Z ⊕ Z/2Z
            10: 6,   # Θ₁₀ ≅ Z/6Z
            11: 992  # Θ₁₁ is complex
            # Higher dimensions have more complex groups
        }
        
        # Get the number of exotic spheres in this dimension
        n_exotic = exotic_sphere_groups.get(self.dimension, "Unknown")
        
        # Create descriptive information
        if n_exotic == 0:
            description = f"There are no exotic {self.dimension}-spheres"
        elif isinstance(n_exotic, int):
            description = f"There are {n_exotic} exotic {self.dimension}-spheres"
        else:
            description = f"The group of exotic {self.dimension}-spheres is complex/unknown"
        
        # Return information
        return {
            'dimension': self.dimension,
            'number_of_exotic_spheres': n_exotic,
            'description': description
        }
    
    def construct_exotic_sphere(self, index=1):
        """
        Construct an exotic sphere in this dimension.
        
        Parameters:
        ----------
        index : int, optional
            Index of the exotic sphere to construct
            
        Returns:
        -------
        SmoothManifold
            An exotic sphere, or None if not constructible
        """
        # Check if exotic spheres exist in this dimension
        exotic_info = self.exotic_spheres()
        n_exotic = exotic_info['number_of_exotic_spheres']
        
        if n_exotic == 0 or n_exotic == "Unknown":
            return None
        
        if isinstance(n_exotic, int) and index > n_exotic:
            return None
        
        # Create a representative exotic sphere
        exotic = SmoothManifold(self.dimension, f"Exotic {self.dimension}-sphere (#{index})")
        
        # Add construction information
        if self.dimension == 7:
            # 7-dimensional exotic spheres can be constructed via Milnor's construction
            exotic._additional_invariants["construction"] = "Milnor's exotic 7-sphere via S³ bundles over S⁴"
        elif self.dimension in [8, 9, 10, 11]:
            # Higher dimensional exotic spheres via plumbing and surgery
            exotic._additional_invariants["construction"] = "Constructed via plumbing and surgery"
        else:
            exotic._additional_invariants["construction"] = "Theoretical exotic sphere"
        
        # Set properties that are the same as standard sphere
        exotic._euler_characteristic = 2 if self.dimension % 2 == 0 else 0
        exotic._orientable = True
        
        return exotic
    
    def classify_by_homotopy_type(self):
        """
        Classify the manifold by homotopy type.
        
        Returns:
        -------
        str
            Homotopy classification information
        """
        # In high dimensions, manifolds with the same homotopy type
        # but different smooth structures exist
        
        if 'sphere' in self.name.lower() or 'S^' in self.name:
            exotic_info = self.exotic_spheres()
            n_exotic = exotic_info['number_of_exotic_spheres']
            
            if n_exotic == 0:
                return f"Homotopy {self.dimension}-sphere ⇒ Standard {self.dimension}-sphere (unique smooth structure)"
            elif isinstance(n_exotic, int):
                return f"Homotopy {self.dimension}-sphere ⇒ One of {n_exotic+1} possible smooth structures"
            else:
                return f"Homotopy {self.dimension}-sphere ⇒ Multiple possible smooth structures"
            
        elif 'torus' in self.name.lower() or 'T^' in self.name:
            return f"Homotopy {self.dimension}-torus ⇒ Unique smooth structure (by surgery theory)"
            
        else:
            return f"Homotopy classification requires detailed surgery theory analysis"


#############################################
# PART 6: EXAMPLES AND VISUALIZATION
#############################################

def main():
    """Main function to run examples and tests."""
    print("SMOOTH MANIFOLD CLASSIFICATION AND INVARIANTS\n")
    
    # Create some example manifolds
    print("Creating example manifolds...")
    sphere2 = SmoothManifold(2, "S^2")
    torus2 = SmoothManifold(2, "T^2")
    rp2 = SmoothManifold(2, "RP^2")
    sphere3 = SmoothManifold(3, "S^3")
    cp2 = SmoothFourManifold("CP^2")
    k3 = SmoothFourManifold("K3")
    sphere7 = HighDimensionalManifold(7, "S^7")
    
    # Compute and display invariants
    print("\nComputing topological invariants...\n")
    
    manifolds = [sphere2, torus2, rp2, sphere3, cp2, k3, sphere7]
    
    for manifold in manifolds:
        print(f"Manifold: {manifold.name}")
        print(f"  Dimension: {manifold.dimension}")
        print(f"  Orientable: {manifold.is_orientable()}")
        print(f"  Euler characteristic: {manifold.euler_characteristic()}")
        
        # Display homology if available
        homology = manifold.compute_homology()
        if homology:
            print("  Homology groups:")
            for k, (rank, torsion) in sorted(homology.items()):
                torsion_str = f" ⊕ {' ⊕ '.join([f'Z_{t}' for t in torsion])}" if torsion else ""
                if rank > 0:
                    print(f"    H_{k} = Z^{rank}{torsion_str}")
                elif torsion:
                    print(f"    H_{k} = {torsion_str[3:]}")  # Remove initial " ⊕ "
                else:
                    print(f"    H_{k} = 0")
        
        # Display signature for 4k-manifolds
        signature = manifold.signature()
        if signature is not None:
            print(f"  Signature: {signature}")
        
        # Display characteristic classes
        char_classes = manifold.compute_characteristic_classes()
        if char_classes:
            print("  Characteristic classes: Available")
        
        # Classification
        print(f"  Classification: {manifold.classify()}")
        print()
    
    # Show Morse theory analysis
    print("\nMorse Theory Analysis:\n")
    
    for manifold in [sphere2, torus2]:
        print(f"Manifold: {manifold.name}")
        morse_fn = MorseTheory.morse_function(manifold)
        critical_points = MorseTheory.find_critical_points(morse_fn, manifold)
        
        print("  Critical points:")
        for cp in critical_points:
            print(f"    Index {cp['index']} at {cp['point']} (value: {cp['value']})")
        
        # Handle decomposition
        decomp = MorseTheory.handle_decomposition(manifold, critical_points)
        print(f"  Handle decomposition: {decomp['description']}")
        print()
    
    # Surgery examples
    print("\nSurgery Theory Examples:\n")
    
    # Create a 3-manifold via Dehn surgery
    class Knot:
        def __init__(self, name, ambient_manifold=None):
            self.name = name
            self.ambient_manifold = ambient_manifold
    
    unknot = Knot("Unknot", sphere3)
    trefoil = Knot("Trefoil", sphere3)
    
    print("Dehn surgery on unknot with various framings:")
    for framing in [0, 1, -1, 2, -2]:
        result = SurgeryTheory.dehn_surgery(unknot, framing)
        print(f"  Framing {framing} → {result.name}")
    
    print("\nDehn surgery on trefoil:")
    for framing in [0, 1]:
        result = SurgeryTheory.dehn_surgery(trefoil, framing)
        print(f"  Framing {framing} → {result.name}")
    
    # 4-manifold examples
    print("\nSmooth 4-Manifold Theory:\n")
    
    for manifold in [cp2, k3]:
        print(f"4-Manifold: {manifold.name}")
        
        # Intersection form
        int_form, form_type = manifold.intersection_form()
        print(f"  Intersection form type: {form_type}")
        
        # Wall's classification
        wall_class = manifold.classify_by_wall_theorem()
        print(f"  Wall's classification: {wall_class}")
        
        # Kirby diagram
        kirby = manifold.kirby_diagram()
        print(f"  Kirby diagram: {kirby['description']}")
        
        # Check for exotic counterparts
        exotic = manifold.create_exotic_pair()
        if exotic:
            print(f"  Exotic version: {exotic.name}")
            print(f"    Description: {exotic._additional_invariants.get('description', 'No description')}")
        else:
            print("  No known exotic version")
        
        print()
    
    # High-dimensional examples
    print("\nHigh-Dimensional Manifold Theory:\n")
    
    for dim in [5, 6, 7, 8, 9, 10, 11]:
        high_sphere = HighDimensionalManifold(dim, f"S^{dim}")
        exotic_info = high_sphere.exotic_spheres()
        
        print(f"{dim}-Sphere: {high_sphere.name}")
        print(f"  Exotic spheres: {exotic_info['description']}")
        
        # Try to construct an exotic sphere
        if exotic_info['number_of_exotic_spheres'] not in [0, "Unknown"]:
            exotic = high_sphere.construct_exotic_sphere()
            if exotic:
                print(f"  Example exotic: {exotic.name}")
                print(f"    Construction: {exotic._additional_invariants.get('construction', 'Unknown')}")
        
        print()
    
    # Visualizations
    print("\nVisualizing Examples:\n")
    
    print("1. Visualizing 2-manifolds")
    for manifold in [sphere2, torus2, rp2]:
        print(f"  Visualizing {manifold.name}...")
        manifold.visualize()
    
    print("\n2. Visualizing Morse flow on torus")
    MorseTheory.visualize_handle_decomposition(torus2)
    
    print("\n3. Visualizing Dehn surgery")
    SurgeryTheory.visualize_surgery(sphere3, {'type': 'Dehn', 'knot': 'unknot', 'framing': 1})
    
    print("\n4. Visualizing Kirby diagram for CP²")
    cp2.visualize_kirby_diagram()

    print("\nClassification and Invariant Computation Complete!")


if __name__ == "__main__":
    main()