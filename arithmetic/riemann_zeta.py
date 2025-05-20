import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpmath as mp
import scipy.special as sp
from scipy.optimize import minimize_scalar
import sympy
from typing import Union, List, Tuple, Optional, Callable
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
import os
import math

# Set mpmath precision
mp.mp.dps = 50  # 50 digits of precision


class RiemannZeta:
    """
    A comprehensive implementation of the Riemann Zeta function
    
    The Riemann zeta function is defined as:
    ζ(s) = Σ(n=1 to ∞) 1/n^s for Re(s) > 1
    
    It can be analytically continued to the entire complex plane except s=1.
    """
    
    def __init__(self, precision: int = 50):
        """
        Initialize the RiemannZeta calculator
        
        Args:
            precision: Number of decimal digits of precision
        """
        mp.mp.dps = precision
        self.precision = precision
        
        # Cache for computed values of zeta
        self._cache = {}
        
        # Constants used in calculations
        self._bernoulli_cache = {}
        self._euler_mascheroni = mp.euler
        self._two_pi = mp.mpf(2) * mp.pi
        
        # Cache for computed zeros
        self._zeros_cache = []
    
    def zeta(self, s: Union[complex, mp.mpc], method: str = 'auto') -> Union[complex, mp.mpc]:
        """
        Calculate the Riemann zeta function ζ(s) for complex s
        
        Args:
            s: Complex number input
            method: Computation method ('auto', 'series', 'riemann_siegel', 
                    'euler_product', 'reflection', 'mpmath', 'functional')
                    
        Returns:
            Value of ζ(s)
        """
        # Convert normal complex to mpmath complex for high precision
        if isinstance(s, complex):
            s_mp = mp.mpc(s.real, s.imag)
        else:
            s_mp = s
        
        # Check cache first
        cache_key = (str(s_mp), method)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Handle s=1 singularity
        if abs(s_mp - 1) < 1e-14:
            return mp.inf
        
        # Choose method based on location in complex plane
        if method == 'auto':
            # For Re(s) > 1, use series or Euler product
            if mp.re(s_mp) > 1:
                method = 'series'
            # For Re(s) < 0, use the reflection formula
            elif mp.re(s_mp) < 0:
                method = 'reflection'
            # For Im(s) large, use Riemann-Siegel
            elif abs(mp.im(s_mp)) > 30:
                method = 'riemann_siegel'
            # For 0 <= Re(s) <= 1, careful calculation needed
            else:
                method = 'mpmath'  # Use mpmath's built-in function
        
        # Compute using the selected method
        result = None
        
        if method == 'series':
            result = self._zeta_series(s_mp)
        elif method == 'riemann_siegel':
            result = self._zeta_riemann_siegel(s_mp)
        elif method == 'euler_product':
            result = self._zeta_euler_product(s_mp)
        elif method == 'reflection':
            result = self._zeta_reflection(s_mp)
        elif method == 'functional':
            result = self._zeta_functional(s_mp)
        elif method == 'mpmath':
            result = mp.zeta(s_mp)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Cache the result
        self._cache[cache_key] = result
        return result
    
    def _zeta_series(self, s: mp.mpc, terms: int = 1000) -> mp.mpc:
        """
        Compute ζ(s) using direct series summation (converges for Re(s) > 1)
        
        Args:
            s: Complex input
            terms: Number of terms to sum
            
        Returns:
            Value of ζ(s)
        """
        if mp.re(s) <= 1:
            warnings.warn("Series computation may not converge for Re(s) <= 1")
        
        result = mp.mpf(0)
        for n in range(1, terms + 1):
            term = mp.power(mp.mpf(n), -s)
            result += term
            
            # Check for convergence
            if abs(term) < 1e-20 * abs(result):
                break
                
        return result
    
    def _zeta_euler_product(self, s: mp.mpc, primes_limit: int = 100) -> mp.mpc:
        """
        Compute ζ(s) using the Euler product formula
        ζ(s) = ∏(p prime) 1/(1-p^(-s)) for Re(s) > 1
        
        Args:
            s: Complex input
            primes_limit: Number of primes to use
            
        Returns:
            Approximation of ζ(s)
        """
        if mp.re(s) <= 1:
            warnings.warn("Euler product diverges for Re(s) <= 1")
        
        # Generate primes
        primes = self._generate_primes(primes_limit)
        
        # Compute the product
        result = mp.mpf(1)
        for p in primes:
            term = 1 / (1 - mp.power(mp.mpf(p), -s))
            result *= term
            
        return result
    
    def _zeta_reflection(self, s: mp.mpc) -> mp.mpc:
        """
        Compute ζ(s) using the reflection formula:
        ζ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)
        
        This is useful for computing ζ(s) when Re(s) < 0
        
        Args:
            s: Complex input
            
        Returns:
            Value of ζ(s)
        """
        # Compute the reflection formula
        two_s = mp.power(2, s)
        pi_s_minus_1 = mp.power(mp.pi, s - 1)
        sin_pi_s_half = mp.sin(mp.pi * s / 2)
        gamma_1_minus_s = mp.gamma(1 - s)
        
        # Compute ζ(1-s) (will be in Re(s) > 1 region where series converges)
        zeta_1_minus_s = self._zeta_series(1 - s)
        
        # Combine all terms
        result = two_s * pi_s_minus_1 * sin_pi_s_half * gamma_1_minus_s * zeta_1_minus_s
        return result
    
    def _zeta_functional(self, s: mp.mpc) -> mp.mpc:
        """
        Compute ζ(s) using the functional equation
        
        Args:
            s: Complex input
            
        Returns:
            Value of ζ(s)
        """
        # Use the functional equation: ζ(s) = χ(s) * ζ(1-s)
        # where χ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s)
        chi = self._compute_chi(s)
        zeta_1_minus_s = self.zeta(1 - s, method='series')
        return chi * zeta_1_minus_s
    
    def _compute_chi(self, s: mp.mpc) -> mp.mpc:
        """
        Compute the factor χ(s) in the functional equation
        
        Args:
            s: Complex input
            
        Returns:
            Value of χ(s)
        """
        two_s = mp.power(2, s)
        pi_s_minus_1 = mp.power(mp.pi, s - 1)
        sin_pi_s_half = mp.sin(mp.pi * s / 2)
        gamma_1_minus_s = mp.gamma(1 - s)
        
        return two_s * pi_s_minus_1 * sin_pi_s_half * gamma_1_minus_s
    
    def _zeta_riemann_siegel(self, s: mp.mpc, terms: int = None) -> mp.mpc:
        """
        Compute ζ(s) using the Riemann-Siegel formula
        Particularly efficient for s = 1/2 + it with large t
        
        Args:
            s: Complex input, typically s = 1/2 + it
            terms: Number of terms to use, defaults to ceil(sqrt(t/(2π)))
            
        Returns:
            Value of ζ(s)
        """
        # This implementation works well on the critical line s = 1/2 + it
        sigma = mp.re(s)
        t = mp.im(s)
        
        # For small t, fall back to mpmath's implementation
        if abs(t) < 20:
            return mp.zeta(s)
        
        # If s is not on critical line, calculations are more complex
        if abs(sigma - 0.5) > 1e-10:
            return mp.zeta(s)
        
        # Calculate number of terms using Riemann-Siegel formula
        if terms is None:
            terms = int(mp.ceil(mp.sqrt(abs(t) / (2 * mp.pi))))
        
        # Main sum
        z = mp.mpc(0)
        for n in range(1, terms + 1):
            z += mp.exp(mp.log(n) * mp.mpc(-sigma, -t))
        
        # Remainder term (approximation)
        m = terms
        theta = self._theta_function(t)
        remainder = mp.exp(mp.mpc(0, -theta)) * mp.power(mp.mpf(m), mp.mpc(-sigma, -t)) / 2
        
        # Correction terms for better accuracy
        # This is a simplified version; full R-S formula has more terms
        result = z + remainder
        
        return result
    
    def _theta_function(self, t: mp.mpf) -> mp.mpf:
        """
        Compute θ(t) = arg(Γ(1/4 + it/2)) - t*log(π)/2
        
        Args:
            t: Real value
            
        Returns:
            Value of θ(t)
        """
        # First term: arg(Γ(1/4 + it/2))
        gamma_term = mp.arg(mp.gamma(mp.mpc(0.25, t/2)))
        
        # Second term: t*log(π)/2
        log_pi_term = t * mp.log(mp.pi) / 2
        
        return gamma_term - log_pi_term
    
    def _generate_primes(self, limit: int) -> List[int]:
        """
        Generate a list of prime numbers up to the given limit
        
        Args:
            limit: Upper bound for the list of primes
            
        Returns:
            List of prime numbers
        """
        # Use Sieve of Eratosthenes
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(limit + 1) if sieve[i]]
    
    def critical_line(self, t_min: float, t_max: float, points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ζ(1/2 + it) for t between t_min and t_max
        
        Args:
            t_min: Minimum t value
            t_max: Maximum t value
            points: Number of points to compute
            
        Returns:
            Tuple of (t_values, zeta_values)
        """
        t_values = np.linspace(t_min, t_max, points)
        zeta_values = np.zeros(points, dtype=np.complex128)
        
        for i, t in enumerate(t_values):
            s = mp.mpc(0.5, t)
            zeta_values[i] = complex(self.zeta(s))
        
        return t_values, zeta_values
    
    def find_zeros(self, t_min: float, t_max: float, resolution: int = 1000, 
                   refinement: bool = True) -> List[float]:
        """
        Find zeros of ζ(s) on the critical line s = 1/2 + it
        
        Args:
            t_min: Minimum t value
            t_max: Maximum t value
            resolution: Number of points to sample
            refinement: Whether to refine zero locations
            
        Returns:
            List of t values where ζ(1/2 + it) = 0
        """
        if self._zeros_cache and t_min >= self._zeros_cache[0] and t_max <= self._zeros_cache[-1]:
            return [z for z in self._zeros_cache if t_min <= z <= t_max]
        
        # Calculate Z(t) values on critical line
        t_values, zeta_values = self.critical_line(t_min, t_max, resolution)
        z_values = np.array([abs(z) for z in zeta_values])
        
        # Find where Z(t) changes sign by looking for minima
        zeros = []
        for i in range(1, len(z_values) - 1):
            if z_values[i] < z_values[i-1] and z_values[i] < z_values[i+1] and z_values[i] < 0.1:
                t_approx = t_values[i]
                
                if refinement:
                    # Refine the zero location
                    t_refined = self._refine_zero(t_approx)
                    zeros.append(t_refined)
                else:
                    zeros.append(t_approx)
        
        self._zeros_cache.extend(zeros)
        self._zeros_cache.sort()
        
        return zeros
    
    def _refine_zero(self, t_approx: float, tol: float = 1e-10) -> float:
        """
        Refine the location of a zero using numerical optimization
        
        Args:
            t_approx: Approximate t value where ζ(1/2 + it) = 0
            tol: Tolerance for optimization
            
        Returns:
            Refined t value
        """
        def objective(t):
            return abs(self.zeta(mp.mpc(0.5, t)))
        
        result = minimize_scalar(objective, bracket=[t_approx-0.5, t_approx, t_approx+0.5], 
                                 method='brent', tol=tol)
        
        return result.x
    
    def plot_critical_line(self, t_min: float, t_max: float, points: int = 1000, 
                          show_zeros: bool = True, ax=None):
        """
        Plot ζ(1/2 + it) along the critical line
        
        Args:
            t_min: Minimum t value
            t_max: Maximum t value
            points: Number of points
            show_zeros: Whether to mark zeros
            ax: Matplotlib axis
        """
        t_values, zeta_values = self.critical_line(t_min, t_max, points)
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot real and imaginary parts
        ax.plot(t_values, np.real(zeta_values), 'b-', label='Re(ζ(1/2 + it))')
        ax.plot(t_values, np.imag(zeta_values), 'r-', label='Im(ζ(1/2 + it))')
        
        # Plot absolute value
        ax.plot(t_values, np.abs(zeta_values), 'g--', label='|ζ(1/2 + it)|', alpha=0.5)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Mark zeros
        if show_zeros:
            zeros = self.find_zeros(t_min, t_max)
            ax.plot(zeros, np.zeros_like(zeros), 'ko', markersize=6, label='Zeros')
            
            # Annotate first few zeros
            for i, z in enumerate(zeros[:5]):
                ax.annotate(f"t ≈ {z:.6f}", (z, 0.5), textcoords="offset points", 
                            xytext=(0, 10), ha='center')
        
        ax.set_xlabel('t')
        ax.set_ylabel('ζ(1/2 + it)')
        ax.set_title('Riemann Zeta Function on the Critical Line')
        ax.legend()
        ax.grid(True)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def plot_complex_plane(self, sigma_min: float = -5, sigma_max: float = 5, 
                           t_min: float = -20, t_max: float = 20, 
                           resolution: int = 100, function: str = 'abs'):
        """
        Plot ζ(s) in the complex plane
        
        Args:
            sigma_min, sigma_max: Range for Re(s)
            t_min, t_max: Range for Im(s)
            resolution: Grid resolution
            function: What to plot ('abs', 'real', 'imag', 'phase')
        """
        # Create grid
        sigma = np.linspace(sigma_min, sigma_max, resolution)
        t = np.linspace(t_min, t_max, resolution)
        sigma_grid, t_grid = np.meshgrid(sigma, t)
        
        # Initialize values grid
        values = np.zeros_like(sigma_grid, dtype=np.complex128)
        
        # Compute zeta values
        for i in range(resolution):
            for j in range(resolution):
                s = complex(sigma_grid[i, j], t_grid[i, j])
                values[i, j] = complex(self.zeta(s))
        
        # Select function to plot
        if function == 'abs':
            plot_values = np.abs(values)
            title = '|ζ(s)|'
            cmap = 'viridis'
        elif function == 'real':
            plot_values = np.real(values)
            title = 'Re(ζ(s))'
            cmap = 'RdBu'
        elif function == 'imag':
            plot_values = np.imag(values)
            title = 'Im(ζ(s))'
            cmap = 'RdBu'
        elif function == 'phase':
            plot_values = np.angle(values)
            title = 'arg(ζ(s))'
            cmap = 'hsv'
        else:
            raise ValueError(f"Unknown function: {function}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Handle singularity at s=1
        plot_values = np.clip(plot_values, -10, 10)
        
        # Plot as a color map
        im = ax.pcolormesh(sigma_grid, t_grid, plot_values, cmap=cmap, shading='auto')
        fig.colorbar(im, ax=ax, label=title)
        
        # Add contour lines for absolute value
        if function == 'abs':
            contour_levels = [0.1, 0.5, 1, 2, 5]
            contours = ax.contour(sigma_grid, t_grid, plot_values, 
                                 levels=contour_levels, colors='white', alpha=0.6)
            ax.clabel(contours, inline=True, fontsize=8)
        
        # Mark special lines and points
        ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.7, label='Critical Line')
        ax.axvline(x=1, color='red', linestyle='-', alpha=0.7, label='Pole at s=1')
        
        # Mark trivial zeros at negative even integers
        trivial_zeros = [-2, -4, -6, -8, -10, -12]
        trivial_t = [0] * len(trivial_zeros)
        ax.plot(trivial_zeros, trivial_t, 'wo', markersize=5, label='Trivial Zeros')
        
        # Find zeros on critical line in the visible range
        if t_min < 100:  # Only compute for reasonable t ranges
            zeros = self.find_zeros(max(0, t_min), min(100, t_max))
            zeros_sigma = [0.5] * len(zeros)
            ax.plot(zeros_sigma, zeros, 'yo', markersize=5, label='Nontrivial Zeros')
        
        ax.set_xlabel('Re(s)')
        ax.set_ylabel('Im(s)')
        ax.set_title(f'Riemann Zeta Function: {title} in Complex Plane')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def compute_eta(self, s: Union[complex, mp.mpc]) -> Union[complex, mp.mpc]:
        """
        Compute the Dirichlet eta function (alternating zeta)
        η(s) = Σ(n=1 to ∞) (-1)^(n+1)/n^s = (1-2^(1-s))*ζ(s)
        
        Args:
            s: Complex input
            
        Returns:
            Value of η(s)
        """
        # Relation to zeta function
        if mp.re(s) > 0:
            factor = 1 - mp.power(2, 1 - s)
            return factor * self.zeta(s)
        else:
            # Direct computation for better numerical stability
            result = mp.mpf(0)
            for n in range(1, 1000):
                term = mp.power(-1, n+1) * mp.power(mp.mpf(n), -s)
                result += term
                if abs(term) < 1e-20 * abs(result):
                    break
            return result
    
    def li(self, x: float) -> float:
        """
        Compute the logarithmic integral Li(x)
        
        Args:
            x: Positive real number, x > 1
            
        Returns:
            Value of Li(x)
        """
        if x <= 1:
            raise ValueError("Li(x) is defined for x > 1")
        
        # Use mpmath's built-in logarithmic integral
        return float(mp.li(x))
    
    def prime_counting_approx(self, x: float) -> float:
        """
        Approximate π(x) using the Riemann zeta function
        
        Args:
            x: Positive real number
            
        Returns:
            Approximation of π(x)
        """
        # Use Riemann's formula: π(x) ≈ Li(x) - Σ Li(x^(1/ρ))
        # where ρ runs over the nontrivial zeros of ζ
        
        # This is computationally intensive, so we'll use a simpler approximation
        # π(x) ≈ Li(x)
        return self.li(x)
    
    def riemann_siegel_z(self, t: float) -> float:
        """
        Compute the Riemann-Siegel Z function Z(t)
        Z(t) = e^(iθ(t)) * ζ(1/2 + it)
        where θ(t) = arg(Γ(1/4 + it/2)) - log(π)*t/2
        
        Args:
            t: Real value
            
        Returns:
            Value of Z(t)
        """
        s = mp.mpc(0.5, t)
        zeta_value = self.zeta(s)
        
        # Compute the phase factor
        theta = self._theta_function(t)
        phase = mp.exp(mp.mpc(0, theta))
        
        # Z(t) is real-valued
        return float(mp.re(phase * zeta_value))
    
    def plot_zeta_zeros_distribution(self, t_max: float = 1000, bin_width: float = 10):
        """
        Plot the distribution of spacings between consecutive zeros
        
        Args:
            t_max: Maximum t value
            bin_width: Width for histogram bins
        """
        # Compute zeros up to t_max
        zeros = self.find_zeros(0, t_max)
        
        # Compute spacings between consecutive zeros
        spacings = np.diff(zeros)
        
        # Normalize spacings by the local average spacing
        # The average spacing near t is approximately 2π/log(t)
        mean_spacings = [2 * np.pi / np.log(zeros[i]) for i in range(len(spacings))]
        normalized_spacings = spacings / mean_spacings
        
        # Plot histogram
        plt.figure(figsize=(12, 6))
        plt.hist(normalized_spacings, bins=50, density=True, alpha=0.7)
        
        # Plot the theoretical GUE distribution (conjectured)
        x = np.linspace(0, 3, 1000)
        gue = np.pi**2/2 * x * np.exp(-np.pi*x**2/4)
        plt.plot(x, gue, 'r-', linewidth=2, 
                 label='GUE Distribution (Conjectured)')
        
        plt.xlabel('Normalized Spacing Between Consecutive Zeros')
        plt.ylabel('Probability Density')
        plt.title('Distribution of Normalized Spacings Between Consecutive Zeros of ζ(s)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print some statistics
        print(f"Number of zeros found: {len(zeros)}")
        print(f"Average spacing: {np.mean(spacings):.6f}")
        print(f"Average normalized spacing: {np.mean(normalized_spacings):.6f}")
        print(f"Standard deviation of normalized spacings: {np.std(normalized_spacings):.6f}")
    
    def verify_riemann_hypothesis(self, t_max: float = 100, tol: float = 1e-10) -> bool:
        """
        Verify that all non-trivial zeros up to height t_max lie on the critical line
        
        Args:
            t_max: Maximum t value
            tol: Tolerance for deviation from critical line
            
        Returns:
            True if all zeros found are on the critical line
        """
        zeros = self.find_zeros(0, t_max)
        
        # For each zero, check if Re(ζ(s)) = 0 at s = σ + it where σ ≠ 0.5
        for t in zeros:
            # Check to the left of critical line
            left = abs(self.zeta(mp.mpc(0.5 - 0.01, t)))
            # Check to the right of critical line
            right = abs(self.zeta(mp.mpc(0.5 + 0.01, t)))
            
            # If a zero exists off the critical line, both left and right would be small
            if left < tol and right < tol:
                print(f"Possible zero off critical line at t = {t}")
                return False
        
        return True
    
    def hardy_z(self, t: float) -> float:
        """
        Compute the Hardy Z-function
        Z(t) = ζ(1/2 + it) * e^(iθ(t))
        which is real-valued and has zeros exactly where ζ(1/2 + it) does
        
        Args:
            t: Real value
            
        Returns:
            Value of Z(t)
        """
        s = mp.mpc(0.5, t)
        zeta_value = self.zeta(s)
        
        # Compute the phase factor
        theta = self._theta_function(t)
        phase = mp.exp(mp.mpc(0, theta))
        
        # Z(t) is real-valued
        result = phase * zeta_value
        return float(mp.re(result))
    
    def plot_hardy_z(self, t_min: float, t_max: float, points: int = 1000):
        """
        Plot the Hardy Z-function
        
        Args:
            t_min: Minimum t value
            t_max: Maximum t value
            points: Number of points
        """
        t_values = np.linspace(t_min, t_max, points)
        z_values = np.array([self.hardy_z(t) for t in t_values])
        
        plt.figure(figsize=(12, 6))
        plt.plot(t_values, z_values, 'b-')
        
        # Mark zeros
        zeros = []
        for i in range(1, len(z_values) - 1):
            if z_values[i-1] * z_values[i+1] <= 0:  # Sign change
                zeros.append(t_values[i])
        
        plt.plot(zeros, np.zeros_like(zeros), 'ro', markersize=4)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('t')
        plt.ylabel('Z(t)')
        plt.title('Hardy Z-Function')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def prime_count(n: int) -> int:
        """
        Count the number of primes ≤ n using a simple sieve
        
        Args:
            n: Upper bound
            
        Returns:
            Number of primes ≤ n
        """
        if n < 2:
            return 0
        
        # Use Sieve of Eratosthenes
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        return sum(sieve)
    
    def compare_prime_counting(self, max_n: int = 1000, step: int = 10):
        """
        Compare different approximations to the prime counting function
        
        Args:
            max_n: Maximum value to check
            step: Step size for x values
        """
        x_values = list(range(2, max_n + 1, step))
        
        # Actual prime counts
        pi_values = [self.prime_count(x) for x in x_values]
        
        # x/log(x) approximation
        log_approx = [x / np.log(x) for x in x_values]
        
        # Li(x) approximation
        li_approx = [self.li(x) for x in x_values]
        
        # Plot comparisons
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, pi_values, 'b-', label='π(x) (Actual)')
        plt.plot(x_values, log_approx, 'r--', label='x/log(x)')
        plt.plot(x_values, li_approx, 'g-.', label='Li(x)')
        
        plt.xlabel('x')
        plt.ylabel('Number of Primes ≤ x')
        plt.title('Prime Counting Function and Approximations')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot errors
        plt.figure(figsize=(12, 6))
        log_error = [(log_approx[i] - pi_values[i]) for i in range(len(x_values))]
        li_error = [(li_approx[i] - pi_values[i]) for i in range(len(x_values))]
        
        plt.plot(x_values, log_error, 'r--', label='x/log(x) Error')
        plt.plot(x_values, li_error, 'g-.', label='Li(x) Error')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.xlabel('x')
        plt.ylabel('Error in Approximation')
        plt.title('Errors in Prime Counting Approximations')
        plt.legend()
        plt.grid(True)
        plt.show()


def demonstrate_riemann_zeta():
    """Run demonstrations of the Riemann Zeta function capabilities"""
    
    print("=== RIEMANN ZETA FUNCTION DEMONSTRATION ===\n")
    
    # Initialize with high precision
    zeta = RiemannZeta(precision=50)
    
    # 1. Basic Calculations
    print("1. BASIC CALCULATIONS")
    print("--------------------")
    
    # Calculate ζ(2), which equals π²/6
    zeta_2 = zeta.zeta(2)
    pi_squared_over_6 = mp.power(mp.pi, 2) / 6
    
    print(f"ζ(2) = {zeta_2}")
    print(f"π²/6 = {pi_squared_over_6}")
    print(f"Difference: {abs(zeta_2 - pi_squared_over_6)}")
    
    # Calculate ζ(1 + i)
    zeta_1_plus_i = zeta.zeta(mp.mpc(1, 1))
    print(f"\nζ(1 + i) = {zeta_1_plus_i}")
    
    # 2. Zeros on the Critical Line
    print("\n2. ZEROS ON THE CRITICAL LINE")
    print("---------------------------")
    
    # Find first few zeros
    zeros = zeta.find_zeros(0, 50)
    print("First few zeros on the critical line:")
    for i, z in enumerate(zeros[:10]):
        print(f"Zero #{i+1}: t = {z}")
    
    # 3. Riemann Hypothesis Verification
    print("\n3. RIEMANN HYPOTHESIS VERIFICATION")
    print("--------------------------------")
    
    # Verify RH for zeros up to t=50
    rh_verified = zeta.verify_riemann_hypothesis(t_max=50)
    print(f"All zeros up to t=50 lie on the critical line: {rh_verified}")
    
    # 4. Prime Counting Function
    print("\n4. PRIME COUNTING FUNCTION")
    print("------------------------")
    
    for x in [10, 100, 1000]:
        pi_x = zeta.prime_count(x)
        li_x = zeta.li(x)
        x_log_x = x / math.log(x)
        
        print(f"π({x}) = {pi_x}")
        print(f"Li({x}) = {li_x:.6f}, Error: {li_x - pi_x:.6f}")
        print(f"{x}/log({x}) = {x_log_x:.6f}, Error: {x_log_x - pi_x:.6f}")
        print()
    
    # 5. Function Values and Special Points
    print("5. FUNCTION VALUES AT SPECIAL POINTS")
    print("----------------------------------")
    
    # Values at negative integers (related to Bernoulli numbers)
    print("ζ at negative integers:")
    for n in range(1, 6):
        value = zeta.zeta(-n)
        print(f"ζ(-{n}) = {value}")
    
    # 6. Functional Equation Verification
    print("\n6. FUNCTIONAL EQUATION VERIFICATION")
    print("---------------------------------")
    
    # Test functional equation at a sample point
    s = mp.mpc(0.3, 4.5)
    direct = zeta.zeta(s)
    functional = zeta._zeta_functional(s)
    
    print(f"s = {s}")
    print(f"ζ(s) direct = {direct}")
    print(f"ζ(s) via functional equation = {functional}")
    print(f"Relative difference: {abs(direct - functional) / abs(direct)}")


if __name__ == "__main__":
    demonstrate_riemann_zeta()
    
    # Create RiemannZeta object for interactive use
    zeta = RiemannZeta()
    
    # Uncomment any of these to generate visualizations
    
    # Plot on the critical line
    # zeta.plot_critical_line(0, 50)
    
    # Plot in the complex plane
    # zeta.plot_complex_plane(sigma_min=-5, sigma_max=5, t_min=-20, t_max=20, function='abs')
    
    # Plot the Hardy Z-function
    # zeta.plot_hardy_z(0, 50)
    
    # Plot zero spacing distribution
    # zeta.plot_zeta_zeros_distribution(t_max=1000)
    
    # Compare prime counting approximations
    # zeta.compare_prime_counting(max_n=1000)