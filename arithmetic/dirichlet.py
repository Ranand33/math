import numpy as np
import matplotlib.pyplot as plt
import sympy
import math
from typing import List, Dict, Tuple, Set, Optional, Union, Callable
from functools import lru_cache
import mpmath as mp
from collections import defaultdict
import cmath


class DirichletTheorem:
    """
    Implementation of Dirichlet's theorem on arithmetic progressions,
    which states that for any coprime integers a and m, there are
    infinitely many primes in the arithmetic progression a + km.
    """
    
    def __init__(self, precision: int = 50):
        """
        Initialize with a given precision for numerical calculations
        
        Args:
            precision: Number of digits of precision for mpmath
        """
        mp.mp.dps = precision
        self.precision = precision
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """
        Check if a number is prime
        
        Args:
            n: The number to check
            
        Returns:
            True if n is prime, False otherwise
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        
        # Check divisibility by numbers of form 6k ± 1
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
            
        return True
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """
        Calculate the greatest common divisor of a and b
        
        Args:
            a, b: Integers
            
        Returns:
            The GCD of a and b
        """
        while b:
            a, b = b, a % b
        return abs(a)
    
    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm
        
        Args:
            a, b: Integers
            
        Returns:
            (gcd, x, y) such that gcd = a*x + b*y
        """
        if a == 0:
            return (b, 0, 1)
        
        gcd, x1, y1 = DirichletTheorem.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return (gcd, x, y)
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """
        Calculate the modular multiplicative inverse of a modulo m
        
        Args:
            a: Integer
            m: Modulus
            
        Returns:
            Integer b such that (a * b) % m = 1
        """
        gcd, x, y = DirichletTheorem.extended_gcd(a, m)
        if gcd != 1:
            raise ValueError(f"Modular inverse does not exist (gcd({a}, {m}) ≠ 1)")
        else:
            return x % m
    
    @staticmethod
    def euler_phi(n: int) -> int:
        """
        Calculate Euler's totient function φ(n)
        
        Args:
            n: Positive integer
            
        Returns:
            Number of integers 1 ≤ k ≤ n coprime to n
        """
        if n <= 0:
            raise ValueError("Input must be a positive integer")
        
        # Special cases
        if n == 1:
            return 1
        
        # Initialize result
        result = n
        
        # Consider all prime factors
        p = 2
        while p * p <= n:
            # Check if p is a prime factor
            if n % p == 0:
                # Subtract multiples of p from result
                result -= result // p
                
                # Remove all factors of p from n
                while n % p == 0:
                    n //= p
            p += 1
        
        # If n has a prime factor greater than sqrt(n)
        if n > 1:
            result -= result // n
            
        return result
    
    @staticmethod
    def generate_primes(limit: int) -> List[int]:
        """
        Generate all primes up to a given limit using the Sieve of Eratosthenes
        
        Args:
            limit: Upper bound
            
        Returns:
            List of primes up to limit
        """
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(limit + 1) if sieve[i]]
    
    def primes_in_progression(self, a: int, m: int, limit: int) -> List[int]:
        """
        Find primes in the arithmetic progression a + km up to a limit
        
        Args:
            a: First term of progression
            m: Common difference
            limit: Upper bound
            
        Returns:
            List of primes in the progression up to limit
        """
        # Ensure a is in the range [0, m-1]
        a = a % m
        
        if self.gcd(a, m) != 1:
            raise ValueError(f"a = {a} and m = {m} must be coprime")
        
        # Find primes in the arithmetic progression
        result = []
        for n in range(limit // m + 1):
            candidate = a + n * m
            if candidate > limit:
                break
            if candidate >= 2 and self.is_prime(candidate):
                result.append(candidate)
                
        return result
    
    def prime_counting_progression(self, a: int, m: int, limit: int) -> Dict[int, int]:
        """
        Count primes in arithmetic progressions a + km for each a coprime to m
        
        Args:
            a: If a >= 0, count primes in a + km. If a < 0, count for all residue classes
            m: Common difference
            limit: Upper bound
            
        Returns:
            Dictionary mapping residue classes to prime counts
        """
        # If a >= 0, only count for one residue class
        if a >= 0:
            a = a % m
            if self.gcd(a, m) != 1:
                raise ValueError(f"a = {a} and m = {m} must be coprime")
            
            residues = [a]
        else:
            # Count for all residue classes coprime to m
            residues = [i for i in range(m) if self.gcd(i, m) == 1]
        
        # Count primes in each residue class
        counts = {}
        for residue in residues:
            counts[residue] = len(self.primes_in_progression(residue, m, limit))
            
        return counts
    
    def prime_race(self, m: int, limit: int) -> Dict[int, List[int]]:
        """
        Track the "race" between different residue classes mod m
        
        Args:
            m: Modulus
            limit: Upper bound for primes to consider
            
        Returns:
            Dictionary mapping residue classes to counts at each milestone
        """
        # Get residue classes coprime to m
        residues = [i for i in range(m) if self.gcd(i, m) == 1]
        
        # Generate primes up to limit
        primes = self.generate_primes(limit)
        
        # Initialize count arrays
        counts = {residue: [0] * len(primes) for residue in residues}
        
        # Count primes in each residue class
        for i, p in enumerate(primes):
            if p >= m:  # Skip small primes less than m
                residue = p % m
                for r in residues:
                    # Copy previous count
                    if i > 0:
                        counts[r][i] = counts[r][i-1]
                    
                    # Increment count for the matching residue
                    if r == residue:
                        counts[r][i] += 1
        
        return counts
    
    def plot_prime_distribution(self, m: int, limit: int):
        """
        Plot the distribution of primes across residue classes mod m
        
        Args:
            m: Modulus
            limit: Upper bound for primes
        """
        counts = self.prime_counting_progression(-1, m, limit)
        residues = sorted(counts.keys())
        
        plt.figure(figsize=(10, 6))
        plt.bar(residues, [counts[r] for r in residues], align='center')
        
        plt.xlabel('Residue class (mod {})'.format(m))
        plt.ylabel('Number of primes ≤ {}'.format(limit))
        plt.title('Distribution of Primes in Arithmetic Progressions mod {}'.format(m))
        
        # Calculate expected count based on Dirichlet's theorem
        phi_m = self.euler_phi(m)
        pi_approx = len(self.generate_primes(limit)) - m + 1  # Approximate π(limit) - π(m-1)
        expected = pi_approx / phi_m
        
        # Add a horizontal line for the expected count
        plt.axhline(y=expected, color='r', linestyle='--', 
                    label=f'Expected asymptotic count: {expected:.1f}')
        
        plt.xticks(residues)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_prime_race(self, m: int, limit: int):
        """
        Plot the "race" between different residue classes mod m
        
        Args:
            m: Modulus
            limit: Upper bound for primes
        """
        race_data = self.prime_race(m, limit)
        primes = self.generate_primes(limit)
        
        plt.figure(figsize=(12, 6))
        
        for residue, counts in race_data.items():
            plt.plot(primes, counts, label=f'a ≡ {residue} (mod {m})')
        
        plt.xlabel('Prime count')
        plt.ylabel('Number of primes in residue class')
        plt.title(f'Prime Race mod {m}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def verify_asymptotic_distribution(self, m: int, max_limit: int, steps: int = 10):
        """
        Verify the asymptotic distribution of primes in arithmetic progressions
        
        Args:
            m: Modulus
            max_limit: Maximum upper bound
            steps: Number of data points to calculate
        """
        limits = [max_limit * (i + 1) // steps for i in range(steps)]
        
        # Get residue classes coprime to m
        residues = [i for i in range(m) if self.gcd(i, m) == 1]
        phi_m = len(residues)
        
        # Calculate prime counts for each limit
        results = []
        for limit in limits:
            counts = self.prime_counting_progression(-1, m, limit)
            # Calculate max deviation from expected distribution
            pi_approx = len(self.generate_primes(limit)) - m + 1
            expected = pi_approx / phi_m
            
            max_dev = max(abs(counts[r] - expected) / expected for r in residues)
            results.append((limit, max_dev))
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot([r[0] for r in results], [r[1] for r in results], 'o-')
        plt.xlabel('Limit')
        plt.ylabel('Maximum relative deviation from expected count')
        plt.title(f'Convergence to Uniform Distribution mod {m}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class DirichletCharacter:
    """
    Implementation of Dirichlet characters, which are key to proving
    Dirichlet's theorem on arithmetic progressions
    """
    
    def __init__(self, m: int):
        """
        Initialize the class for Dirichlet characters modulo m
        
        Args:
            m: Modulus
        """
        self.m = m
        self.phi_m = DirichletTheorem.euler_phi(m)
        self._characters = None
        self._character_values = {}
    
    def generate_characters(self) -> List[Dict[int, complex]]:
        """
        Generate all Dirichlet characters modulo m
        
        Returns:
            List of dictionaries, each representing a character χ
            mapping integers to complex values
        """
        if self._characters is not None:
            return self._characters
        
        # Get residues coprime to m
        residues = [i for i in range(1, self.m) if DirichletTheorem.gcd(i, self.m) == 1]
        
        # For simplicity, we'll first generate the group structure
        # Use the Chinese Remainder Theorem to factor the group (Z/mZ)* into a product of cyclic groups
        
        # Step 1: Factor m into prime powers
        m = self.m
        prime_powers = []
        for p in range(2, m + 1):
            if not DirichletTheorem.is_prime(p):
                continue
            
            power = 0
            while m % p == 0:
                m //= p
                power += 1
            
            if power > 0:
                prime_powers.append((p, power))
        
        # If m was 1, add it as a special case
        if not prime_powers:
            prime_powers = [(1, 1)]
        
        # Step 2: For each prime power p^k, find generators of the group (Z/p^kZ)*
        generators = []
        for p, k in prime_powers:
            if p == 2 and k >= 3:
                # For 2^k with k ≥ 3, the group is not cyclic
                # It's C_2 × C_{2^(k-2)}
                # -1 generates C_2
                # 5 generates C_{2^(k-2)}
                generators.append((self.m // (2**k), -1, 2))
                generators.append((self.m // (2**k), 5, 2**(k-2)))
            elif p == 2 and k == 2:
                # For 2^2 = 4, the group is C_2
                # -1 generates it
                generators.append((self.m // 4, -1, 2))
            elif p == 2 and k == 1:
                # For 2^1 = 2, the group is trivial
                pass
            else:
                # For odd prime powers p^k, the group is cyclic
                # We need to find a generator
                pk = p**k
                phi_pk = (p - 1) * p**(k - 1)  # φ(p^k) = p^(k-1) * (p-1)
                
                # Find a primitive root modulo p^k
                found = False
                for g in range(2, pk):
                    if DirichletTheorem.gcd(g, p) != 1:
                        continue
                    
                    # Check if g generates the group by checking order
                    if pow(g, phi_pk, pk) == 1:
                        # Check if g has order φ(p^k)
                        is_generator = True
                        for d in range(1, phi_pk):
                            if phi_pk % d == 0 and pow(g, d, pk) == 1:
                                is_generator = False
                                break
                        
                        if is_generator:
                            generators.append((self.m // pk, g, phi_pk))
                            found = True
                            break
                
                # If no generator found (which shouldn't happen for prime powers)
                if not found:
                    raise ValueError(f"Could not find a generator for the group mod {pk}")
        
        # Step 3: Generate all characters using the generators
        # Each character corresponds to a choice of root of unity for each generator
        characters = []
        
        # Get the number of characters (should equal φ(m))
        num_chars = 1
        for _, _, order in generators:
            num_chars *= order
        
        # Pre-compute roots of unity for efficiency
        roots_of_unity = {}
        for _, _, order in generators:
            roots_of_unity[order] = [complex(math.cos(2*math.pi*k/order), math.sin(2*math.pi*k/order)) 
                                     for k in range(order)]
        
        # Generate all combinations of roots of unity
        for idx in range(num_chars):
            # Split the index into choices for each generator
            choices = []
            temp_idx = idx
            for _, _, order in generators:
                choices.append(temp_idx % order)
                temp_idx //= order
            
            # Create a character from these choices
            char = {}
            for a in residues:
                char_val = 1.0
                for (factor, g, order), choice in zip(generators, choices):
                    # Find the exponent of g that gives a in the corresponding factor
                    if factor == 1:
                        # Special case for m = 1
                        exp = 0
                    else:
                        # Find the exponent by solving g^e ≡ a (mod p^k)
                        # This is the discrete logarithm problem, which is hard in general
                        # For simplicity, we'll use brute force for small moduli
                        pk = self.m // factor
                        a_mod_pk = a % pk
                        
                        # Special handling for p = 2, k ≥ 3 where we have two generators
                        if order == 2 and g == -1:
                            # This generator contributes -1 when a ≡ -1 (mod 2^k)
                            exp = 1 if a_mod_pk > pk // 2 else 0
                        else:
                            # General case: find the discrete logarithm
                            found = False
                            for e in range(order):
                                if pow(g, e, pk) == a_mod_pk:
                                    exp = e
                                    found = True
                                    break
                            
                            if not found:
                                raise ValueError(f"Could not find discrete logarithm for {a_mod_pk} base {g} mod {pk}")
                    
                    # Multiply by the corresponding root of unity
                    char_val *= roots_of_unity[order][(exp * choice) % order]
                
                # Handle floating point precision issues
                if abs(char_val.real) < 1e-10:
                    char_val = complex(0, char_val.imag)
                if abs(char_val.imag) < 1e-10:
                    char_val = complex(char_val.real, 0)
                
                char[a] = char_val
            
            # Add character values for a ≡ 0 (mod m)
            for a in range(self.m):
                if DirichletTheorem.gcd(a, self.m) != 1:
                    char[a] = 0
            
            characters.append(char)
        
        self._characters = characters
        return characters
    
    def character_value(self, chi_idx: int, n: int) -> complex:
        """
        Get the value of a specific character at n
        
        Args:
            chi_idx: Index of the character (0 is the principal character)
            n: Integer
            
        Returns:
            Value of χ(n)
        """
        if self._characters is None:
            self.generate_characters()
        
        # Ensure n is reduced modulo m
        n = n % self.m
        
        # Handle GCD > 1 case
        if DirichletTheorem.gcd(n, self.m) != 1:
            return 0
        
        return self._characters[chi_idx][n]
    
    def is_principal(self, chi_idx: int) -> bool:
        """
        Check if a character is the principal character
        
        Args:
            chi_idx: Index of the character
            
        Returns:
            True if χ is the principal character, False otherwise
        """
        if self._characters is None:
            self.generate_characters()
        
        # Principal character maps all values to 1
        for n in range(1, self.m):
            if DirichletTheorem.gcd(n, self.m) == 1:
                if self._characters[chi_idx][n] != 1:
                    return False
        
        return True
    
    def is_real(self, chi_idx: int) -> bool:
        """
        Check if a character is real-valued
        
        Args:
            chi_idx: Index of the character
            
        Returns:
            True if χ takes only real values, False otherwise
        """
        if self._characters is None:
            self.generate_characters()
        
        for n in range(1, self.m):
            if DirichletTheorem.gcd(n, self.m) == 1:
                if abs(self._characters[chi_idx][n].imag) > 1e-10:
                    return False
        
        return True
    
    def is_primitive(self, chi_idx: int) -> bool:
        """
        Check if a character is primitive
        
        Args:
            chi_idx: Index of the character
            
        Returns:
            True if χ is primitive, False otherwise
        """
        if self._characters is None:
            self.generate_characters()
        
        # A character is primitive if it doesn't factor through a smaller modulus
        # This happens when the conductor equals the modulus
        
        # For each divisor d of m, check if χ(n) = 1 for all n with gcd(n, m) = 1 and n ≡ 1 (mod d)
        for d in range(1, self.m):
            if self.m % d == 0 and d != self.m:
                all_ones = True
                for n in range(1, self.m):
                    if (DirichletTheorem.gcd(n, self.m) == 1 and 
                        n % d == 1 and 
                        self._characters[chi_idx][n] != 1):
                        all_ones = False
                        break
                
                if all_ones:
                    return False
        
        return True
    
    def plot_character(self, chi_idx: int):
        """
        Visualize a Dirichlet character in the complex plane
        
        Args:
            chi_idx: Index of the character
        """
        if self._characters is None:
            self.generate_characters()
        
        chi = self._characters[chi_idx]
        
        # Get character values for coprime residues
        residues = [i for i in range(1, self.m) if DirichletTheorem.gcd(i, self.m) == 1]
        values = [chi[r] for r in residues]
        
        # Plot in complex plane
        plt.figure(figsize=(8, 8))
        
        # Plot unit circle
        t = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(t), np.sin(t), 'k--', alpha=0.3)
        
        # Plot character values
        for r, v in zip(residues, values):
            plt.plot([0, v.real], [0, v.imag], '-', alpha=0.5)
            plt.plot(v.real, v.imag, 'o', label=f'χ({r})')
        
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Check if principal or real character for title
        is_principal = self.is_principal(chi_idx)
        is_real = self.is_real(chi_idx)
        
        title = f"Dirichlet Character mod {self.m}"
        if is_principal:
            title += " (Principal)"
        if is_real:
            title += " (Real)"
        
        plt.title(title)
        plt.xlabel("Re(χ(n))")
        plt.ylabel("Im(χ(n))")
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_all_characters(self):
        """Plot all Dirichlet characters modulo m"""
        if self._characters is None:
            self.generate_characters()
        
        num_chars = len(self._characters)
        cols = min(3, num_chars)
        rows = (num_chars + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows * cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        # Plot unit circle on each subplot
        t = np.linspace(0, 2*np.pi, 100)
        unit_circle_x = np.cos(t)
        unit_circle_y = np.sin(t)
        
        # Get residues coprime to m
        residues = [i for i in range(1, self.m) if DirichletTheorem.gcd(i, self.m) == 1]
        
        for i, chi in enumerate(self._characters):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            # Plot unit circle
            ax.plot(unit_circle_x, unit_circle_y, 'k--', alpha=0.3)
            
            # Plot character values
            for r in residues:
                v = chi[r]
                ax.plot([0, v.real], [0, v.imag], '-', alpha=0.3)
                ax.plot(v.real, v.imag, 'o')
            
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Check if principal or real character for title
            is_principal = self.is_principal(i)
            is_real = self.is_real(i)
            
            title = f"χ_{i}"
            if is_principal:
                title += " (Principal)"
            if is_real:
                title += " (Real)"
            
            ax.set_title(title)
            ax.set_aspect('equal')
        
        # Hide empty subplots
        for i in range(num_chars, rows * cols):
            row, col = i // cols, i % cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.show()


class DirichletLFunction:
    """
    Implementation of Dirichlet L-functions associated to Dirichlet characters,
    which are the key analytical tools in the proof of Dirichlet's theorem
    """
    
    def __init__(self, character_system: DirichletCharacter):
        """
        Initialize the L-function calculator
        
        Args:
            character_system: DirichletCharacter object
        """
        self.char_system = character_system
        self.m = character_system.m
        self._cached_values = {}
        
        # Precompute Bernoulli numbers for L(1, χ) calculations
        self._bernoulli_cache = {}
    
    def bernoulli_number(self, n: int) -> mp.mpf:
        """
        Calculate the nth Bernoulli number
        
        Args:
            n: Non-negative integer
            
        Returns:
            The nth Bernoulli number as an mpmath value
        """
        if n in self._bernoulli_cache:
            return self._bernoulli_cache[n]
        
        # Use mpmath's built-in function
        result = mp.bernoulli(n)
        self._bernoulli_cache[n] = result
        
        return result
    
    def l_function_series(self, s: Union[float, complex], chi_idx: int, 
                          terms: int = 1000) -> Union[float, complex]:
        """
        Calculate the Dirichlet L-function L(s, χ) using direct series summation
        
        Args:
            s: Complex number with Re(s) > 1
            chi_idx: Index of the character
            terms: Maximum number of terms to sum
            
        Returns:
            Value of L(s, χ)
        """
        if not isinstance(s, complex):
            s = complex(s, 0)
        
        # Convert to mpmath's complex type for high precision
        s_mp = mp.mpc(s.real, s.imag)
        
        # Cache key
        cache_key = (s_mp, chi_idx, terms)
        if cache_key in self._cached_values:
            return self._cached_values[cache_key]
        
        # Check if principal character and s = 1
        if self.char_system.is_principal(chi_idx) and abs(s_mp - 1) < 1e-10:
            return mp.inf
        
        # Direct summation of the series
        result = mp.mpf(0)
        for n in range(1, terms + 1):
            chi_n = self.char_system.character_value(chi_idx, n)
            term = chi_n / mp.power(mp.mpf(n), s_mp)
            result += term
            
            # Check for convergence
            if abs(term) < 1e-15 * abs(result):
                break
        
        # For principal character, compare with Riemann zeta
        if self.char_system.is_principal(chi_idx) and s_mp.real > 1:
            # Principal character is zero for n not coprime to m
            # So we need to adjust the result
            for n in range(1, min(self.m, terms)):
                if DirichletTheorem.gcd(n, self.m) != 1:
                    result += 1 / mp.power(mp.mpf(n), s_mp)
        
        self._cached_values[cache_key] = result
        return result
    
    def l_function_euler_product(self, s: Union[float, complex], chi_idx: int, 
                             prime_limit: int = 100) -> Union[float, complex]:
        """
        Calculate the Dirichlet L-function using the Euler product formula
        
        Args:
            s: Complex number with Re(s) > 1
            chi_idx: Index of the character
            prime_limit: Maximum number of primes to use
            
        Returns:
            Value of L(s, χ)
        """
        if not isinstance(s, complex):
            s = complex(s, 0)
        
        # Convert to mpmath's complex type for high precision
        s_mp = mp.mpc(s.real, s.imag)
        
        # Ensure Re(s) > 1 for convergence
        if s_mp.real <= 1:
            raise ValueError("The Euler product only converges for Re(s) > 1")
        
        # Generate primes up to the limit
        primes = DirichletTheorem.generate_primes(prime_limit)
        
        # Calculate the Euler product
        result = mp.mpf(1)
        for p in primes:
            chi_p = self.char_system.character_value(chi_idx, p)
            term = 1 - chi_p / mp.power(mp.mpf(p), s_mp)
            result *= mp.mpf(1) / term
        
        return result
    
    def l_function_at_one(self, chi_idx: int) -> Union[float, complex]:
        """
        Calculate L(1, χ) which is crucial for the proof of Dirichlet's theorem
        
        Args:
            chi_idx: Index of the character
            
        Returns:
            Value of L(1, χ)
        """
        # Check if principal character
        if self.char_system.is_principal(chi_idx):
            return mp.inf
        
        # For non-principal characters, we have several methods
        if self.char_system.is_real(chi_idx):
            # For real characters, use analytic formula
            return self._l_one_real_character(chi_idx)
        else:
            # For complex characters, use series
            return self.l_function_series(1.01, chi_idx, 10000)
    
    def _l_one_real_character(self, chi_idx: int) -> mp.mpf:
        """
        Calculate L(1, χ) for a real character using special formulas
        
        Args:
            chi_idx: Index of the character
            
        Returns:
            Value of L(1, χ)
        """
        chi = self.char_system._characters[chi_idx]
        
        # For real non-principal character, use the analytic formula:
        # L(1, χ) = -B_{1,χ} / χ(0)
        # where B_{1,χ} is the first generalized Bernoulli number
        
        # Calculate first generalized Bernoulli number directly
        B_1_chi = mp.mpf(0)
        for a in range(1, self.m + 1):
            B_1_chi += chi.get(a, 0) * mp.fdiv(a, self.m)
        
        # Adjust sign based on character properties
        result = -B_1_chi
        
        return result
    
    def verify_non_vanishing(self, max_chi_idx: int = None) -> Dict[int, Union[float, complex]]:
        """
        Verify that L(1, χ) is non-zero for all non-principal characters
        
        Args:
            max_chi_idx: Maximum character index to check, None for all
            
        Returns:
            Dictionary mapping character indices to L(1, χ) values
        """
        if self.char_system._characters is None:
            self.char_system.generate_characters()
        
        if max_chi_idx is None:
            max_chi_idx = len(self.char_system._characters) - 1
        
        results = {}
        for i in range(max_chi_idx + 1):
            if not self.char_system.is_principal(i):
                l_value = self.l_function_at_one(i)
                results[i] = l_value
                
                # Verify non-vanishing
                if abs(l_value) < 1e-10:
                    print(f"Warning: L(1, χ_{i}) = {l_value} is very close to zero!")
        
        return results
    
    def plot_l_function_values(self, chi_idx: int, t_min: float, t_max: float, points: int = 100):
        """
        Plot the values of L(1/2 + it, χ) along the critical line
        
        Args:
            chi_idx: Index of the character
            t_min, t_max: Range of t values
            points: Number of points to compute
        """
        t_values = np.linspace(t_min, t_max, points)
        l_values = []
        
        for t in t_values:
            s = complex(0.5, t)
            l_values.append(complex(self.l_function_series(s, chi_idx)))
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(t_values, [v.real for v in l_values], label='Re')
        plt.plot(t_values, [v.imag for v in l_values], label='Im')
        plt.plot(t_values, [abs(v) for v in l_values], label='|L|')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('t')
        plt.ylabel('L(1/2 + it, χ)')
        plt.title(f'Dirichlet L-function for χ_{chi_idx} on critical line')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot([v.real for v in l_values], [v.imag for v in l_values], 'b-')
        plt.plot([v.real for v in l_values], [v.imag for v in l_values], 'ro', ms=3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Re(L)')
        plt.ylabel('Im(L)')
        plt.title('L-values in complex plane')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()


def dirichlet_theorem_proof_outline():
    """
    Outline the proof of Dirichlet's theorem on arithmetic progressions
    """
    print("=== PROOF OUTLINE OF DIRICHLET'S THEOREM ===\n")
    
    print("Dirichlet's Theorem: For coprime integers a and m, there are ")
    print("infinitely many primes in the arithmetic progression a + km (k ≥ 0).\n")
    
    print("KEY STEPS IN THE PROOF:")
    print("-----------------------\n")
    
    print("1. INTRODUCTION OF DIRICHLET CHARACTERS")
    print("   * Define characters χ: Z → C that are periodic mod m")
    print("   * Properties: χ(ab) = χ(a)χ(b) and χ(a) = 0 if gcd(a,m) > 1")
    print("   * There are φ(m) distinct characters modulo m")
    print("   * Characters form an orthogonal basis for functions on (Z/mZ)*\n")
    
    print("2. DIRICHLET L-FUNCTIONS")
    print("   * Define L(s, χ) = Σ χ(n)/n^s, analogous to Riemann zeta function")
    print("   * For the principal character χ₀: L(s, χ₀) behaves like ζ(s)")
    print("   * L-functions have Euler products: L(s, χ) = Π (1 - χ(p)/p^s)^(-1)")
    print("     connecting them to the distribution of primes\n")
    
    print("3. ANALYTIC PROPERTIES")
    print("   * For non-principal χ, L(s, χ) extends analytically to Re(s) ≥ 1")
    print("   * L(1, χ) ≠ 0 for all non-principal characters (KEY RESULT)")
    print("   * For principal χ₀, L(s, χ₀) has a simple pole at s = 1\n")
    
    print("4. LOGARITHMIC DERIVATIVE AND PRIME COUNTING")
    print("   * Define ψ(x, a, m) = Σ Λ(n) for n ≤ x with n ≡ a (mod m)")
    print("   * Show ψ(x, a, m) ~ x/φ(m) as x → ∞")
    print("   * Use character orthogonality: Σ χ(a)χ(n) = φ(m) if n ≡ a (mod m), 0 otherwise")
    print("   * Express ψ(x, a, m) in terms of L-functions\n")
    
    print("5. FINAL STEP: INFINITUDE OF PRIMES")
    print("   * If finitely many primes in a + km, then ψ(x, a, m) bounded")
    print("   * But we proved ψ(x, a, m) ~ x/φ(m), contradiction")
    print("   * Therefore, infinitely many primes in a + km\n")
    
    print("THE CRITICAL INSIGHT:")
    print("Non-vanishing of L(1, χ) for non-principal characters ensures")
    print("an equal asymptotic distribution of primes among the φ(m) residue")
    print("classes modulo m that are coprime to m.")


def demonstrate_dirichlet_theorem():
    """Demonstrate computational aspects of Dirichlet's theorem"""
    
    print("=== DIRICHLET'S THEOREM DEMONSTRATION ===\n")
    
    # 1. Basic computation of primes in arithmetic progressions
    print("1. PRIMES IN ARITHMETIC PROGRESSIONS")
    print("----------------------------------")
    
    dirichlet = DirichletTheorem()
    
    # Example 1: m = 4, a = 1
    m, a, limit = 4, 1, 50
    primes = dirichlet.primes_in_progression(a, m, limit)
    print(f"Primes of form {a} + {m}k up to {limit}: {primes}")
    
    # Example 2: m = 4, a = 3
    a = 3
    primes = dirichlet.primes_in_progression(a, m, limit)
    print(f"Primes of form {a} + {m}k up to {limit}: {primes}")
    
    # Count primes in different residue classes
    m = 10
    limit = 1000
    counts = dirichlet.prime_counting_progression(-1, m, limit)
    print(f"\nCounts of primes in different residue classes mod {m} up to {limit}:")
    for residue, count in sorted(counts.items()):
        print(f"  a ≡ {residue} (mod {m}): {count} primes")
    
    # 2. Dirichlet characters
    print("\n2. DIRICHLET CHARACTERS")
    print("---------------------")
    
    m = 5
    char_system = DirichletCharacter(m)
    characters = char_system.generate_characters()
    
    print(f"Character values modulo {m}:")
    for idx, chi in enumerate(characters):
        values = []
        for a in range(1, m):
            if DirichletTheorem.gcd(a, m) == 1:
                val = chi[a]
                if abs(val.imag) < 1e-10:
                    val_str = f"{val.real:.1f}"
                else:
                    val_str = f"{val.real:.1f} + {val.imag:.1f}i"
                values.append(f"χ({a}) = {val_str}")
        
        # Identify character type
        char_type = "Principal" if char_system.is_principal(idx) else ""
        if char_system.is_real(idx):
            char_type += " Real" if char_type else "Real"
        if char_system.is_primitive(idx):
            char_type += " Primitive" if char_type else "Primitive"
        
        print(f"  χ_{idx} ({char_type}): {', '.join(values)}")
    
    # 3. L-functions computation
    print("\n3. L-FUNCTIONS")
    print("------------")
    
    l_function = DirichletLFunction(char_system)
    
    # Compute L(1, χ) for all non-principal characters
    l_values = l_function.verify_non_vanishing()
    
    print("L(1, χ) values:")
    for chi_idx, value in l_values.items():
        if isinstance(value, complex) and abs(value.imag) > 1e-10:
            value_str = f"{value.real:.6f} + {value.imag:.6f}i"
        else:
            value_str = f"{float(value.real):.6f}"
        
        print(f"  L(1, χ_{chi_idx}) = {value_str}")
    
    # Compare with Euler product for Re(s) > 1
    s = 2.0
    print(f"\nL({s}, χ) values computed using different methods:")
    for chi_idx in range(len(characters)):
        direct = l_function.l_function_series(s, chi_idx)
        euler = l_function.l_function_euler_product(s, chi_idx)
        
        print(f"  L({s}, χ_{chi_idx}):")
        print(f"    Series:       {float(direct.real):.6f}")
        print(f"    Euler product: {float(euler.real):.6f}")
        print(f"    Relative error: {abs(direct - euler) / abs(direct)}")
    
    # 4. Asymptotic distribution
    print("\n4. ASYMPTOTIC DISTRIBUTION OF PRIMES")
    print("---------------------------------")
    
    m = 4
    limit = 10000
    counts = dirichlet.prime_counting_progression(-1, m, limit)
    
    print(f"Prime counts in residue classes mod {m} up to {limit}:")
    for residue, count in sorted(counts.items()):
        print(f"  a ≡ {residue} (mod {m}): {count} primes")
    
    # Calculate expected counts based on Dirichlet's theorem
    phi_m = DirichletTheorem.euler_phi(m)
    pi_x = len(dirichlet.generate_primes(limit))
    expected = pi_x / phi_m
    
    print(f"\nExpected count for each residue class: ~{expected:.1f}")
    
    for residue, count in sorted(counts.items()):
        error = count - expected
        rel_error = error / expected
        print(f"  a ≡ {residue} (mod {m}): Error = {error:.1f} ({rel_error:.2%})")
    
    # Uncomment to display visualizations
    # dirichlet.plot_prime_distribution(m, limit)
    # char_system.plot_all_characters()
    # l_function.plot_l_function_values(1, 0, 30)


if __name__ == "__main__":
    demonstrate_dirichlet_theorem()
    dirichlet_theorem_proof_outline()
    
    # Create a DirichletTheorem object for interactive use
    # dt = DirichletTheorem()
    # dt.plot_prime_distribution(4, 10000)