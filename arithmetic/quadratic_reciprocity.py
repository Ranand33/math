import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Set
import math
from fractions import Fraction
from collections import defaultdict
import cmath
from functools import lru_cache


class NumberTheory:
    """Class for general number theory utilities"""
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Find the greatest common divisor using Euclidean algorithm"""
        while b:
            a, b = b, a % b
        return abs(a)
    
    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm
        Returns (gcd, x, y) such that a*x + b*y = gcd
        """
        if a == 0:
            return b, 0, 1
        
        gcd, x1, y1 = NumberTheory.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> Optional[int]:
        """
        Find the modular multiplicative inverse of a modulo m
        
        Args:
            a: Integer to find inverse for
            m: Modulus
            
        Returns:
            x such that (a * x) % m == 1, or None if no inverse exists
        """
        g, x, y = NumberTheory.extended_gcd(a, m)
        if g != 1:
            return None  # No modular inverse exists
        else:
            return x % m
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if a number is prime using a simple primality test"""
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
    def prime_factors(n: int) -> Dict[int, int]:
        """
        Find the prime factorization of n
        
        Args:
            n: Integer to factorize
            
        Returns:
            Dictionary mapping prime factors to their exponents
        """
        if n <= 0:
            raise ValueError("Input must be a positive integer")
            
        factors = {}
        # Check divisibility by 2
        while n % 2 == 0:
            factors[2] = factors.get(2, 0) + 1
            n //= 2
            
        # Check odd divisors
        d = 3
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 2
            
        # If n > 1, it is a prime factor
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
            
        return factors
    
    @staticmethod
    def euler_phi(n: int) -> int:
        """
        Calculate Euler's totient function φ(n)
        
        Args:
            n: Positive integer
            
        Returns:
            Number of integers k in the range 1 ≤ k ≤ n that are coprime to n
        """
        if n <= 0:
            raise ValueError("Input must be a positive integer")
            
        # Use the formula: φ(n) = n * Π(1 - 1/p) for all prime p dividing n
        factors = NumberTheory.prime_factors(n)
        result = n
        
        for p in factors:
            result *= (1 - 1/p)
            
        return round(result)
    
    @staticmethod
    def mobius_mu(n: int) -> int:
        """
        Calculate the Möbius function μ(n)
        
        Args:
            n: Positive integer
            
        Returns:
            1 if n is square-free with an even number of prime factors
            -1 if n is square-free with an odd number of prime factors
            0 if n has a squared prime factor
        """
        if n <= 0:
            raise ValueError("Input must be a positive integer")
        if n == 1:
            return 1
            
        factors = NumberTheory.prime_factors(n)
        
        # If any prime factor has exponent >= 2, return 0
        if any(exp >= 2 for exp in factors.values()):
            return 0
            
        # Otherwise, return (-1)^k where k is the number of prime factors
        return (-1) ** len(factors)
    
    @staticmethod
    def divisors(n: int) -> List[int]:
        """
        Find all divisors of n
        
        Args:
            n: Positive integer
            
        Returns:
            List of all divisors of n
        """
        if n <= 0:
            raise ValueError("Input must be a positive integer")
            
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:  # Avoid duplicates for perfect squares
                    divisors.append(n // i)
                    
        return sorted(divisors)
    
    @staticmethod
    @lru_cache(maxsize=None)
    def factorial(n: int) -> int:
        """
        Calculate n! (factorial of n)
        
        Args:
            n: Non-negative integer
            
        Returns:
            n!
        """
        if n < 0:
            raise ValueError("Input must be a non-negative integer")
        if n == 0 or n == 1:
            return 1
        
        return n * NumberTheory.factorial(n - 1)


class Cyclotomy:
    """Class for cyclotomic polynomials and roots of unity"""
    
    def __init__(self):
        """Initialize the Cyclotomy class"""
        self.x = sp.Symbol('x')
        self._cyclotomic_cache = {}
    
    def cyclotomic_polynomial(self, n: int) -> sp.Poly:
        """
        Compute the nth cyclotomic polynomial Φₙ(x)
        
        Args:
            n: Positive integer
            
        Returns:
            The nth cyclotomic polynomial as a sympy polynomial
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        
        # Check cache
        if n in self._cyclotomic_cache:
            return self._cyclotomic_cache[n]
        
        # Method 1: Definition using the formula
        # Φₙ(x) = Π(x^d - 1)^μ(n/d) for all d dividing n
        if n == 1:
            result = sp.Poly(self.x - 1, self.x)
        else:
            divisors = NumberTheory.divisors(n)
            result = sp.Poly(1, self.x)
            
            for d in divisors:
                mu = NumberTheory.mobius_mu(n // d)
                if mu != 0:
                    factor = self.x**d - 1
                    result = result * sp.Poly(factor, self.x)**mu
                
            # Simplify
            result = sp.Poly(sp.expand(result.as_expr()), self.x)
            
        # Cache result
        self._cyclotomic_cache[n] = result
        return result
    
    def cyclotomic_polynomial_alt(self, n: int) -> sp.Poly:
        """
        Compute the nth cyclotomic polynomial Φₙ(x) using a recursive algorithm
        
        Args:
            n: Positive integer
            
        Returns:
            The nth cyclotomic polynomial as a sympy polynomial
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        
        # Check cache
        if n in self._cyclotomic_cache:
            return self._cyclotomic_cache[n]
        
        # Base cases
        if n == 1:
            result = sp.Poly(self.x - 1, self.x)
        elif n == 2:
            result = sp.Poly(self.x + 1, self.x)
        else:
            # Recursive method using the formula:
            # x^n - 1 = Π Φₖ(x) for all k dividing n
            xn_minus_1 = sp.Poly(self.x**n - 1, self.x)
            divisors = NumberTheory.divisors(n)[:-1]  # All divisors except n itself
            
            # Compute the product of all Φₖ(x) for k < n
            product = sp.Poly(1, self.x)
            for k in divisors:
                product *= self.cyclotomic_polynomial_alt(k)
            
            # Φₙ(x) = (x^n - 1) / Π Φₖ(x) for all k < n dividing n
            result = sp.quo(xn_minus_1, product)
        
        # Cache result
        self._cyclotomic_cache[n] = result
        return result
    
    def roots_of_unity(self, n: int, primitive: bool = False) -> List[complex]:
        """
        Compute the nth roots of unity
        
        Args:
            n: Positive integer
            primitive: If True, return only primitive nth roots of unity
            
        Returns:
            List of complex numbers representing the nth roots of unity
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        
        if primitive:
            # Primitive roots are those where gcd(k, n) = 1
            return [cmath.rect(1, 2*math.pi*k/n) for k in range(n) if math.gcd(k, n) == 1]
        else:
            # All nth roots of unity
            return [cmath.rect(1, 2*math.pi*k/n) for k in range(n)]
    
    def plot_roots_of_unity(self, n: int, primitive: bool = False):
        """
        Plot the nth roots of unity on the complex plane
        
        Args:
            n: Positive integer
            primitive: If True, highlight primitive roots
        """
        plt.figure(figsize=(8, 8))
        
        # Plot unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        plt.gca().add_patch(circle)
        
        # Plot all roots
        all_roots = self.roots_of_unity(n, False)
        x_coords = [z.real for z in all_roots]
        y_coords = [z.imag for z in all_roots]
        
        if primitive:
            # Get primitive roots
            prim_roots = self.roots_of_unity(n, True)
            prim_x = [z.real for z in prim_roots]
            prim_y = [z.imag for z in prim_roots]
            
            # Plot non-primitive roots
            non_prim_x = [x for i, x in enumerate(x_coords) if complex(x, y_coords[i]) not in prim_roots]
            non_prim_y = [y for i, y in enumerate(y_coords) if complex(x_coords[i], y) not in prim_roots]
            plt.scatter(non_prim_x, non_prim_y, color='blue', s=50, label='Non-primitive roots')
            
            # Plot primitive roots
            plt.scatter(prim_x, prim_y, color='red', s=70, label='Primitive roots')
        else:
            plt.scatter(x_coords, y_coords, color='blue', s=50)
        
        # Plot connecting lines from origin
        for x, y in zip(x_coords, y_coords):
            plt.plot([0, x], [0, y], 'k-', alpha=0.3)
        
        # Set plot properties
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.title(f"The {n}th Roots of Unity")
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.axis('equal')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        
        if primitive:
            plt.legend()
        
        plt.show()
    
    def gauss_sum(self, n: int, k: int = 1) -> complex:
        """
        Compute the Gauss sum G(k, n) = Σ e^(2πi*k*j/n) for j = 0 to n-1
        
        Args:
            n: Positive integer (modulus)
            k: Integer
            
        Returns:
            Value of the Gauss sum
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        
        result = 0
        for j in range(n):
            result += cmath.exp(2j * math.pi * k * j / n)
        
        # For exact values in special cases
        if k % n == 0:
            return complex(n, 0)
        elif math.gcd(k, n) == 1:
            # If n is a prime p
            if NumberTheory.is_prime(n):
                # For p odd prime: G(1, p) = sqrt(p) * e^(iπ(p-1)/4)
                if n % 4 == 1:  # p ≡ 1 (mod 4)
                    return complex(math.sqrt(n), 0)
                elif n % 4 == 3:  # p ≡ 3 (mod 4)
                    return complex(0, math.sqrt(n))
        
        # For numerical stability, round very small values to zero
        if abs(result.real) < 1e-10:
            result = complex(0, result.imag)
        if abs(result.imag) < 1e-10:
            result = complex(result.real, 0)
            
        return result
    
    def ramanujan_sum(self, n: int, k: int) -> int:
        """
        Compute the Ramanujan sum c_n(k) = Σ e^(2πi*k*j/n) for j in Z_n*
        where Z_n* consists of integers relatively prime to n
        
        Args:
            n: Positive integer (modulus)
            k: Integer
            
        Returns:
            Value of the Ramanujan sum (always an integer)
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        
        # Method 1: Direct computation
        result = 0
        for j in range(1, n):
            if math.gcd(j, n) == 1:
                result += cmath.exp(2j * math.pi * k * j / n)
        
        # Ramanujan sums are always real and integer-valued
        return round(result.real)
    
    def cyclotomic_field_element(self, coeffs: List[int], n: int) -> str:
        """
        Represent an element of the nth cyclotomic field Q(ζ_n)
        
        Args:
            coeffs: List of integer coefficients [a_0, a_1, ..., a_{φ(n)-1}]
            n: Positive integer
            
        Returns:
            String representation of the element
        """
        phi_n = NumberTheory.euler_phi(n)
        
        if len(coeffs) > phi_n:
            raise ValueError(f"Too many coefficients. Need at most φ({n}) = {phi_n}")
        
        # Pad coefficients with zeros if needed
        padded_coeffs = coeffs + [0] * (phi_n - len(coeffs))
        
        # Create symbolic variable for the primitive nth root of unity
        zeta = sp.Symbol(f'ζ_{n}')
        
        # Build the linear combination
        result = sp.Integer(0)
        for i, coeff in enumerate(padded_coeffs):
            if coeff != 0:
                term = coeff * zeta**i
                result += term
        
        return str(result).replace(f'ζ_{n}', f'ζ')
    
    def minimal_polynomial(self, k: int, n: int) -> sp.Poly:
        """
        Find the minimal polynomial of ζ_n^k over Q
        
        Args:
            k: Integer
            n: Positive integer
            
        Returns:
            Minimal polynomial as a sympy polynomial
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        
        # Simplify k mod n
        k = k % n
        
        # If k = 0, the minimal polynomial is x - 1
        if k == 0:
            return sp.Poly(self.x - 1, self.x)
        
        # Calculate d = gcd(k, n)
        d = math.gcd(k, n)
        
        # The minimal polynomial of ζ_n^k is Φ_{n/d}(x^{d/gcd(k/d, n/d)})
        n_d = n // d
        k_d = k // d
        g = math.gcd(k_d, n_d)
        
        # For Q(ζ_{n/d}), the minimal polynomial is Φ_{n/d}(x)
        if g == 1:
            return self.cyclotomic_polynomial(n_d)
        
        # For more complex cases, we need to substitute
        # Create a new variable for substitution
        y = sp.Symbol('y')
        phi = self.cyclotomic_polynomial(n_d).as_expr().subs(self.x, y)
        
        # Substitute y = x^g
        result = phi.subs(y, self.x**g)
        
        return sp.Poly(result, self.x)


class QuadraticReciprocity:
    """Class for quadratic residues and the law of quadratic reciprocity"""
    
    @staticmethod
    def legendre_symbol(a: int, p: int) -> int:
        """
        Compute the Legendre symbol (a/p)
        
        Args:
            a: Integer
            p: Odd prime
            
        Returns:
            1 if a is a quadratic residue modulo p
            -1 if a is a quadratic non-residue modulo p
            0 if a ≡ 0 (mod p)
        """
        if p <= 1 or not NumberTheory.is_prime(p):
            raise ValueError("p must be a prime number")
        
        # Reduce a modulo p
        a = a % p
        
        if a == 0:
            return 0
        
        # Use Euler's criterion: a^((p-1)/2) ≡ (a/p) (mod p)
        result = pow(a, (p - 1) // 2, p)
        
        # Convert result from {0, 1, p-1} to {0, 1, -1}
        if result == p - 1:
            return -1
        return result
    
    @staticmethod
    def jacobi_symbol(a: int, n: int) -> int:
        """
        Compute the Jacobi symbol (a/n)
        
        Args:
            a: Integer
            n: Odd positive integer
            
        Returns:
            Jacobi symbol value (1, -1, or 0)
        """
        if n <= 0 or n % 2 == 0:
            raise ValueError("n must be an odd positive integer")
        
        # Reduce a modulo n
        a = a % n
        
        if a == 0:
            return 0 if n > 1 else 1
        
        if a == 1:
            return 1
        
        # Factor a into a = 2^e * a1, where a1 is odd
        e = 0
        a1 = a
        while a1 % 2 == 0:
            e += 1
            a1 //= 2
        
        # Apply quadratic reciprocity law
        if e % 2 == 0:
            s = 1
        else:
            # Adjust sign based on n mod 8
            s = 1 if n % 8 == 1 or n % 8 == 7 else -1
        
        if n % 4 == 3 and a1 % 4 == 3:
            s = -s
        
        # Recursively compute (a1/n) = (n%a1/a1) if a1 > 1
        if a1 == 1:
            return s
        else:
            return s * QuadraticReciprocity.jacobi_symbol(n % a1, a1)
    
    @staticmethod
    def quadratic_residues(p: int) -> List[int]:
        """
        Find all quadratic residues modulo p
        
        Args:
            p: Prime number
            
        Returns:
            List of quadratic residues modulo p
        """
        if p <= 1 or not NumberTheory.is_prime(p):
            raise ValueError("p must be a prime number")
        
        # A number a is a quadratic residue modulo p if there exists x such that x^2 ≡ a (mod p)
        residues = set()
        for x in range(p):
            residue = (x * x) % p
            residues.add(residue)
        
        return sorted(residues)
    
    @staticmethod
    def tonelli_shanks(n: int, p: int) -> Optional[int]:
        """
        Find a square root of n modulo p, if it exists.
        Uses the Tonelli-Shanks algorithm for p ≡ 1 (mod 4)
        
        Args:
            n: Integer whose square root to find
            p: Odd prime
            
        Returns:
            x such that x^2 ≡ n (mod p), or None if no solution exists
        """
        if p <= 1 or not NumberTheory.is_prime(p):
            raise ValueError("p must be a prime number")
        
        # Ensure n is reduced modulo p
        n = n % p
        
        # Check if n is a quadratic residue
        if n == 0:
            return 0
        
        if QuadraticReciprocity.legendre_symbol(n, p) == -1:
            return None  # No solution exists
        
        # Special case for p ≡ 3 (mod 4)
        if p % 4 == 3:
            return pow(n, (p + 1) // 4, p)
        
        # Factor p-1 as q * 2^s where q is odd
        q, s = p - 1, 0
        while q % 2 == 0:
            q //= 2
            s += 1
        
        # Find a quadratic non-residue z
        z = 2
        while QuadraticReciprocity.legendre_symbol(z, p) != -1:
            z += 1
        
        # Initialize algorithm
        m = s
        c = pow(z, q, p)
        t = pow(n, q, p)
        r = pow(n, (q + 1) // 2, p)
        
        # Main loop
        while t != 1:
            # Find the least i such that t^(2^i) ≡ 1 (mod p)
            i = 0
            t_i = t
            while t_i != 1:
                t_i = (t_i * t_i) % p
                i += 1
                if i >= m:
                    return None  # Should not happen if n is a quadratic residue
            
            # Compute b = c^(2^(m-i-1)) mod p
            b = pow(c, 2**(m - i - 1), p)
            
            # Update variables
            m = i
            c = (b * b) % p
            t = (t * c) % p
            r = (r * b) % p
        
        return r
    
    @staticmethod
    def verify_quadratic_reciprocity(p: int, q: int) -> bool:
        """
        Verify the law of quadratic reciprocity for primes p and q
        
        Args:
            p, q: Distinct odd primes
            
        Returns:
            True if the law holds for p and q
        """
        if p <= 1 or q <= 1 or not NumberTheory.is_prime(p) or not NumberTheory.is_prime(q):
            raise ValueError("p and q must be prime numbers")
        
        # The law states: (p/q)(q/p) = (-1)^((p-1)(q-1)/4)
        p_over_q = QuadraticReciprocity.legendre_symbol(p, q)
        q_over_p = QuadraticReciprocity.legendre_symbol(q, p)
        
        expected_sign = (-1)**((p-1)*(q-1)//4)
        
        return p_over_q * q_over_p == expected_sign
    
    @staticmethod
    def plot_quadratic_character(p: int):
        """
        Plot the quadratic character modulo p
        
        Args:
            p: Prime number
        """
        if p <= 1 or not NumberTheory.is_prime(p):
            raise ValueError("p must be a prime number")
        
        # Compute Legendre symbols (a/p) for a from 1 to p-1
        x_values = list(range(1, p))
        y_values = [QuadraticReciprocity.legendre_symbol(a, p) for a in x_values]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, s=30, c=y_values, cmap='viridis')
        plt.title(f"Quadratic Character Modulo {p}")
        plt.xlabel("a")
        plt.ylabel("(a/p)")
        plt.ylim(-1.5, 1.5)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal lines at y = 1 and y = -1
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=-1, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Show count of residues and non-residues
        residues = y_values.count(1)
        non_residues = y_values.count(-1)
        plt.annotate(f"Quadratic residues: {residues}", (p*0.7, 1.1))
        plt.annotate(f"Quadratic non-residues: {non_residues}", (p*0.7, -1.1))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def kronecker_symbol(a: int, n: int) -> int:
        """
        Compute the Kronecker symbol (a/n) which extends the Jacobi symbol to all integers
        
        Args:
            a: Integer
            n: Integer
            
        Returns:
            Kronecker symbol value
        """
        if n == 0:
            return 1 if abs(a) == 1 else 0
        
        # Handle negative n
        if n < 0:
            return QuadraticReciprocity.kronecker_symbol(a, -n) * (1 if a >= 0 or n % 4 == 1 else -1)
        
        # Factor out powers of 2
        if n % 2 == 0:
            if a % 2 == 0:
                return 0
            n_odd = n
            while n_odd % 2 == 0:
                n_odd //= 2
            
            # Apply rule for Kronecker symbol with n = 2
            if a % 8 == 1 or a % 8 == 7:
                return QuadraticReciprocity.kronecker_symbol(a, n_odd)
            else:
                return -QuadraticReciprocity.kronecker_symbol(a, n_odd)
        
        # For odd n, the Kronecker symbol is the same as the Jacobi symbol
        return QuadraticReciprocity.jacobi_symbol(a, n)
    
    @staticmethod
    def hilbert_symbol(a: int, b: int, p: int) -> int:
        """
        Compute the Hilbert symbol (a, b)_p which is 1 if ax² + by² = z² has a non-trivial solution in Q_p
        
        Args:
            a, b: Integers
            p: Prime or p = -1 (representing the real place)
            
        Returns:
            1 if the quadratic form ax² + by² represents a square in Q_p, -1 otherwise
        """
        if p == -1:  # Real place
            return 1 if a > 0 or b > 0 else -1
        
        if p == 2:  # Special case for p = 2
            # Simplify a and b to remove factors of 4
            while a % 4 == 0:
                a //= 4
            while b % 4 == 0:
                b //= 4
            
            # Check specific cases for p = 2
            if a % 2 == 0 and b % 2 == 0:
                return 1
            
            if a % 2 == 0:
                if b % 8 == 1 or b % 8 == 7:
                    return 1
                return -1
            
            if b % 2 == 0:
                if a % 8 == 1 or a % 8 == 7:
                    return 1
                return -1
            
            # Both a and b are odd
            if a % 4 == 3 and b % 4 == 3:
                return -1
            return 1
        
        # For odd primes
        return QuadraticReciprocity.legendre_symbol(a, p) * QuadraticReciprocity.legendre_symbol(b, p) * QuadraticReciprocity.legendre_symbol(-a * b, p)


def demonstrate_cyclotomy():
    """Demonstrate cyclotomic polynomials and roots of unity"""
    
    print("=== CYCLOTOMY DEMONSTRATION ===\n")
    
    # Create Cyclotomy object
    cyclotomy = Cyclotomy()
    
    # 1. Cyclotomic Polynomials
    print("1. CYCLOTOMIC POLYNOMIALS")
    print("-------------------------")
    
    for n in range(1, 11):
        phi_n = cyclotomy.cyclotomic_polynomial(n)
        print(f"Φ_{n}(x) = {phi_n.as_expr()}")
    
    # 2. Roots of Unity
    print("\n2. ROOTS OF UNITY")
    print("---------------")
    
    n = 8
    roots = cyclotomy.roots_of_unity(n)
    print(f"All {n}th roots of unity:")
    for i, z in enumerate(roots):
        print(f"ζ_{n}^{i} = {z.real:.3f} + {z.imag:.3f}i")
    
    prim_roots = cyclotomy.roots_of_unity(n, primitive=True)
    print(f"\nPrimitive {n}th roots of unity:")
    for z in prim_roots:
        k = roots.index(z)
        print(f"ζ_{n}^{k} = {z.real:.3f} + {z.imag:.3f}i")
    
    # 3. Gauss and Ramanujan Sums
    print("\n3. GAUSS AND RAMANUJAN SUMS")
    print("--------------------------")
    
    p = 5
    print(f"Gauss sums for n = {p}:")
    for k in range(p):
        gauss = cyclotomy.gauss_sum(p, k)
        print(f"G({k}, {p}) = {gauss}")
    
    print(f"\nRamanujan sums c_{p}(k):")
    for k in range(1, p+1):
        ram = cyclotomy.ramanujan_sum(p, k)
        print(f"c_{p}({k}) = {ram}")
    
    # 4. Cyclotomic Field Elements
    print("\n4. CYCLOTOMIC FIELD ELEMENTS")
    print("--------------------------")
    
    n = 5
    phi_n = NumberTheory.euler_phi(n)
    print(f"Q(ζ_{n}) has degree φ({n}) = {phi_n} over Q")
    
    # Examples of elements in Q(ζ₅)
    elements = [
        [1, 0, 0, 0],  # 1
        [0, 1, 0, 0],  # ζ₅
        [0, 0, 1, 0],  # ζ₅²
        [1, 1, 1, 1],  # 1 + ζ₅ + ζ₅² + ζ₅³
    ]
    
    for coeffs in elements:
        elem = cyclotomy.cyclotomic_field_element(coeffs, n)
        print(f"Element: {elem}")
    
    # 5. Minimal Polynomials
    print("\n5. MINIMAL POLYNOMIALS")
    print("--------------------")
    
    n = 8
    for k in range(1, n):
        min_poly = cyclotomy.minimal_polynomial(k, n)
        print(f"Min poly of ζ_{n}^{k} over Q: {min_poly.as_expr()}")
    
    # Uncomment to visualize roots of unity
    # cyclotomy.plot_roots_of_unity(12, primitive=True)


def demonstrate_quadratic_reciprocity():
    """Demonstrate quadratic reciprocity and related concepts"""
    
    print("\n=== QUADRATIC RECIPROCITY DEMONSTRATION ===\n")
    
    # 1. Legendre and Jacobi Symbols
    print("1. LEGENDRE AND JACOBI SYMBOLS")
    print("----------------------------")
    
    p = 13
    print(f"Legendre symbols (a/{p}):")
    for a in range(1, p):
        legendre = QuadraticReciprocity.legendre_symbol(a, p)
        print(f"({a}/{p}) = {legendre}")
    
    # Jacobi symbols
    n = 15  # Composite
    print(f"\nJacobi symbols (a/{n}):")
    for a in range(1, n):
        jacobi = QuadraticReciprocity.jacobi_symbol(a, n)
        print(f"({a}/{n}) = {jacobi}")
    
    # 2. Quadratic Residues
    print("\n2. QUADRATIC RESIDUES")
    print("------------------")
    
    p = 17
    residues = QuadraticReciprocity.quadratic_residues(p)
    print(f"Quadratic residues modulo {p}: {residues}")
    print(f"Number of quadratic residues: {len(residues) - 1}")  # Subtract 1 to exclude 0
    
    # 3. Square Roots Modulo p
    print("\n3. SQUARE ROOTS MODULO p")
    print("----------------------")
    
    p = 17
    for a in range(1, p):
        if QuadraticReciprocity.legendre_symbol(a, p) == 1:
            root = QuadraticReciprocity.tonelli_shanks(a, p)
            print(f"√{a} ≡ {root} (mod {p})  [since {root}² ≡ {(root*root) % p} ≡ {a} (mod {p})]")
    
    # 4. Verification of Quadratic Reciprocity
    print("\n4. VERIFICATION OF QUADRATIC RECIPROCITY")
    print("-------------------------------------")
    
    primes = [3, 5, 7, 11, 13, 17, 19, 23]
    print("Verification of quadratic reciprocity law:")
    for i, p in enumerate(primes):
        for q in primes[i+1:]:
            p_over_q = QuadraticReciprocity.legendre_symbol(p, q)
            q_over_p = QuadraticReciprocity.legendre_symbol(q, p)
            expected = (-1)**((p-1)*(q-1)//4)
            verified = QuadraticReciprocity.verify_quadratic_reciprocity(p, q)
            
            print(f"({p}/{q})·({q}/{p}) = {p_over_q}·{q_over_p} = {p_over_q * q_over_p}, "
                  f"(-1)^({p-1})({q-1})/4 = {expected}, "
                  f"Verified: {verified}")
    
    # 5. Extended Symbols
    print("\n5. EXTENDED RECIPROCITY SYMBOLS")
    print("----------------------------")
    
    # Examples of Kronecker symbol
    print("Kronecker symbols:")
    examples = [(2, 15), (-1, 5), (7, -3), (5, 0), (3, -2)]
    for a, n in examples:
        symbol = QuadraticReciprocity.kronecker_symbol(a, n)
        print(f"({a}/{n}) = {symbol}")
    
    # Examples of Hilbert symbol
    print("\nHilbert symbols:")
    examples = [(2, 3, 5), (-1, -7, 2), (5, -3, 7), (-1, -1, -1)]
    for a, b, p in examples:
        symbol = QuadraticReciprocity.hilbert_symbol(a, b, p)
        print(f"({a},{b})_{p} = {symbol}")
    
    # Uncomment to visualize quadratic characters
    # QuadraticReciprocity.plot_quadratic_character(23)


if __name__ == "__main__":
    demonstrate_cyclotomy()
    demonstrate_quadratic_reciprocity()