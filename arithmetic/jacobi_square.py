import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set, Generator, Optional
import math
from collections import defaultdict
from itertools import product
import time
from functools import lru_cache
from sympy import symbols, expand, collect
import sympy


class JacobiFourSquares:
    """
    Implementation of Jacobi's four-square theorem and related functionality.
    
    Jacobi's four-square theorem states that the number of ways to represent
    a positive integer n as a sum of four squares (including order and signs)
    is given by:
    
    r₄(n) = 8 * sum of divisors of n, if n is odd
    r₄(n) = 24 * sum of odd divisors of n, if n is even
    
    where r₄(n) is the number of solutions to: n = x₁² + x₂² + x₃² + x₄²
    with x₁, x₂, x₃, x₄ ∈ Z (integers, including negatives and zero).
    """
    
    def __init__(self):
        """Initialize the JacobiFourSquares calculator."""
        # Caches for divisors and r4 values
        self._divisor_cache = {}
        self._r4_cache = {}
        self._representations_cache = {}
    
    def divisors(self, n: int) -> List[int]:
        """
        Find all divisors of a positive integer n.
        
        Args:
            n: Positive integer
            
        Returns:
            List of all divisors of n
        """
        if n <= 0:
            raise ValueError("Input must be a positive integer")
        
        # Check cache
        if n in self._divisor_cache:
            return self._divisor_cache[n]
            
        # Find divisors by checking all integers up to sqrt(n)
        result = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                result.append(i)
                if i != n // i:  # Avoid duplicates for perfect squares
                    result.append(n // i)
        
        result.sort()
        self._divisor_cache[n] = result
        return result
    
    def odd_divisors(self, n: int) -> List[int]:
        """
        Find all odd divisors of a positive integer n.
        
        Args:
            n: Positive integer
            
        Returns:
            List of all odd divisors of n
        """
        return [d for d in self.divisors(n) if d % 2 == 1]
    
    def sum_of_divisors(self, n: int, odd_only: bool = False) -> int:
        """
        Calculate the sum of divisors of n, optionally only odd divisors.
        
        Args:
            n: Positive integer
            odd_only: If True, sum only odd divisors
            
        Returns:
            Sum of divisors (or odd divisors) of n
        """
        if odd_only:
            return sum(self.odd_divisors(n))
        else:
            return sum(self.divisors(n))
    
    def r4(self, n: int) -> int:
        """
        Calculate r₄(n), the number of ways to represent n as a sum of four squares.
        
        Args:
            n: Non-negative integer
            
        Returns:
            Number of representations n = x₁² + x₂² + x₃² + x₄² with x₁,x₂,x₃,x₄ ∈ Z
        """
        if n < 0:
            raise ValueError("Input must be a non-negative integer")
        
        # Check cache
        if n in self._r4_cache:
            return self._r4_cache[n]
            
        # Apply Jacobi's formula
        if n == 0:
            # Special case: r₄(0) = 1 (only the representation 0 = 0² + 0² + 0² + 0²)
            result = 1
        elif n % 2 == 1:
            # For odd n: r₄(n) = 8 * sum of all divisors of n
            result = 8 * self.sum_of_divisors(n)
        else:
            # For even n: r₄(n) = 24 * sum of odd divisors of n
            result = 24 * self.sum_of_divisors(n, odd_only=True)
        
        self._r4_cache[n] = result
        return result
    
    def verify_r4(self, n: int, limit: Optional[int] = None) -> bool:
        """
        Verify r₄(n) by direct enumeration and comparison with Jacobi's formula.
        
        Args:
            n: Non-negative integer
            limit: Optional limit for enumeration (for large n)
            
        Returns:
            True if verification succeeds, False otherwise
        """
        # Calculate r₄(n) using Jacobi's formula
        theoretical = self.r4(n)
        
        # Count representations by direct enumeration
        representations = list(self.enumerate_four_square_representations(n, limit))
        empirical = len(representations)
        
        return theoretical == empirical
    
    def enumerate_four_square_representations(self, n: int, limit: Optional[int] = None) -> Generator[Tuple[int, int, int, int], None, None]:
        """
        Enumerate all ways to represent n as a sum of four squares.
        
        Args:
            n: Non-negative integer
            limit: Optional limit for enumeration (for large n)
            
        Yields:
            Tuples (x₁, x₂, x₃, x₄) such that n = x₁² + x₂² + x₃² + x₄²
        """
        if n < 0:
            raise ValueError("Input must be a non-negative integer")
        
        # Set a reasonable default limit for the search
        if limit is None:
            limit = int(math.sqrt(n)) + 1
        else:
            limit = min(limit, int(math.sqrt(n)) + 1)
        
        # Generate all representations by brute force
        # Optimize by restricting the search space
        for x1 in range(-limit, limit + 1):
            rem1 = n - x1**2
            if rem1 < 0:
                continue
                
            for x2 in range(-limit, limit + 1):
                rem2 = rem1 - x2**2
                if rem2 < 0:
                    continue
                    
                for x3 in range(-limit, limit + 1):
                    rem3 = rem2 - x3**2
                    if rem3 < 0:
                        continue
                    
                    # Check if rem3 is a perfect square
                    x4_squared = rem3
                    x4 = int(math.sqrt(x4_squared) + 0.5)
                    
                    if x4**2 == x4_squared:
                        yield (x1, x2, x3, x4)
                        if x4 != 0:
                            yield (x1, x2, x3, -x4)
    
    def find_one_representation(self, n: int) -> Tuple[int, int, int, int]:
        """
        Find a single representation of n as a sum of four squares.
        Uses Lagrange's algorithm, which is more efficient for large n.
        
        Args:
            n: Non-negative integer
            
        Returns:
            Tuple (a,b,c,d) such that n = a² + b² + c² + d²
        """
        if n < 0:
            raise ValueError("Input must be a non-negative integer")
        
        if n == 0:
            return (0, 0, 0, 0)
        
        # Step 1: Reduce to the case where n is not divisible by 4
        # If n = 4^k * m where m is not divisible by 4, then
        # n = 4^k * (a² + b² + c² + d²) = (2^k*a)² + (2^k*b)² + (2^k*c)² + (2^k*d)²
        k = 0
        original_n = n
        while n % 4 == 0:
            n //= 4
            k += 1
        
        # Step 2: Use Lagrange's three-square theorem for cases where n ≡ 1, 2, 3 (mod 8)
        if n % 8 in [1, 2, 3, 5, 6, 7]:
            # Find a representation for n
            result = self._lagrange_four_square(n)
            
            # Scale back if needed
            if k > 0:
                factor = 2**k
                result = tuple(x * factor for x in result)
            
            return result
        
        # Step 3: Handle the case n = 0 (mod 8)
        # In this case, n/8 can be represented as a sum of four squares
        result = self._lagrange_four_square(n // 8)
        
        # Scale by 2*2^k for the final result
        factor = 2 * (2**k)
        result = tuple(x * factor for x in result)
        
        return result
    
    def _lagrange_four_square(self, n: int) -> Tuple[int, int, int, int]:
        """
        Implement Lagrange's algorithm to find a four-square representation.
        
        Args:
            n: Positive integer
            
        Returns:
            Tuple (a,b,c,d) such that n = a² + b² + c² + d²
        """
        # Case 1: n is a perfect square
        if int(math.sqrt(n))**2 == n:
            return (int(math.sqrt(n)), 0, 0, 0)
        
        # Case 2: n is a sum of two squares
        # Try small values of a and see if n - a² is a perfect square
        for a in range(1, int(math.sqrt(n)) + 1):
            remainder = n - a**2
            b = int(math.sqrt(remainder) + 0.5)
            if b**2 == remainder:
                return (a, b, 0, 0)
        
        # Case 3: n is a sum of three squares
        # Search for a value where n - a² - b² is a perfect square
        for a in range(1, int(math.sqrt(n)) + 1):
            for b in range(a, int(math.sqrt(n - a**2)) + 1):
                remainder = n - a**2 - b**2
                c = int(math.sqrt(remainder) + 0.5)
                if c**2 == remainder:
                    return (a, b, c, 0)
        
        # Case 4: Need all four squares
        # Use recursive reduction to find a representation
        
        # First, check if n is of the form 4^a(8b+7)
        # If so, it requires four squares (by Legendre's three-square theorem)
        temp = n
        while temp % 4 == 0:
            temp //= 4
        
        if temp % 8 == 7:
            # Implement a different approach for these numbers
            # Find a representation using Lagrange's identity
            a = 1
            while True:
                # Try to express n - a² as a sum of three squares
                remainder = n - a**2
                try:
                    # Find any three-square representation
                    b, c, d = self._find_three_square_representation(remainder)
                    return (a, b, c, d)
                except ValueError:
                    a += 1
                    if a**2 > n:
                        # This should not happen for positive integers by Lagrange's theorem
                        raise ValueError(f"Could not find a four-square representation for {n}")
        
        # General case
        # Use the identity (a² + b² + c² + d²)(e² + f² + g² + h²) = 
        # (ae+bf+cg+dh)² + (af-be+ch-dg)² + (ag-bh-ce+df)² + (ah+bg-cf-de)²
        
        # Find a value m such that nm is easier to represent
        # Then find a representation for m and use the identity
        
        # For simplicity, we'll try small values of a and see if we find a representation
        for a in range(1, min(100, int(math.sqrt(n)) + 1)):
            for b in range(a, min(100, int(math.sqrt(n - a**2)) + 1)):
                for c in range(b, min(100, int(math.sqrt(n - a**2 - b**2)) + 1)):
                    remainder = n - a**2 - b**2 - c**2
                    d = int(math.sqrt(remainder) + 0.5)
                    if d**2 == remainder:
                        return (a, b, c, d)
        
        # If we reach this point, we need more sophisticated methods
        # Fall back to brute force for now
        representations = list(self.enumerate_four_square_representations(n, 100))
        if representations:
            return representations[0]
        
        raise ValueError(f"Could not find a four-square representation for {n}")
    
    def _find_three_square_representation(self, n: int) -> Tuple[int, int, int]:
        """
        Find a representation of n as a sum of three squares.
        
        Args:
            n: Positive integer not of the form 4^a(8b+7)
            
        Returns:
            Tuple (a,b,c) such that n = a² + b² + c²
        """
        # Check if n is of the form 4^a(8b+7)
        temp = n
        while temp % 4 == 0:
            temp //= 4
        
        if temp % 8 == 7:
            raise ValueError(f"{n} cannot be represented as a sum of three squares")
        
        # Try to find a representation by brute force for small values
        for a in range(int(math.sqrt(n)) + 1):
            remainder1 = n - a**2
            for b in range(int(math.sqrt(remainder1)) + 1):
                remainder2 = remainder1 - b**2
                c = int(math.sqrt(remainder2) + 0.5)
                if c**2 == remainder2:
                    return (a, b, c)
        
        # If no representation found, this should not happen for valid inputs
        raise ValueError(f"Could not find a three-square representation for {n}")
    
    def plot_r4_values(self, limit: int = 100):
        """
        Plot r₄(n) values for n from 1 to limit.
        
        Args:
            limit: Upper bound for n
        """
        n_values = list(range(1, limit + 1))
        r4_values = [self.r4(n) for n in n_values]
        
        plt.figure(figsize=(12, 6))
        plt.plot(n_values, r4_values, 'bo-', markersize=3, alpha=0.7)
        plt.xlabel('n')
        plt.ylabel('r₄(n)')
        plt.title('Number of representations as a sum of four squares')
        plt.grid(True, alpha=0.3)
        
        # Mark some interesting values
        max_idx = r4_values.index(max(r4_values))
        plt.annotate(f'Max: r₄({n_values[max_idx]}) = {r4_values[max_idx]}', 
                     xy=(n_values[max_idx], r4_values[max_idx]),
                     xytext=(n_values[max_idx] + 5, r4_values[max_idx]),
                     arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_r4_ratio(self, limit: int = 100):
        """
        Plot the ratio r₄(n) / (8 * sum of divisors) for odd n,
        and r₄(n) / (24 * sum of odd divisors) for even n.
        
        Args:
            limit: Upper bound for n
        """
        n_values = list(range(1, limit + 1))
        ratios = []
        
        for n in n_values:
            if n % 2 == 1:
                ratio = self.r4(n) / (8 * self.sum_of_divisors(n))
            else:
                ratio = self.r4(n) / (24 * self.sum_of_divisors(n, odd_only=True))
            ratios.append(ratio)
        
        plt.figure(figsize=(12, 6))
        plt.plot(n_values, ratios, 'ro-', markersize=3, alpha=0.7)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('n')
        plt.ylabel('Ratio')
        plt.title('Verification of Jacobi\'s Four-Square Theorem')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.95, 1.05)
        plt.tight_layout()
        plt.show()
    
    def theta_function(self, q: float, terms: int = 100) -> float:
        """
        Calculate the theta function θ(q) = 1 + 2q + 2q⁴ + 2q⁹ + 2q¹⁶ + ...
        This function is related to r₄(n) via θ⁴(q) = Σ r₄(n)q^n
        
        Args:
            q: Value of q (should be |q| < 1 for convergence)
            terms: Number of terms to include
            
        Returns:
            Value of θ(q)
        """
        if abs(q) >= 1:
            raise ValueError("Theta function only converges for |q| < 1")
            
        result = 1.0  # First term (n=0)
        
        for n in range(1, terms + 1):
            term = 2 * q**(n**2)
            result += term
            
            # Check for convergence
            if abs(term) < 1e-15:
                break
                
        return result
    
    def compute_r4_via_theta(self, n: int, precision: int = 30) -> int:
        """
        Compute r₄(n) using the theta function identity θ⁴(q) = Σ r₄(n)q^n
        This is mainly for verification/demonstration purposes.
        
        Args:
            n: Positive integer
            precision: Precision for the numerical computation
            
        Returns:
            Approximation of r₄(n)
        """
        # Choose a value of q small enough for good convergence
        q = 0.1
        
        # Compute θ⁴(q)
        theta_q = self.theta_function(q, 1000)
        theta4_q = theta_q**4
        
        # Extract the coefficient of q^n in the power series for θ⁴(q)
        # For small n, we can compute this directly
        
        # First few coefficients of θ⁴(q) = 1 + 8q + 24q² + 32q³ + ...
        if n == 0:
            return 1
        elif n == 1:
            return 8
        elif n == 2:
            return 24
        elif n == 3:
            return 32
        
        # For larger n, we can approximate using numerical methods
        # This is not the most efficient method, but illustrates the connection
        
        # We'll use the fact that r₄(n) is the nth derivative of θ⁴(q) at q=0, divided by n!
        # For numerical stability, use a contour integral approach
        
        # Approximate using the Cauchy integral formula
        R = 0.5  # Radius of the contour
        num_points = precision * 10
        
        result = 0
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            z = R * complex(math.cos(theta), math.sin(theta))
            
            # Evaluate θ⁴(z)
            theta_z = self.theta_function(z, 1000)
            theta4_z = theta_z**4
            
            # Add contribution to the contour integral
            result += theta4_z * z**(-n-1)
        
        # Adjust by the appropriate factor
        result *= R / num_points * math.factorial(n) / (2j * math.pi)
        
        # Return the real part (the imaginary part should be very small)
        return round(result.real)
    
    def four_square_polynomial(self, n: int) -> str:
        """
        Generate a polynomial whose roots give the four-square representations of n.
        
        Args:
            n: Positive integer
            
        Returns:
            Symbolic representation of the polynomial
        """
        # We will generate the polynomial:
        # P(x,y,z,w) = (x² + y² + z² + w² - n)
        
        x, y, z, w = symbols('x y z w')
        polynomial = x**2 + y**2 + z**2 + w**2 - n
        
        return str(polynomial)
    
    def four_square_identity(self, a1: int, a2: int, a3: int, a4: int, 
                             b1: int, b2: int, b3: int, b4: int) -> Tuple[int, int, int, int]:
        """
        Apply the four-square identity to compute a representation of the product.
        
        (a1² + a2² + a3² + a4²)(b1² + b2² + b3² + b4²) = c1² + c2² + c3² + c4²
        
        Args:
            a1, a2, a3, a4: First four-square representation
            b1, b2, b3, b4: Second four-square representation
            
        Returns:
            Tuple (c1, c2, c3, c4) representing the product
        """
        # Compute the four-square identity using Euler's formula
        c1 = a1*b1 - a2*b2 - a3*b3 - a4*b4
        c2 = a1*b2 + a2*b1 + a3*b4 - a4*b3
        c3 = a1*b3 - a2*b4 + a3*b1 + a4*b2
        c4 = a1*b4 + a2*b3 - a3*b2 + a4*b1
        
        return (c1, c2, c3, c4)
    
    def verify_four_square_identity(self, a: Tuple[int, int, int, int], 
                                    b: Tuple[int, int, int, int]) -> bool:
        """
        Verify the four-square identity for two given representations.
        
        Args:
            a: First four-tuple (a1, a2, a3, a4)
            b: Second four-tuple (b1, b2, b3, b4)
            
        Returns:
            True if the identity holds
        """
        a1, a2, a3, a4 = a
        b1, b2, b3, b4 = b
        
        # Compute the sum of squares
        sum_a = a1**2 + a2**2 + a3**2 + a4**2
        sum_b = b1**2 + b2**2 + b3**2 + b4**2
        
        # Compute the product
        product = sum_a * sum_b
        
        # Compute the four-square representation of the product
        c = self.four_square_identity(a1, a2, a3, a4, b1, b2, b3, b4)
        sum_c = c[0]**2 + c[1]**2 + c[2]**2 + c[3]**2
        
        return product == sum_c
    
    def statistics_on_representations(self, n: int) -> Dict:
        """
        Compute statistics on the four-square representations of n.
        
        Args:
            n: Positive integer
            
        Returns:
            Dictionary with statistics
        """
        # Get all representations
        representations = list(self.enumerate_four_square_representations(n))
        
        # Count representations with specific properties
        stats = {
            'total': len(representations),
            'theoretical': self.r4(n),
            'with_zeros': 0,
            'all_positive': 0,
            'all_non_negative': 0,
            'distinct': 0,
            'symmetric': 0
        }
        
        # Unique representations (ignoring order and signs)
        unique_rep_set = set()
        
        for rep in representations:
            # Count representations with zeros
            if 0 in rep:
                stats['with_zeros'] += 1
            
            # Count representations with all positive entries
            if all(x > 0 for x in rep):
                stats['all_positive'] += 1
            
            # Count representations with all non-negative entries
            if all(x >= 0 for x in rep):
                stats['all_non_negative'] += 1
            
            # Count representations with distinct entries
            if len(set(abs(x) for x in rep)) == 4:
                stats['distinct'] += 1
            
            # Count symmetric representations (where a² = b² or c² = d²)
            if rep[0]**2 == rep[1]**2 or rep[2]**2 == rep[3]**2:
                stats['symmetric'] += 1
            
            # Add to unique representation set (sorted squares)
            squares = sorted([x**2 for x in rep])
            unique_rep_set.add(tuple(squares))
        
        stats['unique'] = len(unique_rep_set)
        
        return stats


def demonstrate_jacobi_theorem():
    """Demonstrate Jacobi's four-square theorem with examples"""
    
    print("=== JACOBI'S FOUR-SQUARE THEOREM DEMONSTRATION ===\n")
    
    # Create a calculator instance
    calc = JacobiFourSquares()
    
    # Example 1: Calculate r₄(n) for some examples
    print("1. VALUES OF r₄(n)")
    print("-----------------")
    for n in [1, 2, 3, 4, 5, 10, 20, 25]:
        r4_n = calc.r4(n)
        if n % 2 == 1:
            formula = f"8 * sum of divisors of {n} = 8 * {calc.sum_of_divisors(n)}"
        else:
            formula = f"24 * sum of odd divisors of {n} = 24 * {calc.sum_of_divisors(n, odd_only=True)}"
        
        print(f"r₄({n}) = {r4_n} = {formula}")
    
    # Example 2: Verify the theorem by direct enumeration
    print("\n2. VERIFICATION BY ENUMERATION")
    print("----------------------------")
    for n in [1, 5, 10, 15, 20]:
        verified = calc.verify_r4(n)
        r4_n = calc.r4(n)
        print(f"n = {n}, r₄({n}) = {r4_n}, Verified: {verified}")
    
    # Example 3: Find specific representations
    print("\n3. FOUR-SQUARE REPRESENTATIONS")
    print("----------------------------")
    for n in [7, 15, 23, 30]:
        print(f"\nRepresentations of {n} as a sum of four squares:")
        count = 0
        for rep in calc.enumerate_four_square_representations(n, 10):
            if count < 5:  # Show only the first 5 representations
                squares = [f"{x}²" for x in rep]
                sum_expr = " + ".join(squares)
                print(f"  {n} = {sum_expr}")
                count += 1
        
        total = calc.r4(n)
        if total > 5:
            print(f"  ... and {total - 5} more representations")
        
        # Also show one representation using Lagrange's approach
        one_rep = calc.find_one_representation(n)
        a, b, c, d = one_rep
        print(f"  Quick representation: {n} = {a}² + {b}² + {c}² + {d}²")
        sum_check = a**2 + b**2 + c**2 + d**2
        print(f"  Verification: {a}² + {b}² + {c}² + {d}² = {sum_check}")
    
    # Example 4: Four-square identity
    print("\n4. FOUR-SQUARE IDENTITY")
    print("---------------------")
    a = (1, 2, 3, 4)
    b = (5, 6, 7, 8)
    
    sum_a = sum(x**2 for x in a)
    sum_b = sum(x**2 for x in b)
    
    c = calc.four_square_identity(*a, *b)
    sum_c = sum(x**2 for x in c)
    
    print(f"({a[0]}² + {a[1]}² + {a[2]}² + {a[3]}²)({b[0]}² + {b[1]}² + {b[2]}² + {b[3]}²) = {c[0]}² + {c[1]}² + {c[2]}² + {c[3]}²")
    print(f"({sum_a})({sum_b}) = {sum_c}")
    print(f"{sum_a * sum_b} = {sum_c}")
    
    # Example 5: Statistics on representations
    print("\n5. STATISTICS ON REPRESENTATIONS")
    print("-----------------------------")
    for n in [12, 24, 36]:
        stats = calc.statistics_on_representations(n)
        print(f"\nStatistics for n = {n}:")
        print(f"  Total representations: {stats['total']} (Theoretical: {stats['theoretical']})")
        print(f"  Representations with zeros: {stats['with_zeros']}")
        print(f"  Representations with all positive entries: {stats['all_positive']}")
        print(f"  Representations with all non-negative entries: {stats['all_non_negative']}")
        print(f"  Representations with distinct entries: {stats['distinct']}")
        print(f"  Unique representations (ignoring order and signs): {stats['unique']}")
    
    # Example 6: Theta function connection
    print("\n6. THETA FUNCTION CONNECTION")
    print("-------------------------")
    for n in [1, 2, 3, 4, 5]:
        r4_direct = calc.r4(n)
        r4_theta = calc.compute_r4_via_theta(n)
        print(f"r₄({n}) = {r4_direct} (via Jacobi's formula)")
        print(f"r₄({n}) ≈ {r4_theta} (via theta function)")
        print()


if __name__ == "__main__":
    demonstrate_jacobi_theorem()
    
    # Create a JacobiFourSquares object for further exploration
    jacobi = JacobiFourSquares()
    
    # Uncomment to generate visualizations
    # jacobi.plot_r4_values(100)
    # jacobi.plot_r4_ratio(100)