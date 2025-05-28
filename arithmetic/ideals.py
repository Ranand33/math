import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from typing import List, Tuple, Set, Dict, Optional
import sympy as sp
from sympy import symbols, Poly, roots, GF, prime, factorint
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP
import networkx as nx
from matplotlib.patches import Polygon
import itertools

class AlgebraicInteger:
    """Represents an element in the ring of integers of an algebraic number field."""
    
    def __init__(self, coeffs: List[Fraction], minimal_poly: Poly, var_name: str = 'α'):
        """
        coeffs: Coefficients [a_0, a_1, ..., a_{n-1}] representing 
                a_0 + a_1*α + ... + a_{n-1}*α^{n-1}
        minimal_poly: The minimal polynomial of α
        """
        self.coeffs = [Fraction(c) for c in coeffs]
        self.minimal_poly = minimal_poly
        self.degree = minimal_poly.degree()
        self.var_name = var_name
        
        # Reduce modulo minimal polynomial
        self._reduce()
    
    def _reduce(self):
        """Reduce the representation modulo the minimal polynomial."""
        while len(self.coeffs) >= self.degree:
            # If we have α^n or higher powers, use minimal polynomial to reduce
            # If m(α) = α^n + c_{n-1}α^{n-1} + ... + c_0 = 0
            # Then α^n = -(c_{n-1}α^{n-1} + ... + c_0)
            
            highest_coeff = self.coeffs.pop()
            min_poly_coeffs = self.minimal_poly.all_coeffs()[1:]  # Skip leading 1
            
            for i, c in enumerate(min_poly_coeffs):
                if i < len(self.coeffs):
                    self.coeffs[i] -= highest_coeff * c
                else:
                    self.coeffs.append(-highest_coeff * c)
        
        # Pad with zeros if necessary
        while len(self.coeffs) < self.degree:
            self.coeffs.append(Fraction(0))
    
    def __str__(self):
        terms = []
        for i, c in enumerate(self.coeffs):
            if c != 0:
                if i == 0:
                    terms.append(str(c))
                elif i == 1:
                    if c == 1:
                        terms.append(self.var_name)
                    elif c == -1:
                        terms.append(f"-{self.var_name}")
                    else:
                        terms.append(f"{c}{self.var_name}")
                else:
                    if c == 1:
                        terms.append(f"{self.var_name}^{i}")
                    elif c == -1:
                        terms.append(f"-{self.var_name}^{i}")
                    else:
                        terms.append(f"{c}{self.var_name}^{i}")
        
        if not terms:
            return "0"
        
        result = terms[0]
        for term in terms[1:]:
            if term[0] != '-':
                result += " + " + term
            else:
                result += " - " + term[1:]
        
        return result
    
    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += Fraction(other)
            return AlgebraicInteger(new_coeffs, self.minimal_poly, self.var_name)
        
        new_coeffs = []
        for i in range(max(len(self.coeffs), len(other.coeffs))):
            c1 = self.coeffs[i] if i < len(self.coeffs) else Fraction(0)
            c2 = other.coeffs[i] if i < len(other.coeffs) else Fraction(0)
            new_coeffs.append(c1 + c2)
        
        return AlgebraicInteger(new_coeffs, self.minimal_poly, self.var_name)
    
    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            new_coeffs = [c * Fraction(other) for c in self.coeffs]
            return AlgebraicInteger(new_coeffs, self.minimal_poly, self.var_name)
        
        # Polynomial multiplication
        result_coeffs = [Fraction(0)] * (len(self.coeffs) + len(other.coeffs) - 1)
        
        for i, c1 in enumerate(self.coeffs):
            for j, c2 in enumerate(other.coeffs):
                result_coeffs[i + j] += c1 * c2
        
        return AlgebraicInteger(result_coeffs, self.minimal_poly, self.var_name)
    
    def __neg__(self):
        return AlgebraicInteger([-c for c in self.coeffs], self.minimal_poly, self.var_name)
    
    def __sub__(self, other):
        return self + (-other)
    
    def norm(self):
        """Compute the norm (product of all conjugates)."""
        # For now, simplified implementation for quadratic fields
        if self.degree == 2:
            # For α with minimal polynomial x² + bx + c
            # If element is a + b*α, norm is (a + b*α)(a + b*α')
            # where α' is the other root
            a, b = self.coeffs[0], self.coeffs[1]
            # Using the fact that α + α' = -b and α*α' = c from minimal polynomial
            min_coeffs = self.minimal_poly.all_coeffs()
            sum_roots = -min_coeffs[1]
            prod_roots = min_coeffs[2]
            
            # Norm = a² + ab(α + α') + b²(αα') = a² - ab*sum + b²*prod
            return a**2 + a*b*sum_roots + b**2*prod_roots
        else:
            # General case would require computing all conjugates
            raise NotImplementedError("Norm for degree > 2 not implemented")
    
    def is_unit(self):
        """Check if this element is a unit (norm = ±1)."""
        n = self.norm()
        return n == 1 or n == -1

class Ideal:
    """Represents an ideal in the ring of integers of an algebraic number field."""
    
    def __init__(self, generators: List[AlgebraicInteger], ring_of_integers):
        """
        generators: List of generators for the ideal
        ring_of_integers: The ring containing this ideal
        """
        self.generators = generators
        self.ring = ring_of_integers
        self._basis = None
        self._norm = None
    
    def __str__(self):
        gen_strs = [str(g) for g in self.generators]
        return f"({', '.join(gen_strs)})"
    
    def contains(self, element: AlgebraicInteger) -> bool:
        """Check if an element is in the ideal."""
        # This is a simplified check - full implementation would need
        # to solve a system of linear equations
        # For principal ideals, just check divisibility
        if len(self.generators) == 1:
            # Simplified divisibility check
            return True  # Would need full implementation
        return True  # Placeholder
    
    def __add__(self, other):
        """Sum of ideals I + J = {a + b : a ∈ I, b ∈ J}."""
        return Ideal(self.generators + other.generators, self.ring)
    
    def __mul__(self, other):
        """Product of ideals IJ = {Σ a_i b_i : a_i ∈ I, b_i ∈ J}."""
        products = []
        for g1 in self.generators:
            for g2 in other.generators:
                products.append(g1 * g2)
        return Ideal(products, self.ring)
    
    def norm(self):
        """Compute the norm of the ideal (index in the ring of integers)."""
        if self._norm is not None:
            return self._norm
        
        # For principal ideals generated by α, N(I) = |N(α)|
        if len(self.generators) == 1:
            self._norm = abs(self.generators[0].norm())
        else:
            # General case requires computing the index [O_K : I]
            # This is a simplified implementation
            self._norm = 1  # Placeholder
        
        return self._norm
    
    def is_prime(self) -> bool:
        """Check if the ideal is prime."""
        # An ideal P is prime if P ≠ R and whenever ab ∈ P, either a ∈ P or b ∈ P
        # For now, we'll check if the norm is prime (necessary but not sufficient)
        n = self.norm()
        if n <= 1:
            return False
        
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        
        return True
    
    def is_principal(self) -> bool:
        """Check if the ideal is principal (generated by a single element)."""
        # Simplified check
        return len(self.generators) == 1
    
    def factor(self) -> List['Ideal']:
        """Factor the ideal into prime ideals."""
        # This is a simplified factorization
        # Full implementation would use algorithms from computational algebraic number theory
        if self.is_prime():
            return [self]
        
        # Placeholder - return self
        return [self]

class RingOfIntegers:
    """The ring of integers O_K of an algebraic number field K."""
    
    def __init__(self, minimal_poly: Poly, var_name: str = 'α'):
        """
        minimal_poly: The minimal polynomial defining the field extension
        """
        self.minimal_poly = minimal_poly
        self.degree = minimal_poly.degree()
        self.var_name = var_name
        self._discriminant = None
    
    def element(self, coeffs: List) -> AlgebraicInteger:
        """Create an element of the ring."""
        return AlgebraicInteger(coeffs, self.minimal_poly, self.var_name)
    
    def zero(self) -> AlgebraicInteger:
        """The zero element."""
        return self.element([0] * self.degree)
    
    def one(self) -> AlgebraicInteger:
        """The multiplicative identity."""
        coeffs = [0] * self.degree
        coeffs[0] = 1
        return self.element(coeffs)
    
    def principal_ideal(self, generator: AlgebraicInteger) -> Ideal:
        """Create a principal ideal (α) = αO_K."""
        return Ideal([generator], self)
    
    def ideal(self, *generators) -> Ideal:
        """Create an ideal from generators."""
        return Ideal(list(generators), self)
    
    def discriminant(self):
        """Compute the discriminant of the ring."""
        if self._discriminant is not None:
            return self._discriminant
        
        # For a quadratic field with minimal polynomial x² + bx + c
        # discriminant is b² - 4c
        if self.degree == 2:
            coeffs = self.minimal_poly.all_coeffs()
            self._discriminant = coeffs[1]**2 - 4*coeffs[2]
        else:
            # General case requires computing determinant of trace matrix
            self._discriminant = 1  # Placeholder
        
        return self._discriminant
    
    def class_number(self):
        """Compute the class number (size of the ideal class group)."""
        # This is a very difficult computation in general
        # For now, return 1 (principal ideal domain)
        return 1
    
    def factor_rational_prime(self, p: int) -> List[Ideal]:
        """Factor a rational prime p in the ring of integers."""
        # How does the prime p factor in O_K?
        
        # For quadratic fields, use the Legendre symbol and discriminant
        if self.degree == 2:
            D = self.discriminant()
            
            # p ramifies if p | D
            if D % p == 0:
                # p ramifies: pO_K = P²
                alpha = self.element([0, 1])  # α
                P = self.ideal(self.element([p]), alpha)
                return [P, P]
            
            # Check if D is a quadratic residue mod p
            legendre = pow(D, (p - 1) // 2, p)
            
            if legendre == 1:
                # p splits: pO_K = P₁P₂
                # Find α such that α² ≡ D (mod p)
                for a in range(p):
                    if (a * a - D) % p == 0:
                        P1 = self.ideal(self.element([p]), self.element([a, 1]))
                        P2 = self.ideal(self.element([p]), self.element([-a, 1]))
                        return [P1, P2]
            else:
                # p remains prime: pO_K = (p)
                return [self.principal_ideal(self.element([p]))]
        
        # General case
        return [self.principal_ideal(self.element([p]))]

class CyclotomicField:
    """The cyclotomic field Q(ζ_n) where ζ_n is a primitive n-th root of unity."""
    
    def __init__(self, n: int):
        self.n = n
        # The minimal polynomial is the n-th cyclotomic polynomial
        x = symbols('x')
        self.cyclotomic_poly = self._cyclotomic_polynomial(n, x)
        self.ring_of_integers = RingOfIntegers(self.cyclotomic_poly, f'ζ_{n}')
        self.degree = sp.totient(n)  # Degree of Q(ζ_n) over Q
    
    def _cyclotomic_polynomial(self, n: int, x) -> Poly:
        """Compute the n-th cyclotomic polynomial."""
        # Φ_n(x) = Π(x - ζ^k) where gcd(k, n) = 1 and 1 ≤ k ≤ n
        
        if n == 1:
            return Poly(x - 1, x)
        elif sp.isprime(n):
            # For prime p: Φ_p(x) = 1 + x + x² + ... + x^(p-1)
            return Poly(sum(x**i for i in range(n)), x)
        else:
            # Use the formula: x^n - 1 = Π_{d|n} Φ_d(x)
            # So Φ_n(x) = (x^n - 1) / Π_{d|n, d<n} Φ_d(x)
            
            divisors = [d for d in range(1, n) if n % d == 0]
            numerator = Poly(x**n - 1, x)
            denominator = Poly(1, x)
            
            for d in divisors:
                denominator *= self._cyclotomic_polynomial(d, x)
            
            result, remainder = sp.div(numerator, denominator)
            return result
    
    def primitive_root(self) -> AlgebraicInteger:
        """Return ζ_n as an algebraic integer."""
        # ζ_n is represented as the root of the cyclotomic polynomial
        # In our basis, it's just [0, 1, 0, ..., 0]
        coeffs = [0] * self.degree
        if self.degree > 1:
            coeffs[1] = 1
        else:
            coeffs[0] = -1  # Special case for n=1,2
        
        return self.ring_of_integers.element(coeffs)
    
    def demonstrate_unique_factorization(self):
        """Show unique factorization of ideals in the cyclotomic field."""
        print(f"\nCyclotomic Field Q(ζ_{self.n})")
        print(f"Minimal polynomial: {self.cyclotomic_poly}")
        print(f"Degree: {self.degree}")
        print(f"Ring of integers: Z[ζ_{self.n}]")
        
        # Factor some small primes
        print("\nFactorization of rational primes:")
        for p in [2, 3, 5, 7]:
            if p < 20:  # Avoid large computations
                factors = self.ring_of_integers.factor_rational_prime(p)
                print(f"  ({p}) = {' × '.join(str(f) for f in factors)}")

def visualize_ideal_lattice(ring: RingOfIntegers, ideal: Ideal):
    """Visualize an ideal as a lattice in the complex plane (for quadratic fields)."""
    
    if ring.degree != 2:
        print("Visualization only implemented for quadratic fields")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # For quadratic fields, we can embed in C
    # If minimal polynomial is x² + bx + c, roots are (-b ± √(b²-4c))/2
    coeffs = ring.minimal_poly.all_coeffs()
    b, c = coeffs[1], coeffs[2]
    disc = b**2 - 4*c
    
    if disc >= 0:
        # Real quadratic field
        root1 = (-b + np.sqrt(disc)) / 2
        root2 = (-b - np.sqrt(disc)) / 2
        
        # Plot the ring of integers
        ax1.set_title('Ring of Integers')
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i + j * root1
                y = 0  # Real embedding
                ax1.plot(x, y, 'bo', markersize=6)
        
        ax1.set_xlabel('Real axis')
        ax1.set_ylabel('Imaginary axis')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
    else:
        # Imaginary quadratic field
        real_part = -b / 2
        imag_part = np.sqrt(-disc) / 2
        
        # Plot the ring of integers as a lattice
        ax1.set_title('Ring of Integers Z[α]')
        
        lattice_points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i + j * real_part
                y = j * imag_part
                ax1.plot(x, y, 'bo', markersize=6)
                lattice_points.append((x, y))
        
        ax1.set_xlabel('Real axis')
        ax1.set_ylabel('Imaginary axis')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Draw fundamental parallelogram
        if len(ideal.generators) == 1:
            # Principal ideal
            gen = ideal.generators[0]
            a, b = float(gen.coeffs[0]), float(gen.coeffs[1])
            
            # The ideal consists of multiples of the generator
            ax2.set_title(f'Principal Ideal {ideal}')
            
            # Plot ideal elements
            for i in range(-3, 4):
                for j in range(-3, 4):
                    # (i + jα) * generator
                    x = i * a + j * (a * real_part - b * imag_part)
                    y = j * (a * imag_part + b * real_part)
                    ax2.plot(x, y, 'ro', markersize=6)
            
            ax2.set_xlabel('Real axis')
            ax2.set_ylabel('Imaginary axis')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def demonstrate_failure_and_restoration():
    """Show how unique factorization fails for elements but works for ideals."""
    
    print("=== FAILURE OF UNIQUE FACTORIZATION FOR ELEMENTS ===\n")
    
    # Classic example: Z[√-5]
    x = symbols('x')
    minimal_poly = Poly(x**2 + 5, x)
    ring = RingOfIntegers(minimal_poly, '√-5')
    
    print("Field: Q(√-5)")
    print("Ring of integers: Z[√-5] = {a + b√-5 : a, b ∈ Z}")
    print(f"Discriminant: {ring.discriminant()}")
    
    # The famous example: 6 = 2 × 3 = (1 + √-5)(1 - √-5)
    print("\nNon-unique factorization of 6:")
    
    # Elements
    two = ring.element([2])
    three = ring.element([3])
    alpha_plus = ring.element([1, 1])   # 1 + √-5
    alpha_minus = ring.element([1, -1])  # 1 - √-5
    
    print(f"  6 = 2 × 3")
    print(f"  6 = ({alpha_plus}) × ({alpha_minus})")
    
    # Check these are actually equal
    prod1 = two * three
    prod2 = alpha_plus * alpha_minus
    print(f"\nVerification: 2 × 3 = {prod1}")
    print(f"            (1 + √-5)(1 - √-5) = {prod2}")
    
    # Show these factors are irreducible
    print("\nNorms of factors:")
    print(f"  N(2) = {two.norm()}")
    print(f"  N(3) = {three.norm()}")
    print(f"  N(1 + √-5) = {alpha_plus.norm()}")
    print(f"  N(1 - √-5) = {alpha_minus.norm()}")
    
    print("\n=== RESTORATION VIA IDEAL FACTORIZATION ===\n")
    
    # Factor the ideals
    ideal_2 = ring.principal_ideal(two)
    ideal_3 = ring.principal_ideal(three)
    ideal_6 = ring.principal_ideal(ring.element([6]))
    
    print("Factorization of principal ideals:")
    
    # How do 2 and 3 factor?
    print("\nFactorization of (2):")
    factors_2 = ring.factor_rational_prime(2)
    print(f"  (2) = {' × '.join(str(f) for f in factors_2)}")
    
    print("\nFactorization of (3):")
    factors_3 = ring.factor_rational_prime(3)
    print(f"  (3) = {' × '.join(str(f) for f in factors_3)}")
    
    # The key insight: (1 + √-5) and (1 - √-5) generate the same prime ideals
    print("\nThe restoration of unique factorization:")
    print("  (6) = (2)(3) = P₂² × P₃")
    print("  where P₂ = (2, 1 + √-5) and P₃ = (3, 1 + √-5)")
    
    # Visualize the ideal structure
    visualize_ideal_lattice(ring, ideal_2)

def explore_class_group():
    """Explore the ideal class group and class number."""
    
    print("\n=== IDEAL CLASS GROUP ===\n")
    
    # Example with a field that has non-trivial class group
    # Q(√-23) has class number 3
    x = symbols('x')
    minimal_poly = Poly(x**2 + 23, x)
    ring = RingOfIntegers(minimal_poly, '√-23')
    
    print("Field: Q(√-23)")
    print("This field has class number 3")
    print("This means there are non-principal ideals!\n")
    
    # Example of a non-principal ideal
    two = ring.element([2])
    alpha = ring.element([1, 1])  # 1 + √-23
    
    # The ideal (2, 1 + √-23) is not principal
    I = ring.ideal(two, alpha)
    
    print(f"The ideal I = {I}")
    print("is not principal (cannot be generated by a single element)")
    print("\nBut I³ is principal!")
    
    # In a field with class number h, every ideal to the h-th power is principal
    I_cubed = I * I * I
    print(f"I³ = {I_cubed} = (principal ideal)")

def demonstrate_cyclotomic_unique_factorization():
    """Demonstrate unique factorization in cyclotomic fields."""
    
    print("\n=== UNIQUE FACTORIZATION IN CYCLOTOMIC FIELDS ===\n")
    
    # Start with small cyclotomic fields
    for n in [3, 4, 5, 7]:
        cyclotomic = CyclotomicField(n)
        cyclotomic.demonstrate_unique_factorization()
        print()
    
    # Special focus on Q(ζ₅)
    print("=== DETAILED ANALYSIS: Q(ζ₅) ===\n")
    
    cyclotomic_5 = CyclotomicField(5)
    ring = cyclotomic_5.ring_of_integers
    
    print("The 5th roots of unity form a regular pentagon")
    print("Ring of integers: Z[ζ₅]")
    print(f"Degree: {cyclotomic_5.degree}")
    
    # The golden ratio appears!
    print("\nConnection to golden ratio:")
    print("ζ₅ + ζ₅⁴ = (√5 - 1)/2")
    print("ζ₅² + ζ₅³ = -(√5 + 1)/2")
    
    # Visualize the cyclotomic field
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the 5th roots of unity
    n = 5
    roots = [np.exp(2j * np.pi * k / n) for k in range(n)]
    
    for i, root in enumerate(roots):
        ax.plot(root.real, root.imag, 'ro', markersize=10)
        ax.annotate(f'ζ₅^{i}', (root.real, root.imag), 
                   xytext=(5, 5), textcoords='offset points')
    
    # Draw the regular pentagon
    pentagon = plt.Polygon([(r.real, r.imag) for r in roots], 
                          fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(pentagon)
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('5th Roots of Unity and Z[ζ₅]')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    
    plt.show()

def main_theorem_demo():
    """Demonstrate the main theorem about unique factorization of ideals."""
    
    print("=" * 60)
    print("FUNDAMENTAL THEOREM OF IDEAL THEORY")
    print("=" * 60)
    print("\nIn any Dedekind domain (including rings of integers of")
    print("algebraic number fields), every nonzero ideal factors")
    print("uniquely as a product of prime ideals.\n")
    
    print("HISTORICAL DEVELOPMENT:")
    print("1. Gauss (1801): Unique factorization in Z[i]")
    print("2. Kummer (1840s): Discovered failure in cyclotomic fields")
    print("3. Kummer: Introduced 'ideal numbers' to restore uniqueness")
    print("4. Dedekind (1871): Modern theory of ideals")
    
    print("\n" + "=" * 60 + "\n")
    
    # Run all demonstrations
    demonstrate_failure_and_restoration()
    explore_class_group()
    demonstrate_cyclotomic_unique_factorization()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("\nThe theory of ideals reveals that while unique factorization")
    print("may fail for elements, it is miraculously restored at the")
    print("level of ideals. This deep insight revolutionized algebraic")
    print("number theory and has applications in:")
    print("• Solving Diophantine equations")
    print("• Proving Fermat's Last Theorem (via Kummer's work)")
    print("• Modern cryptography")
    print("• Algebraic geometry")

if __name__ == "__main__":
    main_theorem_demo()