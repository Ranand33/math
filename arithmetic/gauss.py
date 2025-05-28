import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Set, Tuple, Dict
from collections import defaultdict
import math
from functools import reduce

class ModularRing:
    """Ring of integers modulo n (Z/nZ) as studied by Gauss."""
    
    def __init__(self, n: int):
        self.n = n
        self.elements = list(range(n))
        
    def add(self, a: int, b: int) -> int:
        """Addition in Z/nZ."""
        return (a + b) % self.n
    
    def multiply(self, a: int, b: int) -> int:
        """Multiplication in Z/nZ."""
        return (a * b) % self.n
    
    def power(self, a: int, k: int) -> int:
        """Fast exponentiation: a^k mod n."""
        if k == 0:
            return 1
        result = 1
        base = a % self.n
        while k > 0:
            if k % 2 == 1:
                result = (result * base) % self.n
            base = (base * base) % self.n
            k //= 2
        return result
    
    def is_unit(self, a: int) -> bool:
        """Check if a is a unit (has multiplicative inverse)."""
        return math.gcd(a, self.n) == 1
    
    def multiplicative_inverse(self, a: int) -> int:
        """Find multiplicative inverse using extended Euclidean algorithm."""
        if not self.is_unit(a):
            raise ValueError(f"{a} has no inverse modulo {self.n}")
        
        # Extended Euclidean algorithm
        old_r, r = a, self.n
        old_s, s = 1, 0
        
        while r != 0:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
        
        return old_s % self.n
    
    def addition_table(self):
        """Generate addition table for the ring."""
        table = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                table[i, j] = self.add(i, j)
        return table
    
    def multiplication_table(self):
        """Generate multiplication table for the ring."""
        table = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                table[i, j] = self.multiply(i, j)
        return table

class MultiplicativeGroup:
    """Multiplicative group of integers modulo n, denoted (Z/nZ)*."""
    
    def __init__(self, n: int):
        self.n = n
        self.ring = ModularRing(n)
        self.units = [a for a in range(1, n) if self.ring.is_unit(a)]
        self.order = len(self.units)  # This is Euler's totient φ(n)
        
    def multiply(self, a: int, b: int) -> int:
        """Group operation: multiplication modulo n."""
        return self.ring.multiply(a, b)
    
    def inverse(self, a: int) -> int:
        """Find multiplicative inverse."""
        return self.ring.multiplicative_inverse(a)
    
    def element_order(self, a: int) -> int:
        """Find the order of element a in the group."""
        if a not in self.units:
            raise ValueError(f"{a} is not in the multiplicative group mod {self.n}")
        
        order = 1
        current = a
        while current != 1:
            current = self.multiply(current, a)
            order += 1
        return order
    
    def is_cyclic(self) -> bool:
        """Check if the group is cyclic."""
        for g in self.units:
            if self.element_order(g) == self.order:
                return True
        return False
    
    def find_generators(self) -> List[int]:
        """Find all generators (primitive roots) if the group is cyclic."""
        generators = []
        for g in self.units:
            if self.element_order(g) == self.order:
                generators.append(g)
        return generators
    
    def subgroups(self) -> Dict[int, List[int]]:
        """Find all subgroups and their orders."""
        subgroups = defaultdict(list)
        
        for g in self.units:
            # Generate cyclic subgroup <g>
            subgroup = set()
            current = g
            while current not in subgroup:
                subgroup.add(current)
                current = self.multiply(current, g)
            
            subgroups[len(subgroup)].append(sorted(list(subgroup)))
        
        # Remove duplicates
        for order in subgroups:
            unique = []
            for sg in subgroups[order]:
                if sg not in unique:
                    unique.append(sg)
            subgroups[order] = unique
        
        return dict(subgroups)
    
    def cayley_table(self) -> np.ndarray:
        """Generate Cayley table for the multiplicative group."""
        size = len(self.units)
        table = np.zeros((size, size), dtype=int)
        
        for i, a in enumerate(self.units):
            for j, b in enumerate(self.units):
                product = self.multiply(a, b)
                table[i, j] = self.units.index(product)
        
        return table

class FermatTheorem:
    """Implementation of Fermat's Little Theorem and its consequences."""
    
    @staticmethod
    def fermat_little_theorem(a: int, p: int) -> bool:
        """Verify Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p."""
        if not FermatTheorem.is_prime(p):
            raise ValueError(f"{p} is not prime")
        
        if a % p == 0:
            return True  # Special case: a ≡ 0 (mod p)
        
        ring = ModularRing(p)
        return ring.power(a, p - 1) == 1
    
    @staticmethod
    def euler_theorem(a: int, n: int) -> bool:
        """Verify Euler's generalization: a^φ(n) ≡ 1 (mod n) for gcd(a,n)=1."""
        if math.gcd(a, n) != 1:
            raise ValueError(f"gcd({a}, {n}) != 1")
        
        phi = FermatTheorem.euler_totient(n)
        ring = ModularRing(n)
        return ring.power(a, phi) == 1
    
    @staticmethod
    def euler_totient(n: int) -> int:
        """Calculate Euler's totient function φ(n)."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def primitive_root_exists(n: int) -> bool:
        """Check if primitive roots exist modulo n (Gauss's criterion)."""
        if n == 1 or n == 2 or n == 4:
            return True
        
        # Check if n = p^k for odd prime p
        if n % 2 == 0:
            n //= 2
            if n % 2 == 0:
                return False
        
        # Check if n is a prime power
        for p in range(3, int(n**0.5) + 1, 2):
            if n % p == 0:
                while n % p == 0:
                    n //= p
                return n == 1
        
        return True

class CyclicGroup:
    """General implementation of cyclic groups as discovered by Gauss."""
    
    def __init__(self, n: int, generator: int = 1):
        self.order = n
        self.generator = generator
        self.elements = self._generate_elements()
        
    def _generate_elements(self) -> List[int]:
        """Generate all elements of the cyclic group."""
        elements = []
        current = self.generator
        for i in range(self.order):
            elements.append(current)
            current = (current * self.generator) % (self.order + 1)
        return elements
    
    def operation(self, a: int, b: int) -> int:
        """Group operation (addition for cyclic groups)."""
        return (a + b) % self.order
    
    def inverse(self, a: int) -> int:
        """Find inverse of element a."""
        return (self.order - a) % self.order
    
    def subgroup_lattice(self) -> Dict[int, List[range]]:
        """Find all subgroups (they correspond to divisors of n)."""
        divisors = [d for d in range(1, self.order + 1) if self.order % d == 0]
        subgroups = {}
        
        for d in divisors:
            # Subgroup of order d
            step = self.order // d
            subgroups[d] = list(range(0, self.order, step))
        
        return subgroups

class GaussianIntegers:
    """Gaussian integers a + bi as studied by Gauss."""
    
    def __init__(self, real: int, imag: int):
        self.real = real
        self.imag = imag
    
    def __str__(self):
        if self.imag >= 0:
            return f"{self.real} + {self.imag}i"
        else:
            return f"{self.real} - {-self.imag}i"
    
    def norm(self) -> int:
        """Norm N(a + bi) = a² + b²."""
        return self.real**2 + self.imag**2
    
    def multiply(self, other: 'GaussianIntegers') -> 'GaussianIntegers':
        """Multiply two Gaussian integers."""
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return GaussianIntegers(real, imag)
    
    @staticmethod
    def sum_of_two_squares(p: int) -> Tuple[int, int]:
        """Express prime p ≡ 1 (mod 4) as sum of two squares (Gauss)."""
        if not FermatTheorem.is_prime(p):
            raise ValueError(f"{p} is not prime")
        if p == 2:
            return (1, 1)
        if p % 4 != 1:
            raise ValueError(f"{p} ≡ {p % 4} (mod 4), not 1")
        
        # Find a such that a² ≡ -1 (mod p)
        for a in range(2, p):
            if pow(a, (p - 1) // 2, p) == p - 1:  # a^((p-1)/2) ≡ -1 (mod p)
                break
        
        # Use continued fractions
        m = int(p**0.5)
        prev_x, x = 1, a
        prev_y, y = 0, 1
        
        while True:
            k = (prev_x * prev_x + prev_y * prev_y) % p
            if k == 0:
                return (abs(prev_x), abs(prev_y))
            
            q = ((prev_x * x + prev_y * y) // k) % p
            prev_x, x = x, (prev_x - q * x) % p
            prev_y, y = y, (prev_y - q * y) % p

def visualize_multiplicative_group(n: int):
    """Visualize the structure of (Z/nZ)*."""
    group = MultiplicativeGroup(n)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cayley table
    table = group.cayley_table()
    im1 = ax1.imshow(table, cmap='viridis')
    ax1.set_title(f'Cayley Table of (Z/{n}Z)*')
    ax1.set_xlabel('Element index')
    ax1.set_ylabel('Element index')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Product index')
    
    # 2. Element orders
    orders = [group.element_order(a) for a in group.units]
    ax2.bar(range(len(group.units)), orders)
    ax2.set_title('Orders of Elements')
    ax2.set_xlabel('Element index')
    ax2.set_ylabel('Order')
    ax2.set_xticks(range(len(group.units)))
    ax2.set_xticklabels(group.units, rotation=45)
    
    # 3. Subgroup lattice
    subgroups = group.subgroups()
    G = nx.DiGraph()
    
    # Add nodes for each subgroup
    for order, sgs in subgroups.items():
        for i, sg in enumerate(sgs):
            node_id = f"{order}_{i}"
            G.add_node(node_id, order=order, elements=sg)
    
    # Add edges for subgroup relations
    for order1, sgs1 in subgroups.items():
        for order2, sgs2 in subgroups.items():
            if order1 < order2 and order2 % order1 == 0:
                for i, sg1 in enumerate(sgs1):
                    for j, sg2 in enumerate(sgs2):
                        if all(elem in sg2 for elem in sg1):
                            G.add_edge(f"{order1}_{i}", f"{order2}_{j}")
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax3, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=10, arrows=True)
    ax3.set_title('Subgroup Lattice')
    
    # 4. Powers of elements
    ax4.set_title('Powers of Elements')
    for i, a in enumerate(group.units[:5]):  # Show first 5 units
        powers = []
        labels = []
        current = 1
        for k in range(group.element_order(a)):
            current = group.multiply(current, a)
            powers.append(current)
            labels.append(f"{a}^{k+1}")
        
        ax4.plot(range(len(powers)), powers, 'o-', label=f'a = {a}')
    
    ax4.set_xlabel('Power')
    ax4.set_ylabel('Result')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_gauss_discoveries():
    """Demonstrate Gauss's key discoveries from 1801."""
    
    print("=== GAUSS'S 1801 DISCOVERIES ===\n")
    
    # 1. Fermat's Little Theorem
    print("1. FERMAT'S LITTLE THEOREM")
    print("-" * 40)
    p = 17
    print(f"For prime p = {p}:")
    for a in [2, 3, 5, 7]:
        result = FermatTheorem.fermat_little_theorem(a, p)
        ring = ModularRing(p)
        value = ring.power(a, p - 1)
        print(f"  {a}^{p-1} ≡ {value} (mod {p}) - Verified: {result}")
    
    # 2. Ring of integers modulo n
    print("\n\n2. RING OF INTEGERS MODULO n")
    print("-" * 40)
    n = 12
    ring = ModularRing(n)
    print(f"Z/{n}Z = {{0, 1, 2, ..., {n-1}}}")
    print(f"\nUnits (invertible elements): {[a for a in range(n) if ring.is_unit(a)]}")
    print(f"\nExample operations:")
    print(f"  7 + 8 ≡ {ring.add(7, 8)} (mod {n})")
    print(f"  7 × 8 ≡ {ring.multiply(7, 8)} (mod {n})")
    print(f"  5^(-1) ≡ {ring.multiplicative_inverse(5)} (mod {n})")
    
    # 3. Multiplicative group
    print("\n\n3. MULTIPLICATIVE GROUP (Z/nZ)*")
    print("-" * 40)
    for n in [7, 8, 15]:
        group = MultiplicativeGroup(n)
        print(f"\n(Z/{n}Z)* = {group.units}")
        print(f"Order: {group.order} = φ({n})")
        print(f"Is cyclic: {group.is_cyclic()}")
        if group.is_cyclic():
            print(f"Generators: {group.find_generators()}")
    
    # 4. Primitive roots and cyclic structure
    print("\n\n4. PRIMITIVE ROOTS (GAUSS'S THEOREM)")
    print("-" * 40)
    print("n has primitive roots if and only if n = 1, 2, 4, p^k, or 2p^k")
    for n in [1, 2, 4, 5, 6, 7, 8, 9, 10, 14, 18, 25]:
        exists = FermatTheorem.primitive_root_exists(n)
        group = MultiplicativeGroup(n)
        is_cyclic = group.is_cyclic()
        print(f"n = {n:2d}: Primitive roots exist: {exists}, "
              f"(Z/{n}Z)* is cyclic: {is_cyclic}")
    
    # 5. Gaussian integers and sum of two squares
    print("\n\n5. GAUSSIAN INTEGERS AND SUM OF TWO SQUARES")
    print("-" * 40)
    print("Primes p ≡ 1 (mod 4) can be written as a² + b²:")
    for p in [5, 13, 17, 29, 37, 41]:
        if FermatTheorem.is_prime(p) and p % 4 == 1:
            a, b = GaussianIntegers.sum_of_two_squares(p)
            print(f"  {p} = {a}² + {b}² = {a**2} + {b**2}")
    
    # 6. Quadratic reciprocity preparation
    print("\n\n6. QUADRATIC RESIDUES (Leading to Reciprocity)")
    print("-" * 40)
    p = 11
    ring = ModularRing(p)
    residues = set()
    for a in range(1, p):
        residues.add(ring.power(a, 2))
    print(f"Quadratic residues mod {p}: {sorted(residues)}")
    print(f"Number of residues: {len(residues)} = (p-1)/2")
    
    # Visualize a multiplicative group
    print("\n\nVisualizing (Z/20Z)*...")
    visualize_multiplicative_group(20)

if __name__ == "__main__":
    demonstrate_gauss_discoveries()