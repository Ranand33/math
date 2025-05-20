import math
from typing import List, Tuple, Optional, Union, Generator
from fractions import Fraction
import sympy

class DiophantineEquation:
    """Base class for Diophantine equations - polynomial equations where only integer solutions are considered"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def is_solution(self, *args) -> bool:
        """Check if given integers form a solution to the equation"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def find_solutions(self, *args, **kwargs) -> List[Tuple[int, ...]]:
        """Find integer solutions to the equation based on given constraints"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class LinearDiophantine(DiophantineEquation):
    """
    Linear Diophantine equation of the form ax + by = c
    Properties:
    - Has infinitely many solutions if gcd(a,b) divides c
    - Has no solutions if gcd(a,b) does not divide c
    - If (x0,y0) is a particular solution, all solutions are of the form:
      x = x0 + (b/d)n, y = y0 - (a/d)n where d = gcd(a,b) and n is an integer
    """
    
    def __init__(self, a: int, b: int, c: int):
        """Initialize a linear Diophantine equation ax + by = c"""
        super().__init__("Linear Diophantine", f"{a}x + {b}y = {c}")
        self.a = a
        self.b = b
        self.c = c
        self.gcd = math.gcd(a, b)
        self.solvable = (c % self.gcd == 0)
        
    def is_solution(self, x: int, y: int) -> bool:
        """Check if x and y satisfy the equation ax + by = c"""
        return self.a * x + self.b * y == self.c
    
    def find_particular_solution(self) -> Optional[Tuple[int, int]]:
        """Find a particular solution using the Extended Euclidean Algorithm"""
        if not self.solvable:
            return None
            
        # If a or b is zero, handle specially
        if self.a == 0:
            if self.b == 0:
                return (0, 0) if self.c == 0 else None
            return (0, self.c // self.b)
        if self.b == 0:
            return (self.c // self.a, 0)
            
        # Use Extended Euclidean Algorithm
        # Find Bézout coefficients s, t such that as + bt = gcd(a,b)
        s, t, _ = self.extended_gcd(self.a, self.b)
        
        # Scale to get c as the RHS
        factor = self.c // self.gcd
        return (s * factor, t * factor)
    
    def extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm
        Returns (s, t, g) such that as + bt = g and g = gcd(a, b)
        """
        if a == 0:
            return 0, 1, b
        
        s1, t1, g = self.extended_gcd(b % a, a)
        s = s1 - (b // a) * t1
        t = t1
        return s, t, g
    
    def generate_solutions(self, limit: int = 5) -> List[Tuple[int, int]]:
        """Generate a sample of solutions within a range around the particular solution"""
        if not self.solvable:
            return []
            
        x0, y0 = self.find_particular_solution()
        solutions = []
        
        # Generate solutions using the formula x = x0 + (b/d)n, y = y0 - (a/d)n
        for n in range(-limit, limit+1):
            x = x0 + (self.b // self.gcd) * n
            y = y0 - (self.a // self.gcd) * n
            solutions.append((x, y))
            
        return solutions
    
    def find_solutions(self, x_min: int, x_max: int, y_min: int, y_max: int) -> List[Tuple[int, int]]:
        """Find all solutions within given bounds for x and y"""
        if not self.solvable:
            return []
            
        x0, y0 = self.find_particular_solution()
        solutions = []
        
        # Calculate range of n values that will keep x and y within bounds
        b_d = self.b // self.gcd
        a_d = self.a // self.gcd
        
        # Calculate n bounds for x
        if b_d != 0:
            n_min_x = math.ceil((x_min - x0) / b_d) if b_d > 0 else math.floor((x_min - x0) / b_d)
            n_max_x = math.floor((x_max - x0) / b_d) if b_d > 0 else math.ceil((x_max - x0) / b_d)
        else:
            if x_min <= x0 <= x_max:
                n_min_x, n_max_x = 0, 0
            else:
                return []
                
        # Calculate n bounds for y
        if a_d != 0:
            n_min_y = math.ceil((y0 - y_max) / a_d) if a_d > 0 else math.floor((y0 - y_max) / a_d)
            n_max_y = math.floor((y0 - y_min) / a_d) if a_d > 0 else math.ceil((y0 - y_min) / a_d)
        else:
            if y_min <= y0 <= y_max:
                n_min_y, n_max_y = 0, 0
            else:
                return []
        
        # Intersection of n ranges
        n_min = max(n_min_x, n_min_y)
        n_max = min(n_max_x, n_max_y)
        
        # Generate all solutions in the range
        for n in range(n_min, n_max + 1):
            x = x0 + b_d * n
            y = y0 - a_d * n
            solutions.append((x, y))
            
        return solutions


class PellEquation(DiophantineEquation):
    """
    Pell's equation: x² - D·y² = 1, where D is a positive non-square integer.
    
    Properties:
    - Always has infinitely many solutions for non-square D > 0
    - Fundamental solution (x₁, y₁) can be found using continued fractions
    - All solutions can be generated from the fundamental solution
    - If (xₙ, yₙ) is the nth solution, then:
      xₙ₊₁ + yₙ₊₁√D = (x₁ + y₁√D)^(n+1)
    """
    
    def __init__(self, D: int):
        """Initialize a Pell equation x² - D·y² = 1"""
        if D <= 0 or math.isqrt(D) ** 2 == D:
            raise ValueError("D must be a positive non-square integer")
        super().__init__("Pell's Equation", f"x² - {D}·y² = 1")
        self.D = D
        
    def is_solution(self, x: int, y: int) -> bool:
        """Check if x and y satisfy x² - D·y² = 1"""
        return x**2 - self.D * y**2 == 1
    
    def find_fundamental_solution(self) -> Tuple[int, int]:
        """Find the smallest positive solution using continued fractions"""
        # Using sympy for continued fraction computation
        sqrt_D = sympy.sqrt(self.D)
        continued_fraction = sympy.continued_fraction_iterator(sqrt_D)
        
        # Variables for convergents
        p_prev, p = 1, next(continued_fraction)
        q_prev, q = 0, 1
        
        # Get period of continued fraction
        period = []
        a = next(continued_fraction)
        period.append(a)
        
        # Compute convergents until we find a solution
        while True:
            p_new = a * p + p_prev
            q_new = a * q + q_prev
            
            # Check if we have a solution
            if p_new**2 - self.D * q_new**2 == 1:
                return (p_new, q_new)
            
            p_prev, p = p, p_new
            q_prev, q = q, q_new
            
            # Get next term in continued fraction
            a = next(continued_fraction)
            period.append(a)
    
    def generate_solutions(self, count: int = 5) -> List[Tuple[int, int]]:
        """Generate the first count solutions for the Pell equation"""
        x1, y1 = self.find_fundamental_solution()
        solutions = [(x1, y1)]
        
        if count <= 1:
            return solutions
            
        # Use recurrence relation to generate more solutions
        x_prev, y_prev = x1, y1
        for _ in range(1, count):
            # (x_next, y_next) = (x1, y1) * (x_prev, y_prev) in the extension field Q(√D)
            x_next = x1 * x_prev + self.D * y1 * y_prev
            y_next = x1 * y_prev + y1 * x_prev
            solutions.append((x_next, y_next))
            x_prev, y_prev = x_next, y_next
            
        return solutions
    
    def find_solutions(self, limit: int = 10**6) -> List[Tuple[int, int]]:
        """Find all solutions with x, y < limit"""
        x1, y1 = self.find_fundamental_solution()
        solutions = []
        
        x, y = 1, 0  # Trivial solution
        while x < limit and y < limit:
            solutions.append((x, y))
            # Use recurrence relation
            x_next = x1 * x + self.D * y1 * y
            y_next = x1 * y + y1 * x
            x, y = x_next, y_next
            
        return solutions


class PythagoreanTriples(DiophantineEquation):
    """
    Pythagorean equation: a² + b² = c²
    
    Properties:
    - Primitive triples have gcd(a,b,c) = 1
    - All primitive triples can be parametrized as:
      a = m² - n², b = 2mn, c = m² + n² 
      where m > n > 0, gcd(m,n) = 1, and exactly one of m,n is even
    - All other triples are multiples of primitive ones
    """
    
    def __init__(self):
        """Initialize the Pythagorean equation a² + b² = c²"""
        super().__init__("Pythagorean Equation", "a² + b² = c²")
    
    def is_solution(self, a: int, b: int, c: int) -> bool:
        """Check if (a,b,c) satisfies a² + b² = c²"""
        return a**2 + b**2 == c**2
    
    def is_primitive(self, a: int, b: int, c: int) -> bool:
        """Check if (a,b,c) is a primitive Pythagorean triple"""
        if not self.is_solution(a, b, c):
            return False
        return math.gcd(math.gcd(a, b), c) == 1
    
    def generate_triples(self, limit: int) -> List[Tuple[int, int, int]]:
        """
        Generate all Pythagorean triples with a, b, c ≤ limit
        Uses Euclid's formula to generate primitive triples, then includes multiples
        """
        triples = []
        
        # Generate primitive triples using Euclid's formula
        for m in range(2, math.isqrt(limit) + 1):
            for n in range(1, m):
                # Ensure m and n are coprime and not both odd (for primitive triples)
                if math.gcd(m, n) == 1 and (m % 2 == 0 or n % 2 == 0):
                    a = m*m - n*n
                    b = 2*m*n
                    c = m*m + n*n
                    
                    # Swap a, b if necessary to ensure a < b
                    if a > b:
                        a, b = b, a
                    
                    # Skip if any value exceeds the limit
                    if c > limit:
                        continue
                    
                    # Add the primitive triple
                    triples.append((a, b, c))
                    
                    # Add non-primitive multiples
                    k = 2
                    while k*c <= limit:
                        triples.append((k*a, k*b, k*c))
                        k += 1
        
        # Sort by c value
        return sorted(triples, key=lambda t: t[2])
    
    def find_solutions(self, c_max: int = 100) -> List[Tuple[int, int, int]]:
        """Find all Pythagorean triples with c ≤ c_max"""
        return self.generate_triples(c_max)


class FermatEquation(DiophantineEquation):
    """
    Fermat's equation: xⁿ + yⁿ = zⁿ
    
    Properties:
    - For n = 1: infinite solutions (linear equation)
    - For n = 2: Pythagorean triples
    - For n ≥ 3: No positive integer solutions (Fermat's Last Theorem)
    """
    
    def __init__(self, n: int):
        """Initialize Fermat's equation xⁿ + yⁿ = zⁿ"""
        super().__init__("Fermat's Equation", f"x^{n} + y^{n} = z^{n}")
        self.n = n
    
    def is_solution(self, x: int, y: int, z: int) -> bool:
        """Check if (x,y,z) satisfies xⁿ + yⁿ = zⁿ"""
        return x**self.n + y**self.n == z**self.n
    
    def find_solutions(self, limit: int = 100) -> List[Tuple[int, int, int]]:
        """
        Find solutions to Fermat's equation with values ≤ limit
        Note: For n ≥ 3, no positive integer solutions exist (Fermat's Last Theorem)
        """
        solutions = []
        
        if self.n == 1:
            # For n=1, there are infinitely many solutions
            # We'll generate a small sample: x + y = z where x ≤ y ≤ z ≤ limit
            for z in range(2, limit + 1):
                for y in range(1, z):
                    x = z - y
                    if x <= y and x > 0:
                        solutions.append((x, y, z))
                        
        elif self.n == 2:
            # For n=2, it's the Pythagorean equation
            pythagorean = PythagoreanTriples()
            solutions = pythagorean.find_solutions(limit)
            
        # For n ≥ 3, according to Fermat's Last Theorem, no positive integer solutions exist
        # We could check some trivial solutions like (0, z, z)
        
        return solutions


def demonstrate_diophantine_properties():
    """Function to demonstrate key properties of Diophantine equations"""
    
    print("=== DIOPHANTINE EQUATIONS DEMONSTRATION ===\n")
    
    # 1. Linear Diophantine Equation
    print("1. LINEAR DIOPHANTINE EQUATION (ax + by = c)")
    print("-------------------------------------------")
    # Example 1: Solvable case
    linear1 = LinearDiophantine(4, 6, 10)
    print(f"Equation: {linear1}")
    print(f"GCD(4, 6) = {linear1.gcd}, which divides 10, so solutions exist.")
    
    particular = linear1.find_particular_solution()
    print(f"Particular solution: {particular}")
    
    print("\nSample solutions:")
    for x, y in linear1.generate_solutions(3):
        print(f"  {4}·{x} + {6}·{y} = {4*x + 6*y}")
    
    # Example 2: Unsolvable case
    linear2 = LinearDiophantine(4, 6, 7)
    print("\nEquation: {0}".format(linear2))
    print(f"GCD(4, 6) = {linear2.gcd}, which does not divide 7, so no solutions exist.")
    
    # 2. Pell's Equation
    print("\n2. PELL'S EQUATION (x² - Dy² = 1)")
    print("----------------------------------")
    pell1 = PellEquation(2)
    print(f"Equation: {pell1}")
    
    fundamental = pell1.find_fundamental_solution()
    print(f"Fundamental solution: {fundamental}")
    print(f"Verification: {fundamental[0]}² - {pell1.D}·{fundamental[1]}² = {fundamental[0]**2 - pell1.D * fundamental[1]**2}")
    
    print("\nFirst few solutions:")
    solutions = pell1.generate_solutions(4)
    for i, (x, y) in enumerate(solutions):
        print(f"  Solution {i+1}: ({x}, {y}), Verification: {x}² - {pell1.D}·{y}² = {x**2 - pell1.D * y**2}")
    
    print("\nRecurrence relation demonstration:")
    x1, y1 = solutions[0]
    x2, y2 = solutions[1]
    print(f"  If (x₁, y₁) = ({x1}, {y1}), then:")
    print(f"  x₂ = {x1}·{x1} + {pell1.D}·{y1}·{y1} = {x2}")
    print(f"  y₂ = {x1}·{y1} + {y1}·{x1} = {y2}")
    
    # 3. Pythagorean Triples
    print("\n3. PYTHAGOREAN TRIPLES (a² + b² = c²)")
    print("-------------------------------------")
    pythag = PythagoreanTriples()
    print(f"Equation: {pythag}")
    
    print("\nPrimitive Pythagorean triples (a < b < c ≤ 50):")
    triples = pythag.find_solutions(50)
    for a, b, c in triples:
        if pythag.is_primitive(a, b, c):
            print(f"  ({a}, {b}, {c}): {a}² + {b}² = {a**2 + b**2} = {c}²")
    
    print("\nParametrization example (using Euclid's formula):")
    print("  For m=2, n=1:")
    m, n = 2, 1
    a = m**2 - n**2
    b = 2*m*n
    c = m**2 + n**2
    print(f"  a = m² - n² = {m}² - {n}² = {a}")
    print(f"  b = 2mn = 2·{m}·{n} = {b}")
    print(f"  c = m² + n² = {m}² + {n}² = {c}")
    print(f"  Verification: {a}² + {b}² = {a**2 + b**2} = {c}² = {c**2}")
    
    # 4. Fermat's Equation
    print("\n4. FERMAT'S EQUATION (xⁿ + yⁿ = zⁿ)")
    print("------------------------------------")
    
    # For n=1 (linear equation)
    fermat1 = FermatEquation(1)
    print(f"Equation: {fermat1}")
    print("This is a linear equation with infinitely many solutions.")
    sols1 = fermat1.find_solutions(10)
    print(f"Sample solutions (with values ≤ 10): {len(sols1)} solutions found")
    for i, (x, y, z) in enumerate(sols1[:5]):
        print(f"  {x}¹ + {y}¹ = {z}¹  ({x + y} = {z})")
    if len(sols1) > 5:
        print(f"  ... and {len(sols1) - 5} more")
    
    # For n=2 (Pythagorean equation)
    fermat2 = FermatEquation(2)
    print(f"\nEquation: {fermat2}")
    print("This is the Pythagorean equation with infinitely many solutions.")
    print("These are the Pythagorean triples shown earlier.")
    
    # For n≥3 (Fermat's Last Theorem)
    fermat3 = FermatEquation(3)
    print(f"\nEquation: {fermat3}")
    print("According to Fermat's Last Theorem (proved by Andrew Wiles in 1994),")
    print("there are no positive integer solutions for n ≥ 3.")


# Run the demonstration
if __name__ == "__main__":
    demonstrate_diophantine_properties()