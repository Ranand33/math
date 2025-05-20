import random
import math
import hashlib
from typing import Tuple, Optional, Union, List
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


class FiniteField:
    """
    Implementation of a finite field GF(p) where p is prime
    """
    def __init__(self, p: int):
        """
        Initialize a finite field of order p
        
        Args:
            p: A prime number that defines the field size
        """
        if not self._is_prime(p):
            raise ValueError(f"{p} is not a prime number")
        self.p = p
        
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime using a simple primality test"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def add(self, a: int, b: int) -> int:
        """Add two elements in the field"""
        return (a + b) % self.p
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract two elements in the field"""
        return (a - b) % self.p
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two elements in the field"""
        return (a * b) % self.p
    
    def divide(self, a: int, b: int) -> int:
        """Divide a by b in the field (a/b)"""
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero in the field")
        # Division is multiplication by the modular multiplicative inverse
        return self.multiply(a, self.inv(b))
    
    def inv(self, a: int) -> int:
        """Calculate the modular multiplicative inverse of a"""
        if a == 0:
            raise ZeroDivisionError("Cannot calculate inverse of zero")
        # Extended Euclidean Algorithm to find modular inverse
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            else:
                gcd, x, y = extended_gcd(b % a, a)
                return gcd, y - (b // a) * x, x
                
        gcd, x, y = extended_gcd(a % self.p, self.p)
        if gcd != 1:
            raise ValueError(f"Modular inverse does not exist for {a} in GF({self.p})")
        else:
            return x % self.p
    
    def power(self, a: int, n: int) -> int:
        """Calculate a^n in the field using square-and-multiply algorithm"""
        result = 1
        a = a % self.p
        while n > 0:
            if n % 2 == 1:
                result = self.multiply(result, a)
            a = self.multiply(a, a)
            n //= 2
        return result
    
    def sqrt(self, a: int) -> Optional[int]:
        """
        Calculate the square root of a in the field
        Returns None if the square root doesn't exist
        Uses Tonelli-Shanks algorithm for p ≡ 1 (mod 4)
        """
        # Handle special cases
        if a == 0:
            return 0
        
        # For p ≡ 3 (mod 4), we can use a simpler formula
        if self.p % 4 == 3:
            r = self.power(a, (self.p + 1) // 4)
            if self.multiply(r, r) == a:
                return r
            return None
        
        # Tonelli-Shanks algorithm for p ≡ 1 (mod 4)
        # 1. Factor p-1 as q * 2^s where q is odd
        q, s = self.p - 1, 0
        while q % 2 == 0:
            q //= 2
            s += 1
        
        # 2. Find a non-residue z
        z = 2
        while self.power(z, (self.p - 1) // 2) == 1:
            z += 1
        
        # 3. Initialize variables
        m = s
        c = self.power(z, q)
        t = self.power(a, q)
        r = self.power(a, (q + 1) // 2)
        
        # 4. Main loop
        while t != 1:
            # Find the smallest i such that t^(2^i) = 1
            i, temp = 0, t
            while temp != 1:
                temp = self.multiply(temp, temp)
                i += 1
                if i >= m:
                    return None  # No square root exists
            
            # Calculate b = c^(2^(m-i-1))
            b = c
            for _ in range(m - i - 1):
                b = self.multiply(b, b)
            
            # Update variables
            m = i
            c = self.multiply(b, b)
            t = self.multiply(t, c)
            r = self.multiply(r, b)
        
        if self.multiply(r, r) == a:
            return r
        return None


@dataclass
class Point:
    """
    Represents a point on an elliptic curve
    """
    x: Optional[Union[int, float]]
    y: Optional[Union[int, float]]
    infinity: bool = False
    
    @classmethod
    def at_infinity(cls):
        """Create a point at infinity (the identity element)"""
        return cls(None, None, True)
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        if self.infinity and other.infinity:
            return True
        if self.infinity or other.infinity:
            return False
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        if self.infinity:
            return "Point at infinity"
        return f"({self.x}, {self.y})"


class EllipticCurve:
    """
    Base class for elliptic curves in Weierstrass form: y^2 = x^3 + ax + b
    """
    def __init__(self, a, b, field=None):
        """
        Initialize an elliptic curve with parameters a and b
        
        Args:
            a: Coefficient of x
            b: Constant term
            field: Field over which the curve is defined (None for real numbers)
        """
        self.a = a
        self.b = b
        self.field = field
        
        # Check that 4a^3 + 27b^2 ≠ 0 (non-singular curve)
        if field is None:
            # Over real numbers
            if 4 * a**3 + 27 * b**2 == 0:
                raise ValueError("The curve is singular")
        else:
            # Over finite field
            discriminant = field.add(
                field.multiply(4, field.power(a, 3)),
                field.multiply(27, field.power(b, 2))
            )
            if discriminant == 0:
                raise ValueError("The curve is singular over this field")
    
    def contains_point(self, point: Point) -> bool:
        """Check if a point lies on the curve"""
        if point.infinity:
            return True
        
        x, y = point.x, point.y
        
        if self.field is None:
            # Over real numbers
            return y**2 == x**3 + self.a * x + self.b
        else:
            # Over finite field
            lhs = self.field.power(y, 2)
            rhs = self.field.add(
                self.field.add(
                    self.field.power(x, 3),
                    self.field.multiply(self.a, x)
                ),
                self.b
            )
            return lhs == rhs
    
    def add_points(self, p1: Point, p2: Point) -> Point:
        """
        Add two points on the elliptic curve
        
        Args:
            p1, p2: Points on the curve
            
        Returns:
            A new point representing p1 + p2
        """
        # Check that points are on the curve
        if not self.contains_point(p1):
            raise ValueError("Point p1 is not on the curve")
        if not self.contains_point(p2):
            raise ValueError("Point p2 is not on the curve")
        
        # Handle point at infinity (identity element)
        if p1.infinity:
            return Point(p2.x, p2.y, p2.infinity)
        if p2.infinity:
            return Point(p1.x, p1.y, p1.infinity)
        
        # Real field or finite field
        if self.field is None:
            return self._add_points_real(p1, p2)
        else:
            return self._add_points_finite(p1, p2)
    
    def _add_points_real(self, p1: Point, p2: Point) -> Point:
        """Add two points on the curve over real numbers"""
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        
        # P + (-P) = O (point at infinity)
        if x1 == x2 and y1 == -y2:
            return Point.at_infinity()
        
        # Calculate slope of the line
        if x1 == x2 and y1 == y2:
            # Point doubling
            if y1 == 0:  # Tangent is vertical, result is point at infinity
                return Point.at_infinity()
            # Slope of the tangent line
            m = (3 * x1**2 + self.a) / (2 * y1)
        else:
            # Point addition
            m = (y2 - y1) / (x2 - x1)
        
        # Calculate new point coordinates
        x3 = m**2 - x1 - x2
        y3 = m * (x1 - x3) - y1
        
        return Point(x3, y3)
    
    def _add_points_finite(self, p1: Point, p2: Point) -> Point:
        """Add two points on the curve over a finite field"""
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        field = self.field
        
        # P + (-P) = O (point at infinity)
        if x1 == x2 and field.add(y1, y2) == 0:
            return Point.at_infinity()
        
        # Calculate slope of the line
        if x1 == x2 and y1 == y2:
            # Point doubling
            if y1 == 0:  # Tangent is vertical, result is point at infinity
                return Point.at_infinity()
            # Slope of the tangent line: m = (3x₁² + a) / (2y₁)
            numerator = field.add(
                field.multiply(3, field.power(x1, 2)),
                self.a
            )
            denominator = field.multiply(2, y1)
            m = field.divide(numerator, denominator)
        else:
            # Point addition: m = (y₂ - y₁) / (x₂ - x₁)
            numerator = field.subtract(y2, y1)
            denominator = field.subtract(x2, x1)
            m = field.divide(numerator, denominator)
        
        # Calculate new point coordinates
        # x₃ = m² - x₁ - x₂
        x3 = field.subtract(
            field.subtract(field.power(m, 2), x1),
            x2
        )
        # y₃ = m(x₁ - x₃) - y₁
        y3 = field.subtract(
            field.multiply(m, field.subtract(x1, x3)),
            y1
        )
        
        return Point(x3, y3)
    
    def scalar_multiply(self, k: int, point: Point) -> Point:
        """
        Multiply a point by a scalar using the double-and-add algorithm
        
        Args:
            k: Scalar multiplier
            point: Point on the curve
            
        Returns:
            A new point representing k * P
        """
        if k < 0:
            # Handle negative scalars: (-k)P = -(kP)
            k = -k
            point = self.negate(point)
        
        if k == 0 or point.infinity:
            return Point.at_infinity()
        
        # Double-and-add algorithm
        result = Point.at_infinity()
        addend = Point(point.x, point.y, point.infinity)
        
        while k > 0:
            if k & 1:  # if k is odd
                result = self.add_points(result, addend)
            addend = self.add_points(addend, addend)  # double
            k >>= 1  # divide by 2
        
        return result
    
    def negate(self, point: Point) -> Point:
        """Return the additive inverse of a point on the curve"""
        if point.infinity:
            return Point.at_infinity()
        
        if self.field is None:
            # Over real numbers
            return Point(point.x, -point.y)
        else:
            # Over finite field
            return Point(point.x, self.field.subtract(0, point.y))


class EllipticCurveOverReals(EllipticCurve):
    """
    Elliptic curve over real numbers with additional visualization methods
    """
    def __init__(self, a, b):
        """Initialize curve with y² = x³ + ax + b over real numbers"""
        super().__init__(a, b)
    
    def plot(self, x_range=(-5, 5), num_points=1000):
        """
        Plot the elliptic curve
        
        Args:
            x_range: Range of x values to plot
            num_points: Number of points to compute for the plot
        """
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_squared = x_vals**3 + self.a * x_vals + self.b
        
        # Filter points where y² < 0
        valid_indices = y_squared >= 0
        x_valid = x_vals[valid_indices]
        y_valid = np.sqrt(y_squared[valid_indices])
        
        plt.figure(figsize=(10, 8))
        
        # Plot points for positive y values
        plt.plot(x_valid, y_valid, 'b-', label=f'y² = x³ + {self.a}x + {self.b}')
        
        # Plot points for negative y values
        plt.plot(x_valid, -y_valid, 'b-')
        
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.title("Elliptic Curve")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def illustrate_addition(self, p1: Point, p2: Point):
        """
        Visualize the geometric interpretation of point addition
        
        Args:
            p1, p2: Points on the curve to add
        """
        if not self.contains_point(p1) or not self.contains_point(p2):
            raise ValueError("Both points must be on the curve")
        
        # Calculate the sum
        p3 = self.add_points(p1, p2)
        
        # Plot the curve
        x_min = min(p1.x, p2.x, p3.x if not p3.infinity else 0) - 2
        x_max = max(p1.x, p2.x, p3.x if not p3.infinity else 0) + 2
        
        x_vals = np.linspace(x_min, x_max, 1000)
        y_squared = x_vals**3 + self.a * x_vals + self.b
        
        # Filter points where y² < 0
        valid_indices = y_squared >= 0
        x_valid = x_vals[valid_indices]
        y_valid = np.sqrt(y_squared[valid_indices])
        
        plt.figure(figsize=(10, 8))
        
        # Plot the curve
        plt.plot(x_valid, y_valid, 'b-', label=f'y² = x³ + {self.a}x + {self.b}')
        plt.plot(x_valid, -y_valid, 'b-')
        
        # Plot the points
        plt.plot(p1.x, p1.y, 'ro', markersize=8, label=f'P = ({p1.x}, {p1.y})')
        plt.plot(p2.x, p2.y, 'go', markersize=8, label=f'Q = ({p2.x}, {p2.y})')
        
        if not p3.infinity:
            plt.plot(p3.x, p3.y, 'mo', markersize=8, label=f'P + Q = ({p3.x}, {p3.y})')
            
            # Draw the line through P and Q (or tangent if P = Q)
            if p1.x == p2.x and p1.y == p2.y:
                # Tangent line at P
                if p1.y != 0:  # Non-vertical tangent
                    m = (3 * p1.x**2 + self.a) / (2 * p1.y)
                    b_line = p1.y - m * p1.x
                    y_line = m * x_vals + b_line
                    plt.plot(x_vals, y_line, 'r--', alpha=0.7, label='Tangent at P')
            else:
                # Line through P and Q
                if p1.x != p2.x:  # Non-vertical line
                    m = (p2.y - p1.y) / (p2.x - p1.x)
                    b_line = p1.y - m * p1.x
                    y_line = m * x_vals + b_line
                    plt.plot(x_vals, y_line, 'r--', alpha=0.7, label='Line through P and Q')
                else:  # Vertical line
                    plt.axvline(x=p1.x, color='r', linestyle='--', alpha=0.7, label='Vertical line')
        else:
            plt.title("P + Q = O (Point at infinity)")
        
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.title("Elliptic Curve Point Addition")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


class EllipticCurveOverFiniteField(EllipticCurve):
    """
    Elliptic curve over a finite field GF(p)
    """
    def __init__(self, a: int, b: int, p: int):
        """
        Initialize a curve y² = x³ + ax + b over GF(p)
        
        Args:
            a, b: Curve parameters
            p: Prime modulus defining the field
        """
        field = FiniteField(p)
        super().__init__(a, b, field)
        self.p = p
    
    def find_points(self) -> List[Point]:
        """
        Find all points on the curve over the finite field
        
        Returns:
            List of all points on the curve including the point at infinity
        """
        points = [Point.at_infinity()]
        
        for x in range(self.p):
            # Calculate y² = x³ + ax + b
            x_cubed = self.field.power(x, 3)
            ax = self.field.multiply(self.a, x)
            y_squared = self.field.add(
                self.field.add(x_cubed, ax),
                self.b
            )
            
            # Try to find square root of y²
            y = self.field.sqrt(y_squared)
            
            if y is not None:
                # If y exists, add (x, y) and (x, -y)
                points.append(Point(x, y))
                if y != 0:  # If y ≠ 0, add the point with -y
                    points.append(Point(x, self.field.subtract(0, y)))
        
        return points
    
    def count_points(self) -> int:
        """
        Count the number of points on the curve
        
        Returns:
            The order of the curve (number of points)
        """
        return len(self.find_points())
    
    def find_generator(self) -> Optional[Point]:
        """
        Find a generator point of large order
        
        Returns:
            A point that generates a large subgroup, or None if not found
        """
        points = self.find_points()
        
        # Sort points by descending order (to find high-order points first)
        orders = []
        for point in points[1:]:  # Skip point at infinity
            order = self.find_point_order(point)
            orders.append((point, order))
        
        # Sort by order (descending)
        orders.sort(key=lambda x: x[1], reverse=True)
        
        if orders:
            return orders[0][0]  # Return point with highest order
        
        return None
    
    def find_point_order(self, point: Point) -> int:
        """
        Find the order of a point (smallest k such that k*P = O)
        
        Args:
            point: Point on the curve
            
        Returns:
            Order of the point
        """
        if point.infinity:
            return 1
        
        # Compute multiples of the point until we reach the identity
        k = 1
        multiple = Point(point.x, point.y)
        
        # Limit the search to curve order (which is at most p+1+2√p by Hasse's theorem)
        max_order = self.p + 1 + 2 * int(math.sqrt(self.p))
        
        while k <= max_order:
            multiple = self.add_points(multiple, point)
            k += 1
            if multiple.infinity:
                return k
        
        return 0  # Should not reach here unless there's a computational error
    
    def plot(self):
        """
        Visualize the points on the elliptic curve over the finite field
        """
        points = self.find_points()
        
        # Extract x and y coordinates, excluding point at infinity
        x_coords = [p.x for p in points if not p.infinity]
        y_coords = [p.y for p in points if not p.infinity]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords, y_coords, c='blue', s=25)
        
        plt.grid(True)
        plt.title(f"Elliptic Curve y² = x³ + {self.a}x + {self.b} over GF({self.p})")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # Set axis limits
        plt.xlim(-1, self.p)
        plt.ylim(-1, self.p)
        
        # Add annotations for small fields
        if self.p <= 20:
            for point in points:
                if not point.infinity:
                    plt.annotate(f"({point.x}, {point.y})", 
                                 (point.x, point.y),
                                 textcoords="offset points",
                                 xytext=(5, 5),
                                 ha='left')
        
        plt.tight_layout()
        plt.show()


class ECDSAKey:
    """
    Implements ECDSA key generation and operations
    """
    def __init__(self, curve, generator=None, order=None):
        """
        Initialize an ECDSA key with a curve and generator point
        
        Args:
            curve: The elliptic curve
            generator: Generator point (will find one if not provided)
            order: Order of the generator point (will compute if not provided)
        """
        self.curve = curve
        
        # For a curve over a finite field
        if isinstance(curve, EllipticCurveOverFiniteField):
            if generator is None:
                generator = curve.find_generator()
                if generator is None:
                    raise ValueError("Could not find suitable generator point")
            
            if order is None:
                order = curve.find_point_order(generator)
        else:
            if generator is None or order is None:
                raise ValueError("For non-finite curves, must provide generator and order")
        
        self.generator = generator
        self.order = order
        self.private_key = None
        self.public_key = None
    
    def generate_keypair(self):
        """Generate a random private key and corresponding public key"""
        # Generate random private key in range [1, order-1]
        self.private_key = random.randint(1, self.order - 1)
        
        # Compute public key as Q = dG
        self.public_key = self.curve.scalar_multiply(self.private_key, self.generator)
        
        return self.private_key, self.public_key
    
    def set_private_key(self, private_key: int):
        """
        Set the private key and compute the corresponding public key
        
        Args:
            private_key: Integer value for the private key
        """
        if private_key <= 0 or private_key >= self.order:
            raise ValueError(f"Private key must be in range [1, {self.order - 1}]")
        
        self.private_key = private_key
        self.public_key = self.curve.scalar_multiply(private_key, self.generator)
    
    def sign(self, message: bytes) -> Tuple[int, int]:
        """
        Sign a message using ECDSA
        
        Args:
            message: The message to sign as bytes
            
        Returns:
            Tuple (r, s) representing the signature
        """
        if self.private_key is None:
            raise ValueError("Private key not set. Generate or set a key pair first.")
        
        # 1. Calculate the hash of the message
        hash_obj = hashlib.sha256(message)
        z = int.from_bytes(hash_obj.digest(), byteorder='big') % self.order
        
        # 2. Select a random nonce k
        while True:
            k = random.randint(1, self.order - 1)
            
            # 3. Calculate the curve point (x, y) = k * G
            point = self.curve.scalar_multiply(k, self.generator)
            
            if point.infinity:
                continue  # Try again with a different k
            
            # 4. Calculate r = x mod n
            r = point.x % self.order
            if r == 0:
                continue  # Try again with a different k
            
            # 5. Calculate s = k^(-1) * (z + r * private_key) mod n
            if isinstance(self.curve, EllipticCurveOverFiniteField):
                field = FiniteField(self.order)  # Create a field with modulus = order
                k_inv = field.inv(k)
                s = field.multiply(k_inv, field.add(z, field.multiply(r, self.private_key)))
            else:
                # Use extended GCD for modular inverse
                def extended_gcd(a, b):
                    if a == 0:
                        return b, 0, 1
                    else:
                        gcd, x, y = extended_gcd(b % a, a)
                        return gcd, y - (b // a) * x, x
                
                _, k_inv, _ = extended_gcd(k, self.order)
                k_inv = k_inv % self.order
                s = (k_inv * (z + r * self.private_key)) % self.order
            
            if s == 0:
                continue  # Try again with a different k
            
            return r, s
    
    def verify(self, message: bytes, signature: Tuple[int, int]) -> bool:
        """
        Verify an ECDSA signature
        
        Args:
            message: The message that was signed
            signature: Tuple (r, s) representing the signature
            
        Returns:
            True if the signature is valid, False otherwise
        """
        if self.public_key is None:
            raise ValueError("Public key not set")
        
        r, s = signature
        
        # Check that r and s are in [1, n-1]
        if r <= 0 or r >= self.order or s <= 0 or s >= self.order:
            return False
        
        # 1. Calculate the hash of the message
        hash_obj = hashlib.sha256(message)
        z = int.from_bytes(hash_obj.digest(), byteorder='big') % self.order
        
        # 2. Calculate u1 and u2
        if isinstance(self.curve, EllipticCurveOverFiniteField):
            field = FiniteField(self.order)
            s_inv = field.inv(s)
            u1 = field.multiply(z, s_inv)
            u2 = field.multiply(r, s_inv)
        else:
            # Use extended GCD for modular inverse
            def extended_gcd(a, b):
                if a == 0:
                    return b, 0, 1
                else:
                    gcd, x, y = extended_gcd(b % a, a)
                    return gcd, y - (b // a) * x, x
            
            _, s_inv, _ = extended_gcd(s, self.order)
            s_inv = s_inv % self.order
            u1 = (z * s_inv) % self.order
            u2 = (r * s_inv) % self.order
        
        # 3. Calculate the curve point P = u1*G + u2*Q
        point1 = self.curve.scalar_multiply(u1, self.generator)
        point2 = self.curve.scalar_multiply(u2, self.public_key)
        P = self.curve.add_points(point1, point2)
        
        # 4. Verify that P is not at infinity and r = P.x mod n
        if P.infinity:
            return False
        
        return r == P.x % self.order


def ecdh_key_exchange_demo(curve, generator, order):
    """
    Demonstrate Elliptic Curve Diffie-Hellman key exchange
    
    Args:
        curve: Elliptic curve to use
        generator: Generator point
        order: Order of the generator
    
    Returns:
        True if key exchange succeeded, False otherwise
    """
    # 1. Alice generates her key pair
    alice = ECDSAKey(curve, generator, order)
    alice_private, alice_public = alice.generate_keypair()
    
    # 2. Bob generates his key pair
    bob = ECDSAKey(curve, generator, order)
    bob_private, bob_public = bob.generate_keypair()
    
    # 3. Alice computes the shared secret
    alice_shared = curve.scalar_multiply(alice_private, bob_public)
    
    # 4. Bob computes the shared secret
    bob_shared = curve.scalar_multiply(bob_private, alice_public)
    
    # The shared secrets should be equal
    return alice_shared == bob_shared


def demonstrate_curves():
    """Demonstrate elliptic curve operations and applications"""
    
    print("=== ELLIPTIC CURVES DEMONSTRATION ===\n")
    
    # 1. Elliptic Curve over Real Numbers
    print("1. ELLIPTIC CURVE OVER REAL NUMBERS")
    print("-----------------------------------")
    curve_real = EllipticCurveOverReals(a=-3, b=2)
    print(f"Curve equation: y² = x³ + ({curve_real.a})x + {curve_real.b}")
    
    # Define some points
    p1 = Point(1, 0)
    p2 = Point(-1, 0)
    
    if curve_real.contains_point(p1):
        print(f"\nPoint P = {p1} is on the curve")
    else:
        print(f"\nPoint P = {p1} is not on the curve")
    
    if curve_real.contains_point(p2):
        print(f"Point Q = {p2} is on the curve")
    else:
        print(f"Point Q = {p2} is not on the curve")
    
    # 2. Elliptic Curve over Finite Field
    print("\n2. ELLIPTIC CURVE OVER FINITE FIELD")
    print("----------------------------------")
    # Using a small prime for demonstration
    p = 17
    curve_finite = EllipticCurveOverFiniteField(a=2, b=3, p=p)
    print(f"Curve equation: y² = x³ + {curve_finite.a}x + {curve_finite.b} over GF({p})")
    
    # Find all points on the curve
    points = curve_finite.find_points()
    point_count = len(points)
    print(f"\nNumber of points on the curve: {point_count}")
    
    if p <= 17:  # Only print all points for small fields
        print("\nPoints on the curve:")
        for i, point in enumerate(points):
            print(f"{i+1}. {point}")
    
    # Find a generator point
    generator = curve_finite.find_generator()
    if generator:
        generator_order = curve_finite.find_point_order(generator)
        print(f"\nGenerator point: {generator}")
        print(f"Order of the generator: {generator_order}")
        
        # Generate multiples of the generator
        print("\nMultiples of the generator:")
        for i in range(1, min(5, generator_order) + 1):
            multiple = curve_finite.scalar_multiply(i, generator)
            print(f"{i}G = {multiple}")
        if generator_order > 5:
            print("...")
        print(f"{generator_order}G = {Point.at_infinity()}")
    
    # 3. ECDSA Signature
    print("\n3. ECDSA SIGNATURE")
    print("----------------")
    
    # Use a standard curve for ECDSA (simplified P-192 for demonstration)
    p_192 = 6277101735386680763835789423207666416083908700390324961279
    a_192 = -3
    b_192 = 2455155546008943817740293915197451784769108058161191238065
    
    # For demo purposes, we'll use a much smaller curve
    ecdsa_curve = EllipticCurveOverFiniteField(a=2, b=3, p=17)
    
    # Find a generator and its order
    ecdsa_generator = ecdsa_curve.find_generator()
    ecdsa_order = ecdsa_curve.find_point_order(ecdsa_generator)
    
    print(f"Using curve: y² = x³ + {ecdsa_curve.a}x + {ecdsa_curve.b} over GF({ecdsa_curve.p})")
    print(f"Generator: {ecdsa_generator}")
    print(f"Generator order: {ecdsa_order}")
    
    # Create ECDSA key
    ecdsa_key = ECDSAKey(ecdsa_curve, ecdsa_generator, ecdsa_order)
    private_key, public_key = ecdsa_key.generate_keypair()
    
    print(f"\nPrivate key: {private_key}")
    print(f"Public key: {public_key}")
    
    # Sign a message
    message = b"Hello, Elliptic Curve Cryptography!"
    signature = ecdsa_key.sign(message)
    
    print(f"\nMessage: {message.decode()}")
    print(f"Signature (r, s): {signature}")
    
    # Verify the signature
    verification = ecdsa_key.verify(message, signature)
    print(f"Signature verification: {'Success' if verification else 'Failed'}")
    
    # 4. ECDH Key Exchange
    print("\n4. ELLIPTIC CURVE DIFFIE-HELLMAN (ECDH)")
    print("---------------------------------------")
    
    ecdh_success = ecdh_key_exchange_demo(ecdsa_curve, ecdsa_generator, ecdsa_order)
    print(f"ECDH key exchange {'succeeded' if ecdh_success else 'failed'}")
    
    # 5. Standard Elliptic Curves for Cryptography
    print("\n5. STANDARD CURVES FOR CRYPTOGRAPHY")
    print("----------------------------------")
    print("NIST P-256 (secp256r1) parameters:")
    print("p = 115792089210356248762697446949407573530086143415290314195533631308867097853951")
    print("a = -3")
    print("b = 41058363725152142129326129780047268409114441015993725554835345729293459290880")
    
    print("\nCurve25519 parameters:")
    print("p = 2^255 - 19")
    print("Montgomery form: y^2 = x^3 + 486662x^2 + x (mod p)")


if __name__ == "__main__":
    demonstrate_curves()
    
    # Uncomment to visualize a curve over real numbers
    # curve_real = EllipticCurveOverReals(a=-3, b=2)
    # curve_real.plot()
    
    # Uncomment to visualize a point addition
    # p1 = Point(1, 0)
    # p2 = Point(0, np.sqrt(2))  # Point (0, sqrt(2)) on the curve y^2 = x^3 - 3x + 2
    # curve_real.illustrate_addition(p1, p2)
    
    # Uncomment to visualize a curve over a finite field
    # curve_finite = EllipticCurveOverFiniteField(a=2, b=3, p=17)
    # curve_finite.plot()