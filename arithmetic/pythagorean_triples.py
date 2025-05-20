import math
from typing import List, Tuple, Generator

def generate_pythagorean_triples(limit: int) -> List[Tuple[int, int, int]]:
    """
    Generate Pythagorean triples (a, b, c) where a² + b² = c² and max(a,b,c) <= limit.
    Uses Euclid's formula to generate primitive triples, then adds their multiples.
    
    Args:
        limit: Upper bound for the values in the triples
        
    Returns:
        List of tuples (a, b, c) representing Pythagorean triples
    """
    triples = []
    
    # Generate primitive Pythagorean triples using Euclid's formula
    # For every m > n > 0 where m and n are coprime and not both odd:
    # a = m² - n², b = 2mn, c = m² + n²
    m = 2
    while m * m <= limit:  # m² must be <= limit
        for n in range(1, m):
            # Ensure m and n are coprime and not both odd
            if (m % 2 == 0 or n % 2 == 0) and math.gcd(m, n) == 1:
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                
                # Ensure the triple doesn't exceed our limit
                if c <= limit:
                    # Add the primitive triple
                    triples.append((a, b, c))
                    
                    # Add multiples of the primitive triple
                    k = 2
                    while k * c <= limit:
                        triples.append((k * a, k * b, k * c))
                        k += 1
        m += 1
    
    # Sort by c value (hypotenuse)
    return sorted(triples, key=lambda t: t[2])


def triangular_numbers(n: int) -> List[int]:
    """
    Generate the first n triangular numbers.
    Triangular numbers are of the form T_i = i(i+1)/2
    
    Args:
        n: Number of triangular numbers to generate
        
    Returns:
        List of the first n triangular numbers
    """
    return [i * (i + 1) // 2 for i in range(1, n + 1)]


def pentagonal_numbers(n: int) -> List[int]:
    """
    Generate the first n pentagonal numbers.
    Pentagonal numbers are of the form P_i = i(3i-1)/2
    
    Args:
        n: Number of pentagonal numbers to generate
        
    Returns:
        List of the first n pentagonal numbers
    """
    return [i * (3 * i - 1) // 2 for i in range(1, n + 1)]


def is_triangular(num: int) -> bool:
    """
    Check if a number is triangular.
    A number is triangular if 8*num + 1 is a perfect square.
    
    Args:
        num: Number to check
        
    Returns:
        True if the number is triangular, False otherwise
    """
    # Solve n(n+1)/2 = num for n
    # This gives n² + n - 2*num = 0
    # Using quadratic formula and checking if n is a positive integer
    discriminant = 1 + 8 * num
    sqrt_discriminant = int(math.sqrt(discriminant))
    return sqrt_discriminant * sqrt_discriminant == discriminant and (sqrt_discriminant - 1) % 2 == 0


def is_pentagonal(num: int) -> bool:
    """
    Check if a number is pentagonal.
    A number is pentagonal if (1+sqrt(1+24*num))/6 is an integer.
    
    Args:
        num: Number to check
        
    Returns:
        True if the number is pentagonal, False otherwise
    """
    # Solve n(3n-1)/2 = num for n
    # This gives 3n² - n - 2*num = 0
    # Using quadratic formula and checking if n is a positive integer
    discriminant = 1 + 24 * num
    sqrt_discriminant = int(math.sqrt(discriminant))
    return sqrt_discriminant * sqrt_discriminant == discriminant and (sqrt_discriminant + 1) % 6 == 0


def find_special_numbers(limit: int) -> None:
    """
    Find numbers that are both triangular and pentagonal under a given limit.
    
    Args:
        limit: Upper bound for the search
    """
    print(f"Numbers under {limit} that are both triangular and pentagonal:")
    for i in range(1, limit + 1):
        if is_triangular(i) and is_pentagonal(i):
            print(i)


# Example usage
if __name__ == "__main__":
    # Generate and print the first 10 Pythagorean triples
    triples = generate_pythagorean_triples(100)
    print("First 10 Pythagorean triples:")
    for i, triple in enumerate(triples[:10]):
        print(f"{i+1}. {triple} ({triple[0]}² + {triple[1]}² = {triple[2]}²)")
    
    # Generate and print the first 10 triangular numbers
    tri_nums = triangular_numbers(10)
    print("\nFirst 10 triangular numbers:")
    for i, num in enumerate(tri_nums):
        print(f"T_{i+1} = {num}")
    
    # Generate and print the first 10 pentagonal numbers
    pent_nums = pentagonal_numbers(10)
    print("\nFirst 10 pentagonal numbers:")
    for i, num in enumerate(pent_nums):
        print(f"P_{i+1} = {num}")
    
    # Find numbers that are both triangular and pentagonal
    find_special_numbers(100000)