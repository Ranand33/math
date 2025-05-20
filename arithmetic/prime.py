import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Generator
from collections import Counter
from scipy.special import zeta


class PrimeDistribution:
    """A class to analyze and visualize the distribution of prime numbers"""
    
    def __init__(self, limit: int = 1000000):
        """
        Initialize with a limit for prime analysis
        
        Args:
            limit: Upper bound for prime number generation
        """
        self.limit = limit
        self.primes = []
        self.prime_count = {}  # Cache for prime counting function
        
    def generate_primes(self, n: int = None) -> List[int]:
        """
        Generate all primes up to n using the Sieve of Eratosthenes
        
        Args:
            n: Upper limit (defaults to self.limit if None)
            
        Returns:
            List of prime numbers
        """
        if n is None:
            n = self.limit
            
        # Initialize the sieve
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0:2] = False  # 0 and 1 are not prime
        
        # Main sieve loop
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                # Mark all multiples of i as composite
                sieve[i*i:n+1:i] = False
        
        # Extract primes
        self.primes = np.where(sieve)[0].tolist()
        return self.primes
    
    def count_primes(self, x: int) -> int:
        """
        Compute π(x), the number of primes ≤ x
        
        Args:
            x: Upper limit
            
        Returns:
            Count of primes ≤ x
        """
        if x in self.prime_count:
            return self.prime_count[x]
            
        if not self.primes or self.primes[-1] < x:
            self.generate_primes(max(x, self.limit))
            
        count = sum(1 for p in self.primes if p <= x)
        self.prime_count[x] = count
        return count
    
    def prime_number_theorem_estimate(self, x: int) -> float:
        """
        Estimate π(x) using the Prime Number Theorem: π(x) ~ x/ln(x)
        
        Args:
            x: The value to estimate for
            
        Returns:
            Estimated count of primes ≤ x
        """
        if x < 2:
            return 0
        return x / math.log(x)
    
    def li_estimate(self, x: int) -> float:
        """
        Estimate π(x) using the logarithmic integral Li(x)
        A more accurate approximation than x/ln(x)
        
        Args:
            x: The value to estimate for
            
        Returns:
            Estimated count of primes using Li(x)
        """
        if x < 2:
            return 0
            
        # Numerical approximation of Li(x)
        result = 0
        dt = 0.01
        t = 2  # Start at t=2 to avoid division by zero
        
        while t <= x:
            result += dt / math.log(t)
            t += dt
            
        return result
    
    def riemann_r_estimate(self, x: int, terms: int = 10) -> float:
        """
        Estimate π(x) using Riemann's R function, which gives 
        a more accurate approximation than Li(x)
        
        Args:
            x: The value to estimate for
            terms: Number of terms to use in the approximation
            
        Returns:
            Estimated count of primes using Riemann's R function
        """
        if x < 2:
            return 0
            
        result = 0
        for n in range(1, terms + 1):
            result += (math.pow(math.log(x), n) / math.factorial(n) / n)
            
        return result * x
    
    def prime_gaps(self) -> List[int]:
        """
        Calculate gaps between consecutive primes
        
        Returns:
            List of gaps between consecutive primes
        """
        if not self.primes:
            self.generate_primes()
            
        return [self.primes[i+1] - self.primes[i] for i in range(len(self.primes)-1)]
    
    def analyze_prime_gaps(self) -> Dict:
        """
        Analyze the distribution of gaps between consecutive primes
        
        Returns:
            Dictionary with gap statistics
        """
        gaps = self.prime_gaps()
        
        # Count occurrences of each gap size
        gap_counts = Counter(gaps)
        
        # Calculate statistics
        stats = {
            'min_gap': min(gaps),
            'max_gap': max(gaps),
            'avg_gap': sum(gaps) / len(gaps),
            'gap_frequencies': dict(sorted(gap_counts.items())),
            'most_common_gap': gap_counts.most_common(1)[0][0]
        }
        
        return stats
    
    def twin_primes(self) -> List[Tuple[int, int]]:
        """
        Find all twin primes (pairs of primes that differ by 2)
        
        Returns:
            List of twin prime pairs
        """
        if not self.primes:
            self.generate_primes()
            
        return [(self.primes[i], self.primes[i+1]) 
                for i in range(len(self.primes)-1) 
                if self.primes[i+1] - self.primes[i] == 2]
    
    def prime_races(self, modulus: int = 4, remainder1: int = 1, remainder2: int = 3) -> Dict:
        """
        Analyze "prime races" - the competition between counts of primes in different residue classes
        
        Args:
            modulus: The modulus for the congruence classes
            remainder1, remainder2: The two remainder classes to compare
            
        Returns:
            Dictionary with race statistics
        """
        if not self.primes:
            self.generate_primes()
            
        counts1 = []
        counts2 = []
        
        count1 = 0
        count2 = 0
        
        for p in self.primes:
            if p % modulus == remainder1:
                count1 += 1
            elif p % modulus == remainder2:
                count2 += 1
                
            counts1.append(count1)
            counts2.append(count2)
            
        # Determine lead switches (Chebyshev bias)
        lead_switches = 0
        leading = None
        
        for i in range(len(counts1)):
            if counts1[i] > counts2[i] and (leading is None or leading == 2):
                lead_switches += (1 if leading == 2 else 0)
                leading = 1
            elif counts2[i] > counts1[i] and (leading is None or leading == 1):
                lead_switches += (1 if leading == 1 else 0)
                leading = 2
                
        return {
            'final_count1': counts1[-1],
            'final_count2': counts2[-1],
            'lead_switches': lead_switches,
            'counts1': counts1,
            'counts2': counts2,
            'final_diff': counts1[-1] - counts2[-1]
        }
    
    def plot_prime_distribution(self, max_x: int = None, step: int = 100):
        """
        Plot the actual and estimated distribution of primes
        
        Args:
            max_x: Maximum x value for the plot
            step: Step size for sampling points
        """
        if max_x is None:
            max_x = min(100000, self.limit)
            
        x_values = list(range(10, max_x, step))
        actual_counts = [self.count_primes(x) for x in x_values]
        pnt_estimates = [self.prime_number_theorem_estimate(x) for x in x_values]
        li_estimates = [self.li_estimate(x) for x in x_values]
        
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, actual_counts, 'b-', label='π(x) (Actual)')
        plt.plot(x_values, pnt_estimates, 'r--', label='x/ln(x) (PNT)')
        plt.plot(x_values, li_estimates, 'g:', label='Li(x)')
        
        plt.xlabel('x')
        plt.ylabel('Number of primes ≤ x')
        plt.title('Prime Number Distribution')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot the relative error
        plt.figure(figsize=(12, 6))
        pnt_errors = [(pnt_estimates[i] - actual_counts[i]) / actual_counts[i] 
                     for i in range(len(x_values))]
        li_errors = [(li_estimates[i] - actual_counts[i]) / actual_counts[i] 
                    for i in range(len(x_values))]
        
        plt.plot(x_values, pnt_errors, 'r--', label='x/ln(x) Error')
        plt.plot(x_values, li_errors, 'g:', label='Li(x) Error')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.xlabel('x')
        plt.ylabel('Relative Error')
        plt.title('Relative Error in Prime Number Approximations')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_prime_gaps(self):
        """Plot the distribution of gaps between consecutive primes"""
        if not self.primes:
            self.generate_primes()
            
        gaps = self.prime_gaps()
        
        plt.figure(figsize=(12, 6))
        
        # Plot gap sizes vs. index
        plt.subplot(1, 2, 1)
        plt.plot(range(len(gaps)), gaps, 'b.', alpha=0.5)
        plt.xlabel('Index')
        plt.ylabel('Gap Size')
        plt.title('Prime Gaps')
        plt.grid(True)
        
        # Plot gap frequency histogram
        plt.subplot(1, 2, 2)
        gap_counts = Counter(gaps)
        max_gap = max(gaps)
        plt.bar(list(range(2, max_gap + 2, 2)), 
                [gap_counts.get(i, 0) for i in range(2, max_gap + 2, 2)],
                width=1.5)
        plt.xlabel('Gap Size')
        plt.ylabel('Frequency')
        plt.title('Prime Gap Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_prime_races(self, modulus: int = 4, remainder1: int = 1, remainder2: int = 3):
        """
        Plot prime races between residue classes
        
        Args:
            modulus: The modulus for the congruence classes
            remainder1, remainder2: The two remainder classes to compare
        """
        race_data = self.prime_races(modulus, remainder1, remainder2)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.primes, race_data['counts1'], 'b-', 
                 label=f'Primes ≡ {remainder1} mod {modulus}')
        plt.plot(self.primes, race_data['counts2'], 'r-', 
                 label=f'Primes ≡ {remainder2} mod {modulus}')
        
        plt.xlabel('x')
        plt.ylabel('Count')
        plt.title(f'Prime Race: {remainder1} vs {remainder2} (mod {modulus})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot difference
        diff = [race_data['counts1'][i] - race_data['counts2'][i] 
                for i in range(len(self.primes))]
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.primes, diff, 'g-')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.xlabel('x')
        plt.ylabel(f'π(x,{modulus},{remainder1}) - π(x,{modulus},{remainder2})')
        plt.title(f'Difference in Prime Counts: {remainder1} vs {remainder2} (mod {modulus})')
        plt.grid(True)
        plt.show()
    
    def mertens_function(self, n: int) -> int:
        """
        Calculate the Mertens function M(n) = sum of μ(k) for k from 1 to n
        where μ is the Möbius function
        
        Args:
            n: Upper limit
            
        Returns:
            Value of the Mertens function at n
        """
        # Calculate Möbius function values
        mobius = np.ones(n + 1, dtype=int)
        
        # 0 is not in the domain of the Möbius function
        if n >= 0:
            mobius[0] = 0
            
        # 1 is defined to be 1
        if n >= 1:
            mobius[1] = 1
            
        # Sieve to calculate Möbius values
        for i in range(2, int(math.sqrt(n)) + 1):
            if mobius[i] == 1:
                # i is prime, mark all multiples of i²
                for j in range(i, n + 1, i):
                    mobius[j] *= -1  # Multiply by -1 for each prime factor
                
                # Mark all multiples of i² with 0 (not square-free)
                for j in range(i**2, n + 1, i**2):
                    mobius[j] = 0
        
        # Calculate Mertens function (cumulative sum)
        mertens = np.cumsum(mobius)
        return mertens[n]
    
    def plot_mertens_function(self, limit: int = 1000):
        """
        Plot the Mertens function M(n) which gives insight into the 
        distribution of primes through the Möbius function
        
        Args:
            limit: Upper limit for the plot
        """
        x_values = list(range(1, limit + 1))
        m_values = [self.mertens_function(x) for x in x_values]
        
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, m_values, 'b-')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.xlabel('n')
        plt.ylabel('M(n)')
        plt.title('Mertens Function')
        plt.grid(True)
        plt.show()
        
        # Plot M(n)/sqrt(n), which is conjectured to be bounded
        scaled_values = [m_values[i] / math.sqrt(x_values[i]) for i in range(len(x_values))]
        
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, scaled_values, 'r-')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.xlabel('n')
        plt.ylabel('M(n)/√n')
        plt.title('Scaled Mertens Function (M(n)/√n)')
        plt.grid(True)
        plt.show()
    
    def primes_in_intervals(self, interval_size: int = 1000) -> List[int]:
        """
        Count primes in consecutive intervals of fixed size
        
        Args:
            interval_size: Size of each interval
            
        Returns:
            List of prime counts in each interval
        """
        if not self.primes:
            self.generate_primes()
            
        max_value = self.primes[-1]
        intervals = (max_value // interval_size) + 1
        
        counts = [0] * intervals
        
        for p in self.primes:
            interval_idx = p // interval_size
            counts[interval_idx] += 1
            
        return counts
    
    def plot_interval_distribution(self, interval_size: int = 1000):
        """
        Plot the distribution of primes in consecutive intervals
        
        Args:
            interval_size: Size of each interval
        """
        counts = self.primes_in_intervals(interval_size)
        
        plt.figure(figsize=(12, 6))
        x_values = [(i + 0.5) * interval_size for i in range(len(counts))]
        
        plt.bar(x_values, counts, width=interval_size * 0.8)
        
        # Add theoretical curve based on Prime Number Theorem
        theoretical = [interval_size / math.log(max(x, 2)) for x in x_values]
        plt.plot(x_values, theoretical, 'r-', label='PNT Estimate')
        
        plt.xlabel('Interval Midpoint')
        plt.ylabel(f'Number of Primes in Interval of Size {interval_size}')
        plt.title('Distribution of Primes in Consecutive Intervals')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def legendre_sum(self, x: int, a: int, N: int) -> int:
        """
        Calculate Legendre's sum S(x,a,N) = number of integers n ≤ x
        such that n has exactly a prime factors ≤ N (with multiplicity)
        
        Args:
            x: Upper bound for integers
            a: Exact number of prime factors
            N: Consider only prime factors ≤ N
            
        Returns:
            Value of Legendre's sum S(x,a,N)
        """
        if not self.primes:
            self.generate_primes()
            
        # Get primes ≤ N
        primes_n = [p for p in self.primes if p <= N]
        
        # Base case
        if a == 0:
            return 1 if x >= 1 else 0
            
        if not primes_n or a < 0 or x < 1:
            return 0
            
        # Recursive implementation of Legendre's formula
        p = primes_n[-1]  # Last prime ≤ N
        remaining_primes = primes_n[:-1]  # All primes < p
        
        # S(x,a,N) = S(x,a,p-1) + S(x/p,a-1,p)
        result = self.legendre_sum(x, a, p-1)
        
        # Add contribution from numbers with p as a factor
        if p <= x:
            result += self.legendre_sum(x // p, a - 1, p)
            
        return result


def demonstrate_prime_distribution():
    """Demonstrate various aspects of prime number distribution"""
    
    # Create PrimeDistribution object
    pd = PrimeDistribution(limit=100000)
    
    print("=== PRIME NUMBER DISTRIBUTION ANALYSIS ===\n")
    
    # Generate primes
    primes = pd.generate_primes(1000)
    print(f"First 10 prime numbers: {primes[:10]}")
    print(f"Number of primes ≤ 1000: {len(primes)}")
    
    # Prime counting function
    print("\n=== PRIME COUNTING FUNCTION π(x) ===")
    for x in [10, 100, 1000, 10000, 100000]:
        actual = pd.count_primes(x)
        pnt_estimate = pd.prime_number_theorem_estimate(x)
        li_estimate = pd.li_estimate(x)
        print(f"π({x}) = {actual}")
        print(f"x/ln(x) estimate: {pnt_estimate:.2f}, error: {((pnt_estimate-actual)/actual*100):.2f}%")
        print(f"Li(x) estimate: {li_estimate:.2f}, error: {((li_estimate-actual)/actual*100):.2f}%")
    
    # Prime gaps
    print("\n=== PRIME GAPS ===")
    pd.generate_primes(10000)
    gap_stats = pd.analyze_prime_gaps()
    print(f"Min gap: {gap_stats['min_gap']}")
    print(f"Max gap: {gap_stats['max_gap']}")
    print(f"Average gap: {gap_stats['avg_gap']:.2f}")
    print(f"Most common gap: {gap_stats['most_common_gap']}")
    
    # Most frequent gaps
    most_common = sorted(gap_stats['gap_frequencies'].items(), 
                         key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 most frequent gaps: {most_common}")
    
    # Twin primes
    twin_primes = pd.twin_primes()
    print(f"\nNumber of twin prime pairs ≤ 10000: {len(twin_primes)}")
    print(f"First 5 twin prime pairs: {twin_primes[:5]}")
    
    # Prime races
    print("\n=== PRIME RACES (mod 4) ===")
    race_data = pd.prime_races(4, 1, 3)
    print(f"Primes ≡ 1 (mod 4) ≤ 10000: {race_data['final_count1']}")
    print(f"Primes ≡ 3 (mod 4) ≤ 10000: {race_data['final_count2']}")
    print(f"Difference: {race_data['final_diff']}")
    print(f"Lead switches: {race_data['lead_switches']}")
    
    # Plot distributions (uncomment to display plots)
    # pd.plot_prime_distribution(max_x=10000, step=100)
    # pd.plot_prime_gaps()
    # pd.plot_prime_races(4, 1, 3)
    # pd.plot_interval_distribution(interval_size=500)
    # pd.plot_mertens_function(limit=1000)


if __name__ == "__main__":
    demonstrate_prime_distribution()