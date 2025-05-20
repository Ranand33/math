import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Callable
import time
import cmath
from functools import lru_cache
from matplotlib.colors import LogNorm
import scipy.io.wavfile as wavfile
from scipy import signal


class FFT:
    """
    Fast Fourier Transform implementations and applications.
    
    This class provides multiple implementations of the FFT algorithm:
    - Recursive Cooley-Tukey FFT (decimation-in-time)
    - Iterative FFT using bit-reversal
    - Bluestein's algorithm for non-power-of-2 sizes
    
    Along with various applications:
    - Signal analysis and filtering
    - Polynomial multiplication
    - Convolution and cross-correlation
    - Prime-length FFT
    """
    
    def __init__(self):
        """Initialize the FFT calculator."""
        # Cache for twiddle factors to avoid recomputation
        self._twiddle_cache = {}
        
    @staticmethod
    def is_power_of_two(n: int) -> bool:
        """
        Check if n is a power of 2
        
        Args:
            n: Integer to check
            
        Returns:
            True if n is a power of 2, False otherwise
        """
        return n > 0 and (n & (n - 1)) == 0

    def fft_recursive(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the FFT of x using the recursive Cooley-Tukey algorithm.
        
        Args:
            x: Input array (will be padded to power of 2 if necessary)
            
        Returns:
            FFT of x
        """
        n = len(x)
        
        # Base case
        if n == 1:
            return x
        
        # Check if n is a power of 2, pad if not
        if not self.is_power_of_two(n):
            # Find next power of 2
            next_power = 1
            while next_power < n:
                next_power *= 2
            
            # Pad with zeros
            x_padded = np.zeros(next_power, dtype=complex)
            x_padded[:n] = x
            return self.fft_recursive(x_padded)
        
        # Split into even and odd indices (decimation in time)
        even = self.fft_recursive(x[0::2])
        odd = self.fft_recursive(x[1::2])
        
        # Combine the results
        factor = np.exp(-2j * np.pi * np.arange(n // 2) / n)
        result = np.zeros(n, dtype=complex)
        half = n // 2
        
        result[:half] = even + factor * odd
        result[half:] = even - factor * odd
        
        return result
    
    def ifft_recursive(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the inverse FFT using the recursive algorithm.
        
        Args:
            x: Input array
            
        Returns:
            Inverse FFT of x
        """
        n = len(x)
        
        # For IFFT, we conjugate the input, compute FFT, and conjugate again
        # Then divide by n
        x_conj = np.conjugate(x)
        fft_result = self.fft_recursive(x_conj)
        
        return np.conjugate(fft_result) / n
    
    def fft_iterative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the FFT using an iterative, in-place algorithm.
        This implementation uses the Cooley-Tukey algorithm with bit-reversal.
        
        Args:
            x: Input array (will be padded to power of 2 if necessary)
            
        Returns:
            FFT of x
        """
        x = np.asarray(x, dtype=complex)
        n = len(x)
        
        # Check if n is a power of 2, pad if not
        if not self.is_power_of_two(n):
            # Find next power of 2
            next_power = 1
            while next_power < n:
                next_power *= 2
            
            # Pad with zeros
            x_padded = np.zeros(next_power, dtype=complex)
            x_padded[:n] = x
            x = x_padded
            n = len(x)
        
        # Bit-reversal permutation
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j >= bit:
                j -= bit
                bit >>= 1
            j += bit
            
            if i < j:
                x[i], x[j] = x[j], x[i]
        
        # Cooley-Tukey decimation-in-time algorithm
        # Process stages (log2(n) stages for n-point FFT)
        stage = 2
        while stage <= n:
            omega_m = np.exp(-2j * np.pi / stage)
            
            # Process each group in the stage
            for k in range(0, n, stage):
                omega = 1.0
                
                # Process butterfly operations in each group
                for j in range(stage // 2):
                    idx1 = k + j
                    idx2 = k + j + stage // 2
                    
                    # Butterfly operation
                    temp = omega * x[idx2]
                    x[idx2] = x[idx1] - temp
                    x[idx1] += temp
                    
                    omega *= omega_m
            
            stage *= 2
        
        return x
    
    def ifft_iterative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the inverse FFT using the iterative algorithm.
        
        Args:
            x: Input array
            
        Returns:
            Inverse FFT of x
        """
        n = len(x)
        
        # For IFFT, we conjugate the input, compute FFT, and conjugate again
        # Then divide by n
        x_conj = np.conjugate(x)
        fft_result = self.fft_iterative(x_conj)
        
        return np.conjugate(fft_result) / n
    
    def bluestein_fft(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the FFT using Bluestein's algorithm for arbitrary input sizes.
        This algorithm converts an arbitrary-length FFT to a convolution.
        
        Args:
            x: Input array of any length
            
        Returns:
            FFT of x
        """
        n = len(x)
        
        # If n is a power of 2, use the faster radix-2 algorithm
        if self.is_power_of_two(n):
            return self.fft_iterative(x)
        
        # Choose a power of 2 that's at least 2*n-1
        m = 1
        while m < 2 * n - 1:
            m *= 2
        
        # Create the chirp sequences
        a = np.zeros(m, dtype=complex)
        b = np.zeros(m, dtype=complex)
        
        # Fill in the chirp sequences
        for k in range(n):
            phase = np.pi * (k * k) / n
            a[k] = x[k] * np.exp(-1j * phase)
            b[k] = np.exp(1j * phase)
            b[m-k-1] = b[k]  # b is symmetric
        
        # Compute the convolution using the Convolution Theorem
        c = self.fft_iterative(a)
        d = self.fft_iterative(b)
        e = c * d
        y = self.ifft_iterative(e)
        
        # Extract the result
        result = np.zeros(n, dtype=complex)
        for k in range(n):
            phase = np.pi * (k * k) / n
            result[k] = y[k] * np.exp(-1j * phase)
        
        return result
    
    def fft(self, x: np.ndarray, method: str = 'auto') -> np.ndarray:
        """
        Compute the FFT of x using the specified method.
        
        Args:
            x: Input array
            method: 'recursive', 'iterative', 'bluestein', or 'auto'
            
        Returns:
            FFT of x
        """
        x = np.asarray(x, dtype=complex)
        n = len(x)
        
        if method == 'auto':
            # Choose the appropriate method based on input size
            if self.is_power_of_two(n):
                method = 'iterative'  # Iterative is usually faster for power-of-2 sizes
            else:
                method = 'bluestein'  # Bluestein's algorithm for non-power-of-2 sizes
        
        if method == 'recursive':
            return self.fft_recursive(x)
        elif method == 'iterative':
            return self.fft_iterative(x)
        elif method == 'bluestein':
            return self.bluestein_fft(x)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def ifft(self, x: np.ndarray, method: str = 'auto') -> np.ndarray:
        """
        Compute the inverse FFT of x using the specified method.
        
        Args:
            x: Input array
            method: 'recursive', 'iterative', 'bluestein', or 'auto'
            
        Returns:
            Inverse FFT of x
        """
        x = np.asarray(x, dtype=complex)
        n = len(x)
        
        if method == 'auto':
            # Choose the appropriate method based on input size
            if self.is_power_of_two(n):
                method = 'iterative'
            else:
                method = 'bluestein'
        
        if method == 'recursive':
            return self.ifft_recursive(x)
        elif method == 'iterative':
            return self.ifft_iterative(x)
        elif method == 'bluestein':
            # For Bluestein's algorithm, we convert the IFFT to FFT
            x_conj = np.conjugate(x)
            fft_result = self.bluestein_fft(x_conj)
            return np.conjugate(fft_result) / n
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def convolve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the convolution of x and y using FFT.
        
        Args:
            x, y: Input arrays
            
        Returns:
            Convolution of x and y
        """
        n_x = len(x)
        n_y = len(y)
        n = n_x + n_y - 1
        
        # Pad to next power of 2 greater than n
        n_pow2 = 1
        while n_pow2 < n:
            n_pow2 *= 2
        
        x_padded = np.zeros(n_pow2, dtype=complex)
        x_padded[:n_x] = x
        
        y_padded = np.zeros(n_pow2, dtype=complex)
        y_padded[:n_y] = y
        
        # Use the Convolution Theorem: conv(x, y) = IFFT(FFT(x) * FFT(y))
        X = self.fft(x_padded)
        Y = self.fft(y_padded)
        
        # Element-wise multiplication
        Z = X * Y
        
        # Inverse FFT
        z = self.ifft(Z)
        
        # Return only the relevant part (first n elements)
        return z[:n].real if np.all(np.isreal(x)) and np.all(np.isreal(y)) else z[:n]
    
    def cross_correlate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the cross-correlation of x and y using FFT.
        
        Args:
            x, y: Input arrays
            
        Returns:
            Cross-correlation of x and y
        """
        # Cross-correlation is convolution with one signal reversed
        # cor(x, y)[n] = sum_k(x[k] * y[n+k])
        # This is equivalent to conv(x, y_reversed)
        y_reversed = np.array(y[::-1])
        
        return self.convolve(x, y_reversed)
    
    def polynomial_multiply(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Multiply two polynomials using FFT.
        
        Args:
            p1, p2: Coefficients of the polynomials (ascending order)
            
        Returns:
            Coefficients of the product polynomial
        """
        # Polynomial multiplication is equivalent to convolution of coefficients
        return self.convolve(p1, p2)
    
    def fft2d(self, x: np.ndarray) -> np.ndarray:
        """
        2D FFT of a matrix.
        
        Args:
            x: 2D input array
            
        Returns:
            2D FFT of x
        """
        # First apply 1D FFT to each row
        rows = np.zeros_like(x, dtype=complex)
        for i in range(x.shape[0]):
            rows[i] = self.fft(x[i])
        
        # Then apply 1D FFT to each column
        cols = np.zeros_like(rows, dtype=complex)
        for j in range(x.shape[1]):
            cols[:, j] = self.fft(rows[:, j])
        
        return cols
    
    def ifft2d(self, x: np.ndarray) -> np.ndarray:
        """
        2D inverse FFT of a matrix.
        
        Args:
            x: 2D input array
            
        Returns:
            2D inverse FFT of x
        """
        # First apply 1D IFFT to each row
        rows = np.zeros_like(x, dtype=complex)
        for i in range(x.shape[0]):
            rows[i] = self.ifft(x[i])
        
        # Then apply 1D IFFT to each column
        cols = np.zeros_like(rows, dtype=complex)
        for j in range(x.shape[1]):
            cols[:, j] = self.ifft(rows[:, j])
        
        return cols
    
    def spectrum_analyze(self, signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze the frequency spectrum of a signal.
        
        Args:
            signal: Input signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        n = len(signal)
        
        # Compute the FFT
        fft_result = self.fft(signal)
        
        # Calculate the magnitudes (absolute values)
        magnitudes = np.abs(fft_result)
        
        # Convert to power spectrum (optional)
        power_spectrum = magnitudes**2 / n
        
        # Calculate the frequencies
        frequencies = np.fft.fftfreq(n, 1/sample_rate)
        
        # Only return the positive frequencies (up to Nyquist frequency)
        positive_indices = frequencies >= 0
        
        return frequencies[positive_indices], power_spectrum[positive_indices]
    
    def filter_signal(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply a filter to a signal using FFT-based convolution.
        
        Args:
            signal: Input signal
            kernel: Filter kernel
            
        Returns:
            Filtered signal
        """
        return self.convolve(signal, kernel)
    
    def bandpass_filter(self, signal: np.ndarray, sample_rate: float, 
                        low_cutoff: float, high_cutoff: float) -> np.ndarray:
        """
        Apply a bandpass filter to a signal.
        
        Args:
            signal: Input signal
            sample_rate: Sample rate in Hz
            low_cutoff: Lower cutoff frequency in Hz
            high_cutoff: Upper cutoff frequency in Hz
            
        Returns:
            Filtered signal
        """
        n = len(signal)
        
        # Compute the FFT
        fft_result = self.fft(signal)
        
        # Create the frequency bins
        frequencies = np.fft.fftfreq(n, 1/sample_rate)
        
        # Create the filter mask
        mask = np.logical_and(np.abs(frequencies) >= low_cutoff, 
                              np.abs(frequencies) <= high_cutoff)
        
        # Apply the filter in the frequency domain
        filtered_fft = fft_result * mask
        
        # Inverse FFT to get back to the time domain
        filtered_signal = self.ifft(filtered_fft)
        
        # Return the real part (should be real for real input signals)
        return filtered_signal.real
    
    def plot_fft(self, signal: np.ndarray, sample_rate: float, title: str = "FFT"):
        """
        Plot the FFT of a signal.
        
        Args:
            signal: Input signal
            sample_rate: Sample rate in Hz
            title: Plot title
        """
        # Compute the FFT
        fft_result = self.fft(signal)
        
        # Compute magnitude and phase
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        # Plot original signal
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        time_axis = np.arange(len(signal)) / sample_rate
        plt.plot(time_axis, signal)
        plt.title("Time Domain Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Plot magnitude spectrum
        plt.subplot(3, 1, 2)
        freq_axis = np.fft.fftfreq(len(signal), 1/sample_rate)
        positive_freq_mask = freq_axis >= 0
        plt.plot(freq_axis[positive_freq_mask], magnitude[positive_freq_mask])
        plt.title("Frequency Spectrum - Magnitude")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        
        # Plot phase spectrum
        plt.subplot(3, 1, 3)
        plt.plot(freq_axis[positive_freq_mask], phase[positive_freq_mask])
        plt.title("Frequency Spectrum - Phase")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radians)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    def plot_spectrogram(self, signal: np.ndarray, sample_rate: float, 
                         window_size: int = 256, overlap: int = 128):
        """
        Plot a spectrogram of a signal using the Short-Time Fourier Transform (STFT).
        
        Args:
            signal: Input signal
            sample_rate: Sample rate in Hz
            window_size: Size of each STFT window
            overlap: Overlap between consecutive windows
        """
        # Compute STFT
        hop_length = window_size - overlap
        n_windows = (len(signal) - window_size) // hop_length + 1
        spectrogram = np.zeros((window_size, n_windows), dtype=complex)
        
        # Apply Hann window and compute FFT for each frame
        window = np.hanning(window_size)
        for i in range(n_windows):
            start = i * hop_length
            frame = signal[start:start + window_size] * window
            spectrogram[:, i] = self.fft(frame)
        
        # Convert to power spectrogram
        power_spectrogram = np.abs(spectrogram)**2
        
        # Plot spectrogram
        plt.figure(figsize=(12, 6))
        
        # Only plot positive frequencies up to Nyquist
        half_window = window_size // 2 + 1
        time_axis = np.arange(n_windows) * hop_length / sample_rate
        freq_axis = np.fft.fftfreq(window_size, 1/sample_rate)[:half_window]
        
        plt.pcolormesh(time_axis, freq_axis, 
                      10 * np.log10(power_spectrogram[:half_window, :] + 1e-10),  # dB scale
                      shading='gouraud', cmap='viridis')
        
        plt.colorbar(label='Power (dB)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.show()
    
    def benchmark(self, sizes: List[int], methods: List[str] = ['recursive', 'iterative', 'bluestein']):
        """
        Benchmark different FFT implementations.
        
        Args:
            sizes: List of input sizes to benchmark
            methods: List of methods to benchmark
        """
        results = {}
        for method in methods:
            results[method] = []
        
        # Also include numpy's FFT for comparison
        results['numpy'] = []
        
        for size in sizes:
            # Create a random complex array
            x = np.random.rand(size) + 1j * np.random.rand(size)
            
            # Time each method
            for method in methods:
                start_time = time.time()
                self.fft(x, method=method)
                end_time = time.time()
                results[method].append(end_time - start_time)
            
            # Time numpy's FFT
            start_time = time.time()
            np.fft.fft(x)
            end_time = time.time()
            results['numpy'].append(end_time - start_time)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        for method, times in results.items():
            plt.plot(sizes, times, marker='o', linestyle='-', label=method)
        
        plt.title("FFT Performance Benchmark")
        plt.xlabel("Input Size")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.legend()
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()


def demonstrate_fft():
    """Demonstrate the Fast Fourier Transform with examples."""
    
    print("=== FFT DEMONSTRATION ===\n")
    
    # Create an FFT calculator
    fft_calc = FFT()
    
    # Example 1: Basic FFT and IFFT
    print("1. BASIC FFT AND IFFT")
    print("-------------------")
    
    # Create a simple signal: x = [1, 2, 3, 4]
    x = np.array([1, 2, 3, 4])
    
    # Compute the FFT
    X_recursive = fft_calc.fft_recursive(x)
    X_iterative = fft_calc.fft_iterative(x)
    X_numpy = np.fft.fft(x)
    
    print(f"Original signal: {x}")
    print(f"FFT (recursive): {X_recursive}")
    print(f"FFT (iterative): {X_iterative}")
    print(f"FFT (numpy): {X_numpy}")
    
    # Compute the inverse FFT
    x_reconstructed = fft_calc.ifft(X_recursive)
    
    print(f"Reconstructed signal: {x_reconstructed}")
    print(f"Error: {np.linalg.norm(x - x_reconstructed)}")
    
    # Example 2: FFT of sinusoidal signals
    print("\n2. FFT OF SINUSOIDAL SIGNALS")
    print("--------------------------")
    
    # Create a sum of two sinusoids
    sample_rate = 1000  # Hz
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    freq1 = 50  # Hz
    freq2 = 120  # Hz
    
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    
    # Compute and plot the FFT
    # fft_calc.plot_fft(signal, sample_rate, title="FFT of Sum of Sinusoids")
    
    # Get the frequency spectrum
    freqs, magnitudes = fft_calc.spectrum_analyze(signal, sample_rate)
    
    # Find the peaks
    peak_indices = np.argsort(magnitudes)[-2:]  # Get indices of two highest peaks
    peak_freqs = freqs[peak_indices]
    
    print(f"Original frequencies: {freq1} Hz, {freq2} Hz")
    print(f"Detected frequencies: {peak_freqs[0]:.1f} Hz, {peak_freqs[1]:.1f} Hz")
    
    # Example 3: Polynomial Multiplication
    print("\n3. POLYNOMIAL MULTIPLICATION")
    print("--------------------------")
    
    # Define two polynomials: p1 = 1 + 2x + 3x^2, p2 = 4 + 5x
    p1 = np.array([1, 2, 3])
    p2 = np.array([4, 5])
    
    # Expected result: p1 * p2 = 4 + 13x + 22x^2 + 15x^3
    product = fft_calc.polynomial_multiply(p1, p2)
    
    print(f"Polynomial 1: {p1} (coefficients in ascending order)")
    print(f"Polynomial 2: {p2}")
    print(f"Product: {product.real.astype(int)}")
    
    # Example 4: Convolution
    print("\n4. CONVOLUTION")
    print("------------")
    
    # Define two sequences
    x = np.array([1, 2, 3, 4])
    h = np.array([0.5, 0.5])  # Moving average filter
    
    # Compute the convolution
    y = fft_calc.convolve(x, h)
    y_numpy = np.convolve(x, h)
    
    print(f"Sequence 1: {x}")
    print(f"Sequence 2 (filter): {h}")
    print(f"Convolution (FFT): {y}")
    print(f"Convolution (NumPy): {y_numpy}")
    
    # Example 5: 2D FFT for Image Processing
    print("\n5. 2D FFT FOR IMAGE PROCESSING")
    print("----------------------------")
    
    # Create a simple 2D pattern
    size = 16
    pattern = np.zeros((size, size))
    pattern[size//4:3*size//4, size//4:3*size//4] = 1  # Create a square
    
    # Compute 2D FFT
    pattern_fft = fft_calc.fft2d(pattern)
    
    # Display magnitude of the 2D FFT (logarithmic scale for better visualization)
    magnitude = np.abs(pattern_fft)
    
    print(f"2D pattern shape: {pattern.shape}")
    print(f"2D FFT shape: {pattern_fft.shape}")
    print(f"Max magnitude: {np.max(magnitude)}")
    
    # Example 6: Benchmark
    print("\n6. BENCHMARK")
    print("----------")
    
    # Benchmark different methods with small sizes for demonstration
    sizes = [8, 16, 32, 64, 128]
    methods = ['recursive', 'iterative', 'numpy']
    
    results = {}
    for method in methods:
        results[method] = []
    
    for size in sizes:
        x = np.random.rand(size) + 1j * np.random.rand(size)
        
        # Time each method
        for method in methods:
            if method == 'numpy':
                start_time = time.time()
                np.fft.fft(x)
                end_time = time.time()
            else:
                start_time = time.time()
                fft_calc.fft(x, method=method)
                end_time = time.time()
            
            results[method].append(end_time - start_time)
    
    # Print results
    print("Execution time (seconds):")
    print(f"{'Size':<10}", end="")
    for method in methods:
        print(f"{method:<12}", end="")
    print()
    
    for i, size in enumerate(sizes):
        print(f"{size:<10}", end="")
        for method in methods:
            print(f"{results[method][i]:.6f}    ", end="")
        print()


if __name__ == "__main__":
    demonstrate_fft()