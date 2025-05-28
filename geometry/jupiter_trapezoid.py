import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class JupiterCalculator:
    """
    A class to calculate Jupiter's position and motion in time-velocity space
    using trapezoidal integration methods for high precision orbital mechanics.
    """
    
    # Physical constants
    G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
    M_SUN = 1.9885e30  # Solar mass (kg)
    M_JUPITER = 1.8982e27  # Jupiter mass (kg)
    
    # Jupiter's orbital parameters (mean values)
    SEMI_MAJOR_AXIS = 778.57e9  # meters
    ECCENTRICITY = 0.0489
    ORBITAL_PERIOD = 4332.59 * 86400  # seconds (11.86 years)
    
    def __init__(self, start_date, end_date, steps=1000):
        """
        Initialize the calculator with a time range and resolution.
        
        Args:
            start_date: datetime object for starting calculations
            end_date: datetime object for ending calculations
            steps: number of calculation steps (resolution)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.steps = steps
        
        # Time array (in seconds from start_date)
        self.time_span = (end_date - start_date).total_seconds()
        self.time_array = np.linspace(0, self.time_span, steps)
        
        # Results storage
        self.positions = None
        self.velocities = None
        self.accelerations = None
    
    def _mean_anomaly(self, time_seconds):
        """Calculate mean anomaly at given time."""
        mean_motion = 2 * np.pi / self.ORBITAL_PERIOD
        return (mean_motion * time_seconds) % (2 * np.pi)
    
    def _eccentric_anomaly(self, mean_anomaly, tol=1e-8, max_iter=100):
        """
        Solve Kepler's equation for eccentric anomaly using Newton-Raphson.
        E - e*sin(E) = M, where E is eccentric anomaly, e is eccentricity, M is mean anomaly
        """
        E = mean_anomaly  # Initial guess
        
        for i in range(max_iter):
            delta = (E - self.ECCENTRICITY * np.sin(E) - mean_anomaly) / (1 - self.ECCENTRICITY * np.cos(E))
            E = E - delta
            if abs(delta) < tol:
                break
                
        return E
    
    def _true_anomaly(self, eccentric_anomaly):
        """Calculate true anomaly from eccentric anomaly."""
        e = self.ECCENTRICITY
        cos_E = np.cos(eccentric_anomaly)
        sin_E = np.sin(eccentric_anomaly)
        
        # Formula for true anomaly from eccentric anomaly
        cos_v = (cos_E - e) / (1 - e * cos_E)
        sin_v = (np.sqrt(1 - e*e) * sin_E) / (1 - e * cos_E)
        
        return np.arctan2(sin_v, cos_v)
    
    def _calculate_position(self, true_anomaly):
        """Calculate heliocentric position from true anomaly."""
        e = self.ECCENTRICITY
        a = self.SEMI_MAJOR_AXIS
        
        # Distance from Sun to Jupiter
        r = a * (1 - e*e) / (1 + e * np.cos(true_anomaly))
        
        # Position in orbital plane
        x = r * np.cos(true_anomaly)
        y = r * np.sin(true_anomaly)
        
        return np.array([x, y, 0])  # Jupiter's orbit is nearly in the ecliptic plane
    
    def _calculate_velocity(self, true_anomaly):
        """Calculate orbital velocity at a given true anomaly."""
        # Semi-latus rectum
        p = self.SEMI_MAJOR_AXIS * (1 - self.ECCENTRICITY**2)
        
        # Gravitational parameter
        mu = self.G * self.M_SUN
        
        # Speed
        r = self.SEMI_MAJOR_AXIS * (1 - self.ECCENTRICITY**2) / (1 + self.ECCENTRICITY * np.cos(true_anomaly))
        speed = np.sqrt(mu * (2/r - 1/self.SEMI_MAJOR_AXIS))
        
        # Flight path angle
        phi = np.arctan(self.ECCENTRICITY * np.sin(true_anomaly) / (1 + self.ECCENTRICITY * np.cos(true_anomaly)))
        
        # Velocity components
        vx = -speed * np.sin(true_anomaly - phi)
        vy = speed * np.cos(true_anomaly - phi)
        
        return np.array([vx, vy, 0])
    
    def _calculate_acceleration(self, position):
        """Calculate acceleration due to solar gravity."""
        r = np.linalg.norm(position)
        return -self.G * self.M_SUN * position / r**3
    
    def compute_orbit(self):
        """
        Compute Jupiter's orbit using trapezoidal integration.
        This method calculates position, velocity, and acceleration at each time step.
        """
        positions = np.zeros((self.steps, 3))
        velocities = np.zeros((self.steps, 3))
        accelerations = np.zeros((self.steps, 3))
        
        # Initial conditions from Kepler's equations
        for i, t in enumerate(self.time_array):
            M = self._mean_anomaly(t)
            E = self._eccentric_anomaly(M)
            v = self._true_anomaly(E)
            
            positions[i] = self._calculate_position(v)
            velocities[i] = self._calculate_velocity(v)
            accelerations[i] = self._calculate_acceleration(positions[i])
        
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations
        
        return positions, velocities, accelerations
    
    def integrate_trapezoid(self, initial_position, initial_velocity, time_span=None):
        """
        Perform numerical integration using the trapezoidal method to compute
        Jupiter's trajectory directly from Newton's laws.
        
        This method offers an alternative to the Keplerian orbital elements approach,
        integrating the equations of motion directly.
        
        Args:
            initial_position: Initial position vector [x, y, z] in meters
            initial_velocity: Initial velocity vector [vx, vy, vz] in m/s
            time_span: Optional custom time span in seconds
        
        Returns:
            Tuple of (positions, velocities, accelerations) arrays
        """
        if time_span is None:
            time_span = self.time_span
            time_array = self.time_array
            steps = self.steps
        else:
            steps = self.steps
            time_array = np.linspace(0, time_span, steps)
        
        # Arrays to store results
        positions = np.zeros((steps, 3))
        velocities = np.zeros((steps, 3))
        accelerations = np.zeros((steps, 3))
        
        # Initial conditions
        positions[0] = initial_position
        velocities[0] = initial_velocity
        accelerations[0] = self._calculate_acceleration(initial_position)
        
        # Time step
        dt = time_array[1] - time_array[0]
        
        # Trapezoidal integration
        for i in range(1, steps):
            # Predict position using current velocity and half acceleration
            pred_pos = positions[i-1] + velocities[i-1] * dt + 0.5 * accelerations[i-1] * dt**2
            
            # Calculate new acceleration at predicted position
            pred_acc = self._calculate_acceleration(pred_pos)
            
            # Update position using trapezoidal rule
            positions[i] = positions[i-1] + velocities[i-1] * dt + 0.5 * (accelerations[i-1] + pred_acc) * dt**2
            
            # Calculate final acceleration
            accelerations[i] = self._calculate_acceleration(positions[i])
            
            # Update velocity using trapezoidal rule
            velocities[i] = velocities[i-1] + 0.5 * (accelerations[i-1] + accelerations[i]) * dt
        
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations
        
        return positions, velocities, accelerations
    
    def time_velocity_analysis(self):
        """
        Perform time-velocity space analysis using trapezoidal methods.
        Returns a dictionary of computed metrics in time-velocity space.
        """
        if self.positions is None or self.velocities is None:
            self.compute_orbit()
            
        # Calculate speed at each time step
        speeds = np.linalg.norm(self.velocities, axis=1)
        
        # Compute time-velocity metrics using trapezoidal rule
        dt = self.time_array[1] - self.time_array[0]
        
        # Distance traveled (path length)
        displacement = np.zeros(self.steps)
        for i in range(1, self.steps):
            displacement[i] = displacement[i-1] + np.linalg.norm(self.positions[i] - self.positions[i-1])
            
        # Average velocity using trapezoidal rule
        avg_velocity = np.trapz(speeds, self.time_array) / self.time_span
        
        # Kinetic energy over time
        kinetic_energy = 0.5 * self.M_JUPITER * speeds**2
        
        # Angular momentum over time
        angular_momentum = np.zeros((self.steps, 3))
        for i in range(self.steps):
            angular_momentum[i] = np.cross(self.positions[i], self.M_JUPITER * self.velocities[i])
        
        # Compute specific angular momentum magnitude
        h_magnitude = np.linalg.norm(angular_momentum, axis=1) / self.M_JUPITER
        
        return {
            "time": self.time_array,
            "speed": speeds,
            "displacement": displacement,
            "average_velocity": avg_velocity,
            "kinetic_energy": kinetic_energy,
            "angular_momentum": angular_momentum,
            "specific_angular_momentum": h_magnitude
        }
    
    def plot_orbit(self, title="Jupiter's Orbit"):
        """Plot Jupiter's orbit in the x-y plane."""
        if self.positions is None:
            self.compute_orbit()
            
        plt.figure(figsize=(10, 10))
        plt.plot(0, 0, 'yo', markersize=10, label='Sun')
        plt.plot(self.positions[:, 0], self.positions[:, 1], 'b-', label='Jupiter')
        
        # Add current position
        plt.plot(self.positions[-1, 0], self.positions[-1, 1], 'ro', markersize=5, label='Current Position')
        
        # Format plot
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        # Format axis with scientific notation
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        
        plt.show()
    
    def plot_time_velocity(self):
        """Plot Jupiter's velocity as a function of time."""
        analysis = self.time_velocity_analysis()
        
        # Convert time to days for better readability
        time_days = analysis["time"] / 86400
        
        plt.figure(figsize=(12, 8))
        
        # Plot speed over time
        plt.subplot(2, 1, 1)
        plt.plot(time_days, analysis["speed"] / 1000, 'b-')
        plt.title("Jupiter's Speed over Time")
        plt.xlabel('Time (days)')
        plt.ylabel('Speed (km/s)')
        plt.grid(True)
        
        # Plot kinetic energy over time
        plt.subplot(2, 1, 2)
        plt.plot(time_days, analysis["kinetic_energy"] / 1e27, 'r-')
        plt.title("Jupiter's Kinetic Energy over Time")
        plt.xlabel('Time (days)')
        plt.ylabel('Kinetic Energy (× 10²⁷ J)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Calculate Jupiter's position and motion over 1 year
    start_date = datetime(2025, 5, 20)  # Today
    end_date = datetime(2026, 5, 20)    # One year later
    
    # Create calculator with 1000 time steps
    jupiter = JupiterCalculator(start_date, end_date, steps=1000)
    
    # Method 1: Calculate using Keplerian elements
    jupiter.compute_orbit()
    
    # Method 2: Calculate using trapezoidal integration
    # Get initial conditions from Keplerian computation
    initial_position = jupiter.positions[0]
    initial_velocity = jupiter.velocities[0]
    
    # Run numerical integration with trapezoidal method
    jupiter.integrate_trapezoid(initial_position, initial_velocity)
    
    # Analyze in time-velocity space
    tv_analysis = jupiter.time_velocity_analysis()
    
    # Print some results
    print(f"Jupiter's average orbital speed: {tv_analysis['average_velocity']/1000:.2f} km/s")
    print(f"Maximum speed: {np.max(tv_analysis['speed'])/1000:.2f} km/s")
    print(f"Minimum speed: {np.min(tv_analysis['speed'])/1000:.2f} km/s")
    
    # Plot results
    jupiter.plot_orbit()
    jupiter.plot_time_velocity()