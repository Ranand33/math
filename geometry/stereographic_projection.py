import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import pandas as pd
import os
from matplotlib.offsetbox import AnchoredText
import urllib.request
import zipfile
import io

class StereographicProjection:
    """
    A class for implementing stereographic projection from a sphere to a plane.
    Specifically designed for mapping the Earth with various options for
    projection points and coordinate transformations.
    """
    
    def __init__(self, radius=6371.0, projection_point='north'):
        """
        Initialize stereographic projection parameters.
        
        Parameters:
        ----------
        radius : float
            Radius of the Earth in kilometers (default: 6371.0 km)
        projection_point : str or tuple
            Point from which to project. Options:
            - 'north': Project from North pole (default)
            - 'south': Project from South pole
            - (lat, lon): Custom projection point in degrees
        """
        self.R = radius
        
        # Set projection point
        if projection_point == 'north':
            self.projection_lat = 90.0
            self.projection_lon = 0.0
        elif projection_point == 'south':
            self.projection_lat = -90.0
            self.projection_lon = 0.0
        elif isinstance(projection_point, tuple) and len(projection_point) == 2:
            self.projection_lat, self.projection_lon = projection_point
        else:
            raise ValueError("projection_point must be 'north', 'south', or (lat, lon) tuple")
            
        # Convert projection point to 3D Cartesian coordinates
        self.projection_point_3d = self._lat_lon_to_cartesian(self.projection_lat, self.projection_lon)
        
        # Set antipodal point (used for handling singularity in projection)
        self.antipodal_lat = -self.projection_lat
        self.antipodal_lon = (self.projection_lon + 180.0) % 360.0 - 180.0
    
    def _lat_lon_to_cartesian(self, lat, lon):
        """
        Convert latitude and longitude to 3D Cartesian coordinates on the sphere.
        
        Parameters:
        ----------
        lat : float or array-like
            Latitude in degrees (-90 to 90)
        lon : float or array-like
            Longitude in degrees (-180 to 180)
            
        Returns:
        -------
        tuple or ndarray
            (x, y, z) coordinates
        """
        # Convert degrees to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Convert from spherical to Cartesian coordinates
        x = self.R * np.cos(lat_rad) * np.cos(lon_rad)
        y = self.R * np.cos(lat_rad) * np.sin(lon_rad)
        z = self.R * np.sin(lat_rad)
        
        return np.array([x, y, z])
    
    def _cartesian_to_lat_lon(self, x, y, z):
        """
        Convert 3D Cartesian coordinates to latitude and longitude.
        
        Parameters:
        ----------
        x, y, z : float or array-like
            Cartesian coordinates
            
        Returns:
        -------
        tuple
            (latitude, longitude) in degrees
        """
        # Handle zero or near-zero cases
        epsilon = 1e-10
        r_xy = np.sqrt(x**2 + y**2)
        
        # Calculate latitude
        lat_rad = np.arctan2(z, r_xy)
        lat = np.degrees(lat_rad)
        
        # Calculate longitude
        lon_rad = np.arctan2(y, x)
        lon = np.degrees(lon_rad)
        
        return lat, lon
    
    def forward(self, lat, lon):
        """
        Forward stereographic projection: map (lat, lon) to planar (x, y).
        
        Parameters:
        ----------
        lat : float or array-like
            Latitude in degrees (-90 to 90)
        lon : float or array-like
            Longitude in degrees (-180 to 180)
            
        Returns:
        -------
        tuple
            (x, y) planar coordinates in the same units as the radius
        """
        # Convert input to numpy arrays if they aren't already
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        
        # Check if points are too close to the antipodal point of the projection point
        # (these points approach infinity in the projection)
        tol = 1e-10
        is_near_antipodal = np.abs(lat - self.antipodal_lat) < tol
        if isinstance(is_near_antipodal, np.ndarray):
            if np.any(is_near_antipodal):
                near_idx = np.where(is_near_antipodal)[0]
                for idx in near_idx:
                    if abs((lon[idx] - self.antipodal_lon + 180) % 360 - 180) < tol:
                        raise ValueError(f"Point ({lat[idx]}, {lon[idx]}) is too close to the antipodal point of the projection")
        elif is_near_antipodal:
            if abs((lon - self.antipodal_lon + 180) % 360 - 180) < tol:
                raise ValueError(f"Point ({lat}, {lon}) is too close to the antipodal point of the projection")
        
        # Convert to 3D Cartesian coordinates
        P = self._lat_lon_to_cartesian(lat, lon)
        
        # Handle array inputs
        if isinstance(lat, np.ndarray) and lat.size > 1:
            # Reshape P if it's multi-dimensional
            original_shape = lat.shape
            P_reshaped = P.reshape(3, -1)
            
            # Project each point
            x_proj = np.zeros(P_reshaped.shape[1])
            y_proj = np.zeros(P_reshaped.shape[1])
            
            for i in range(P_reshaped.shape[1]):
                point = P_reshaped[:, i]
                N = self.projection_point_3d
                
                # Vector from N to P
                v = point - N
                
                # Plane point: intersection of line N->P with the tangent plane at N
                # We use the fact that the normal to the tangent plane at N is parallel to N
                # The plane equation is dot(X - N, N) = 0
                # The line equation is X = N + t * v
                # Solving for t: dot(t * v, N) = 0
                # If N and v are parallel (antipodal points), this is undefined
                
                dot_product = np.dot(v, N)
                if abs(dot_product) < tol:
                    # This happens when the point is antipodal to N or very close to N
                    x_proj[i] = np.nan
                    y_proj[i] = np.nan
                else:
                    # Scale factor
                    t = -np.dot(N, N) / dot_product
                    
                    # Intersection point
                    intersection = N + t * v
                    
                    # Project onto the xy-plane (tangent to the sphere at N)
                    # We need to orient the plane based on the projection point
                    
                    # Create a local coordinate system with N as the z-axis
                    z_axis = N / np.linalg.norm(N)
                    
                    # Choose any vector not parallel to z_axis for an initial x-axis
                    if abs(z_axis[0]) < 0.9:
                        x_init = np.array([1.0, 0.0, 0.0])
                    else:
                        x_init = np.array([0.0, 1.0, 0.0])
                    
                    # Compute y-axis and orthogonalize x-axis
                    y_axis = np.cross(z_axis, x_init)
                    y_axis = y_axis / np.linalg.norm(y_axis)
                    x_axis = np.cross(y_axis, z_axis)
                    
                    # Get the coordinates in the local system
                    x_proj[i] = np.dot(intersection - N, x_axis)
                    y_proj[i] = np.dot(intersection - N, y_axis)
            
            # Reshape back to original shape
            x_proj = x_proj.reshape(original_shape)
            y_proj = y_proj.reshape(original_shape)
            
        else:
            # Single point case
            N = self.projection_point_3d
            
            # Vector from N to P
            v = P - N
            
            # Check if point is antipodal to N
            dot_product = np.dot(v, N)
            if abs(dot_product) < tol:
                return np.nan, np.nan
            
            # Scale factor
            t = -np.dot(N, N) / dot_product
            
            # Intersection point
            intersection = N + t * v
            
            # Create a local coordinate system with N as the z-axis
            z_axis = N / np.linalg.norm(N)
            
            # Choose any vector not parallel to z_axis for an initial x-axis
            if abs(z_axis[0]) < 0.9:
                x_init = np.array([1.0, 0.0, 0.0])
            else:
                x_init = np.array([0.0, 1.0, 0.0])
            
            # Compute y-axis and orthogonalize x-axis
            y_axis = np.cross(z_axis, x_init)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            # Get the coordinates in the local system
            x_proj = np.dot(intersection - N, x_axis)
            y_proj = np.dot(intersection - N, y_axis)
        
        return x_proj, y_proj
    
    def inverse(self, x, y):
        """
        Inverse stereographic projection: map planar (x, y) to (lat, lon).
        
        Parameters:
        ----------
        x, y : float or array-like
            Planar coordinates
            
        Returns:
        -------
        tuple
            (latitude, longitude) in degrees
        """
        # Convert input to numpy arrays if they aren't already
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Handle array inputs
        if isinstance(x, np.ndarray) and x.size > 1:
            # Process arrays
            original_shape = x.shape
            x_flat = x.flatten()
            y_flat = y.flatten()
            
            lat_result = np.zeros_like(x_flat)
            lon_result = np.zeros_like(x_flat)
            
            for i in range(x_flat.size):
                x_i, y_i = x_flat[i], y_flat[i]
                
                # Handle the inverse mapping for each point
                N = self.projection_point_3d
                z_axis = N / np.linalg.norm(N)
                
                # Choose any vector not parallel to z_axis for an initial x-axis
                if abs(z_axis[0]) < 0.9:
                    x_init = np.array([1.0, 0.0, 0.0])
                else:
                    x_init = np.array([0.0, 1.0, 0.0])
                
                # Compute y-axis and orthogonalize x-axis
                y_axis = np.cross(z_axis, x_init)
                y_axis = y_axis / np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                
                # Point on the tangent plane
                P_tangent = N + x_i * x_axis + y_i * y_axis
                
                # Ray from origin through P_tangent
                ray_dir = P_tangent
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Intersection with the sphere
                # The equation is |O + t*ray_dir| = R
                # Solving for t: t^2 = R^2
                t = self.R
                
                # Intersection point
                P_sphere = t * ray_dir
                
                # Convert back to latitude and longitude
                lat, lon = self._cartesian_to_lat_lon(P_sphere[0], P_sphere[1], P_sphere[2])
                lat_result[i] = lat
                lon_result[i] = lon
            
            # Reshape back to original shape
            lat_result = lat_result.reshape(original_shape)
            lon_result = lon_result.reshape(original_shape)
            
        else:
            # Single point case
            N = self.projection_point_3d
            z_axis = N / np.linalg.norm(N)
            
            # Choose any vector not parallel to z_axis for an initial x-axis
            if abs(z_axis[0]) < 0.9:
                x_init = np.array([1.0, 0.0, 0.0])
            else:
                x_init = np.array([0.0, 1.0, 0.0])
            
            # Compute y-axis and orthogonalize x-axis
            y_axis = np.cross(z_axis, x_init)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            # Point on the tangent plane
            P_tangent = N + x * x_axis + y * y_axis
            
            # Ray from origin through P_tangent
            ray_dir = P_tangent
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            # Intersection with the sphere
            t = self.R
            
            # Intersection point
            P_sphere = t * ray_dir
            
            # Convert back to latitude and longitude
            lat_result, lon_result = self._cartesian_to_lat_lon(P_sphere[0], P_sphere[1], P_sphere[2])
        
        return lat_result, lon_result
    
    def calculate_distortion(self, lat, lon):
        """
        Calculate the distortion (scale factor) at given points.
        
        Parameters:
        ----------
        lat : float or array-like
            Latitude in degrees (-90 to 90)
        lon : float or array-like
            Longitude in degrees (-180 to 180)
            
        Returns:
        -------
        float or array-like
            Scale factor at each point
        """
        # Convert lat/lon to 3D Cartesian coordinates
        P = self._lat_lon_to_cartesian(lat, lon)
        N = self.projection_point_3d
        
        # For a stereographic projection from point N,
        # the scale factor at a point P is:
        # k = 2R / (R + N·P/|N||P|)
        # where R is the radius, N is the projection point, and P is the point on the sphere
        
        # Normalize N and P
        N_norm = N / np.linalg.norm(N)
        
        if isinstance(lat, np.ndarray) and lat.size > 1:
            # Reshape P if it's multi-dimensional
            original_shape = lat.shape
            P_reshaped = P.reshape(3, -1)
            
            # Calculate scale factor for each point
            scale_factor = np.zeros(P_reshaped.shape[1])
            
            for i in range(P_reshaped.shape[1]):
                point = P_reshaped[:, i]
                P_norm = point / np.linalg.norm(point)
                
                # Dot product N·P
                dot_product = np.dot(N_norm, P_norm)
                
                # Scale factor
                scale_factor[i] = 2.0 / (1.0 + dot_product)
            
            # Reshape back to original shape
            scale_factor = scale_factor.reshape(original_shape)
            
        else:
            # Single point case
            P_norm = P / np.linalg.norm(P)
            
            # Dot product N·P
            dot_product = np.dot(N_norm, P_norm)
            
            # Scale factor
            scale_factor = 2.0 / (1.0 + dot_product)
        
        return scale_factor
    
    def plot_grid(self, lat_spacing=10, lon_spacing=10, projection_point=None, ax=None, color='gray', alpha=0.5):
        """
        Plot a graticule (lat/lon grid) in stereographic projection.
        
        Parameters:
        ----------
        lat_spacing : float
            Spacing between latitude lines in degrees
        lon_spacing : float
            Spacing between longitude lines in degrees
        projection_point : str or tuple, optional
            Point from which to project (overrides the instance projection point)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        color : str, optional
            Color of grid lines
        alpha : float, optional
            Transparency of grid lines
            
        Returns:
        -------
        matplotlib.axes.Axes
            The axes with the plot
        """
        if projection_point is not None:
            # Create a new projector with the specified point
            proj = StereographicProjection(radius=self.R, projection_point=projection_point)
        else:
            proj = self
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create the grid
        lats = np.arange(-90, 91, lat_spacing)
        lons = np.arange(-180, 181, lon_spacing)
        
        # Plot latitude lines
        for lat in lats:
            if lat == -90 or lat == 90:
                continue  # Skip poles
            
            lon_values = np.linspace(-180, 180, 360)
            x, y = proj.forward(np.full_like(lon_values, lat), lon_values)
            
            # Skip lines that contain NaN values
            if not np.isnan(x).any() and not np.isnan(y).any():
                ax.plot(x, y, color=color, alpha=alpha, linestyle='-', linewidth=0.5)
        
        # Plot longitude lines
        for lon in lons:
            lat_values = np.linspace(-89.9, 89.9, 180)  # Avoid exactly at poles
            x, y = proj.forward(lat_values, np.full_like(lat_values, lon))
            
            # Skip lines that contain NaN values
            if not np.isnan(x).any() and not np.isnan(y).any():
                ax.plot(x, y, color=color, alpha=alpha, linestyle='-', linewidth=0.5)
        
        # Draw the edge of the projection (the image of the equator)
        if abs(proj.projection_lat) == 90:
            # For pole projections, the boundary is a circle
            equator_lons = np.linspace(0, 360, 360)
            equator_lats = np.zeros_like(equator_lons)
            x, y = proj.forward(equator_lats, equator_lons)
            
            # Find the radius (should be constant for all points)
            radius = np.sqrt(x[0]**2 + y[0]**2)
            circle = plt.Circle((0, 0), radius, fill=False, color=color, linestyle='-', linewidth=1.5)
            ax.add_artist(circle)
        
        ax.set_aspect('equal')
        ax.grid(False)
        
        return ax
    
    def plot_coastlines(self, ax=None, color='black', linewidth=1.0, projection_point=None):
        """
        Plot Earth's coastlines in stereographic projection.
        
        Parameters:
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        color : str, optional
            Color of coastlines
        linewidth : float, optional
            Width of coastline lines
        projection_point : str or tuple, optional
            Point from which to project (overrides the instance projection point)
            
        Returns:
        -------
        matplotlib.axes.Axes
            The axes with the plot
        """
        if projection_point is not None:
            # Create a new projector with the specified point
            proj = StereographicProjection(radius=self.R, projection_point=projection_point)
        else:
            proj = self
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use cartopy's coastlines data
        land_features = cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                   facecolor='none', edgecolor=color)
        
        # Get the geometries from the feature
        for geometry in land_features.geometries():
            if hasattr(geometry, 'coords'):
                # It's a LineString
                coords = np.array(geometry.coords)
                lons, lats = coords[:, 0], coords[:, 1]
                x, y = proj.forward(lats, lons)
                
                # Skip segments with NaN values
                valid_indices = ~np.isnan(x) & ~np.isnan(y)
                segments = np.split(np.arange(len(x)), np.where(~valid_indices)[0])
                
                for segment in segments:
                    if len(segment) > 1 and np.all(valid_indices[segment]):
                        ax.plot(x[segment], y[segment], color=color, linewidth=linewidth)
            
            elif hasattr(geometry, 'geoms'):
                # It's a MultiLineString or similar
                for geom in geometry.geoms:
                    if hasattr(geom, 'coords'):
                        coords = np.array(geom.coords)
                        lons, lats = coords[:, 0], coords[:, 1]
                        x, y = proj.forward(lats, lons)
                        
                        # Skip segments with NaN values
                        valid_indices = ~np.isnan(x) & ~np.isnan(y)
                        segments = np.split(np.arange(len(x)), np.where(~valid_indices)[0])
                        
                        for segment in segments:
                            if len(segment) > 1 and np.all(valid_indices[segment]):
                                ax.plot(x[segment], y[segment], color=color, linewidth=linewidth)
        
        ax.set_aspect('equal')
        ax.grid(False)
        
        return ax
    
    def plot_distortion(self, ax=None, resolution=1, projection_point=None):
        """
        Plot distortion map for the stereographic projection.
        
        Parameters:
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        resolution : float, optional
            Resolution of the distortion grid in degrees
        projection_point : str or tuple, optional
            Point from which to project (overrides the instance projection point)
            
        Returns:
        -------
        matplotlib.axes.Axes
            The axes with the plot
        """
        if projection_point is not None:
            # Create a new projector with the specified point
            proj = StereographicProjection(radius=self.R, projection_point=projection_point)
        else:
            proj = self
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a grid of lat/lon points
        lats = np.arange(-89, 90, resolution)
        lons = np.arange(-180, 180, resolution)
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        
        # Calculate distortion at each point
        distortion = proj.calculate_distortion(lat_grid, lon_grid)
        
        # Project the grid
        x, y = proj.forward(lat_grid, lon_grid)
        
        # Create a triangulation for the projection
        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        distortion_valid = distortion[valid_indices]
        
        # Use grid data to interpolate onto a regular grid for plotting
        xi = np.linspace(np.min(x_valid), np.max(x_valid), 500)
        yi = np.linspace(np.min(y_valid), np.max(y_valid), 500)
        XI, YI = np.meshgrid(xi, yi)
        
        # Interpolate distortion values
        ZI = griddata((x_valid, y_valid), distortion_valid, (XI, YI), method='cubic')
        
        # Plot distortion map
        cmap = plt.cm.viridis
        contour = ax.contourf(XI, YI, ZI, 50, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Scale Factor (Distortion)')
        
        # Draw coastlines on top
        proj.plot_coastlines(ax=ax, color='black', linewidth=0.5)
        
        # Add a graticule
        proj.plot_grid(lat_spacing=20, lon_spacing=20, ax=ax, color='gray', alpha=0.5)
        
        ax.set_aspect('equal')
        ax.set_title('Distortion Map for Stereographic Projection')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        
        return ax
    
    def plot_world_map(self, projection_point=None, show_distortion=False, resolution=2):
        """
        Plot a complete world map using stereographic projection.
        
        Parameters:
        ----------
        projection_point : str or tuple, optional
            Point from which to project (overrides the instance projection point)
        show_distortion : bool, optional
            Whether to show the distortion overlay
        resolution : float, optional
            Resolution of the distortion grid in degrees if shown
            
        Returns:
        -------
        matplotlib.figure.Figure
            The figure with the plot
        """
        if projection_point is not None:
            # Create a new projector with the specified point
            proj = StereographicProjection(radius=self.R, projection_point=projection_point)
        else:
            proj = self
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        if show_distortion:
            # Plot distortion map
            proj.plot_distortion(ax=ax, resolution=resolution)
        else:
            # Plot coastlines and grid
            proj.plot_coastlines(ax=ax)
            proj.plot_grid(lat_spacing=15, lon_spacing=15, ax=ax)
        
        # Determine projection point information for title
        if proj.projection_lat == 90:
            proj_point_str = "North Pole"
        elif proj.projection_lat == -90:
            proj_point_str = "South Pole"
        else:
            proj_point_str = f"({proj.projection_lat:.1f}°, {proj.projection_lon:.1f}°)"
        
        ax.set_title(f'Stereographic Projection from {proj_point_str}')
        
        # Add explanation box
        explanation = (
            "Stereographic Projection\n"
            f"Projection Point: {proj_point_str}\n"
            "Properties:\n"
            "- Conformal (preserves angles)\n"
            "- Not equal-area (distorts sizes)\n"
            "- Circles map to circles"
        )
        
        at = AnchoredText(explanation, loc='lower left', prop=dict(size=10), frameon=True)
        at.patch.set_boxstyle("round,pad=0.3")
        at.patch.set_alpha(0.8)
        ax.add_artist(at)
        
        # Set equal aspect ratio and remove axes
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        plt.tight_layout()
        
        return fig
    
    def visualize_3d_projection(self, lat=None, lon=None, projection_point=None):
        """
        Visualize the stereographic projection process in 3D.
        
        Parameters:
        ----------
        lat, lon : float or array-like, optional
            Specific points to project
        projection_point : str or tuple, optional
            Point from which to project (overrides the instance projection point)
            
        Returns:
        -------
        matplotlib.figure.Figure
            The figure with the 3D plot
        """
        if projection_point is not None:
            # Create a new projector with the specified point
            proj = StereographicProjection(radius=self.R, projection_point=projection_point)
        else:
            proj = self
        
        # Create figure
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw the Earth as a wireframe sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        x = proj.R * np.outer(np.cos(u), np.sin(v))
        y = proj.R * np.outer(np.sin(u), np.sin(v))
        z = proj.R * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot sphere with light blue color and transparency
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        
        # Draw lat/lon grid on the sphere
        lat_lines = np.arange(-90, 91, 30)
        lon_lines = np.arange(-180, 181, 30)
        
        for lat_line in lat_lines:
            if lat_line == -90 or lat_line == 90:
                continue  # Skip poles
                
            lons = np.linspace(-180, 180, 100)
            lats = np.full_like(lons, lat_line)
            points = proj._lat_lon_to_cartesian(lats, lons)
            ax.plot(points[0], points[1], points[2], color='gray', alpha=0.5, linewidth=0.5)
        
        for lon_line in lon_lines:
            lats = np.linspace(-89, 89, 100)  # Avoid exact poles
            lons = np.full_like(lats, lon_line)
            points = proj._lat_lon_to_cartesian(lats, lons)
            ax.plot(points[0], points[1], points[2], color='gray', alpha=0.5, linewidth=0.5)
        
        # Determine the projection plane
        N = proj.projection_point_3d
        N_norm = N / np.linalg.norm(N)
        
        # Create a tangent plane at the projection point
        # The plane is perpendicular to the line from origin to projection point
        d = np.dot(N_norm, N)  # Distance from origin to plane along normal
        
        # Create a grid of points on the tangent plane
        xx, yy = np.meshgrid(np.linspace(-1.5*proj.R, 1.5*proj.R, 10), 
                            np.linspace(-1.5*proj.R, 1.5*proj.R, 10))
        
        # Create local coordinate system on the tangent plane
        if abs(N_norm[0]) < 0.9:
            x_init = np.array([1.0, 0.0, 0.0])
        else:
            x_init = np.array([0.0, 1.0, 0.0])
            
        y_axis = np.cross(N_norm, x_init)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, N_norm)
        
        # Generate grid points on the tangent plane
        plane_points = np.zeros((xx.shape[0], xx.shape[1], 3))
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                plane_points[i, j] = N + xx[i, j] * x_axis + yy[i, j] * y_axis
        
        # Plot the tangent plane
        ax.plot_surface(plane_points[:, :, 0], plane_points[:, :, 1], plane_points[:, :, 2],
                      color='green', alpha=0.2)
        
        # Draw the projection point
        ax.scatter([N[0]], [N[1]], [N[2]], color='red', s=100, label='Projection Point')
        
        # Draw projection lines for specific points if provided
        if lat is not None and lon is not None:
            if not isinstance(lat, (list, np.ndarray)):
                lat = [lat]
            if not isinstance(lon, (list, np.ndarray)):
                lon = [lon]
            
            for la, lo in zip(lat, lon):
                # Point on the sphere
                P = proj._lat_lon_to_cartesian(la, lo)
                ax.scatter([P[0]], [P[1]], [P[2]], color='blue', s=50, label='Point on Sphere')
                
                # Draw line from projection point to the point on the sphere
                ax.plot([N[0], P[0]], [N[1], P[1]], [N[2], P[2]], 'r-', alpha=0.7)
                
                # Calculate the projected point on the plane
                try:
                    x_proj, y_proj = proj.forward(la, lo)
                    
                    # Convert back to 3D coordinates
                    proj_point = N + x_proj * x_axis + y_proj * y_axis
                    ax.scatter([proj_point[0]], [proj_point[1]], [proj_point[2]], 
                             color='orange', s=50, label='Projected Point')
                    
                    # Draw line from projection point to projected point
                    ax.plot([N[0], proj_point[0]], [N[1], proj_point[1]], [N[2], proj_point[2]], 
                           'g-', alpha=0.7)
                except:
                    print(f"Could not project point ({la}, {lo})")
        
        # Draw the origin
        ax.scatter([0], [0], [0], color='black', s=50, label='Earth Center')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set title and labels
        ax.set_title('3D Visualization of Stereographic Projection Process')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        
        # Remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.tight_layout()
        
        return fig

    def plot_elevation_map(self, elevation_data=None, projection_point=None, resolution=1.0, 
                           show_colorbar=True, cmap='terrain'):
        """
        Plot Earth's elevation using stereographic projection.
        
        Parameters:
        ----------
        elevation_data : tuple or None, optional
            (lats, lons, elevations) arrays, or None to download data
        projection_point : str or tuple, optional
            Point from which to project (overrides the instance projection point)
        resolution : float, optional
            Resolution for downloaded data in degrees
        show_colorbar : bool, optional
            Whether to show the colorbar
        cmap : str, optional
            Colormap for elevation
            
        Returns:
        -------
        matplotlib.figure.Figure
            The figure with the plot
        """
        if projection_point is not None:
            # Create a new projector with the specified point
            proj = StereographicProjection(radius=self.R, projection_point=projection_point)
        else:
            proj = self
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Get elevation data if not provided
        if elevation_data is None:
            # Download sample elevation data (this would be a simplified version)
            print("Downloading sample elevation data...")
            
            # In a real implementation, you might fetch global elevation data from a service
            # For demonstration, we'll create simplified elevation data
            lat_range = np.arange(-90, 90, resolution)
            lon_range = np.arange(-180, 180, resolution)
            lats, lons = np.meshgrid(lat_range, lon_range)
            
            # Simplified elevation model (in a real implementation, download actual data)
            elevations = (
                # Major mountain ranges approximately
                2000 * np.exp(-((lats - 35)**2 + (lons - -110)**2) / 300) +  # Rockies
                3000 * np.exp(-((lats - 30)**2 + (lons - 80)**2) / 300) +    # Himalayas
                1500 * np.exp(-((lats - -20)**2 + (lons - -60)**2) / 300) +  # Andes
                1000 * np.exp(-((lats - 5)**2 + (lons - 35)**2) / 300) +     # Ethiopian Highlands
                1200 * np.exp(-((lats - -30)**2 + (lons - 150)**2) / 300) +  # Great Dividing Range
                # Ocean trenches
                -3000 * np.exp(-((lats - 10)**2 + (lons - 140)**2) / 200) +  # Mariana Trench
                -2000 * np.exp(-((lats - -40)**2 + (lons - -80)**2) / 200)   # Peru-Chile Trench
            )
            
            # Flatten for griddata
            lats_flat = lats.flatten()
            lons_flat = lons.flatten()
            elevations_flat = elevations.flatten()
            
        else:
            # Use provided data
            lats_flat, lons_flat, elevations_flat = elevation_data
        
        # Project the coordinates
        x, y = proj.forward(lats_flat, lons_flat)
        
        # Filter out invalid points (e.g., near the antipode of projection point)
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        elevations_valid = elevations_flat[valid_mask]
        
        # Create a regular grid for the projection
        margin = 0.05
        x_min, x_max = np.min(x_valid), np.max(x_valid)
        y_min, y_max = np.min(y_valid), np.max(y_valid)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        xi = np.linspace(x_min - margin * x_range, x_max + margin * x_range, 500)
        yi = np.linspace(y_min - margin * y_range, y_max + margin * y_range, 500)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate elevation data onto the regular grid
        zi_grid = griddata((x_valid, y_valid), elevations_valid, (xi_grid, yi_grid), method='cubic')
        
        # Create a colormap that handles NaN values
        cmap_with_nan = plt.cm.get_cmap(cmap).copy()
        cmap_with_nan.set_bad('white', 1.0)
        
        # Create the elevation map
        im = ax.pcolormesh(xi_grid, yi_grid, zi_grid, cmap=cmap_with_nan, shading='auto')
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Elevation (m)')
        
        # Add coastlines
        proj.plot_coastlines(ax=ax, color='black', linewidth=0.5)
        
        # Add graticule
        proj.plot_grid(lat_spacing=30, lon_spacing=30, ax=ax, color='white', alpha=0.3)
        
        # Determine projection point information for title
        if proj.projection_lat == 90:
            proj_point_str = "North Pole"
        elif proj.projection_lat == -90:
            proj_point_str = "South Pole"
        else:
            proj_point_str = f"({proj.projection_lat:.1f}°, {proj.projection_lon:.1f}°)"
        
        ax.set_title(f'Stereographic Projection of Earth Relief from {proj_point_str}')
        
        # Set equal aspect ratio and remove axes
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        plt.tight_layout()
        
        return fig


def download_sample_elevation():
    """
    Download sample elevation data (ETOPO1 Global Relief Model).
    
    Returns:
    -------
    tuple
        (lats, lons, elevations) arrays
    """
    print("This would download the ETOPO1 Global Relief Model.")
    print("For demonstration purposes, we'll create simplified elevation data.")
    
    # Create a simple grid
    lat_range = np.arange(-90, 90, 5)
    lon_range = np.arange(-180, 180, 5)
    lats, lons = np.meshgrid(lat_range, lon_range)
    
    # Simplified elevation model
    elevations = (
        # Major mountain ranges approximately
        2000 * np.exp(-((lats - 35)**2 + (lons - -110)**2) / 300) +  # Rockies
        3000 * np.exp(-((lats - 30)**2 + (lons - 80)**2) / 300) +    # Himalayas
        1500 * np.exp(-((lats - -20)**2 + (lons - -60)**2) / 300) +  # Andes
        1000 * np.exp(-((lats - 5)**2 + (lons - 35)**2) / 300) +     # Ethiopian Highlands
        1200 * np.exp(-((lats - -30)**2 + (lons - 150)**2) / 300) +  # Great Dividing Range
        # Ocean trenches
        -3000 * np.exp(-((lats - 10)**2 + (lons - 140)**2) / 200) +  # Mariana Trench
        -2000 * np.exp(-((lats - -40)**2 + (lons - -80)**2) / 200)   # Peru-Chile Trench
    )
    
    return lats.flatten(), lons.flatten(), elevations.flatten()


def main():
    """
    Main function to demonstrate stereographic projection capabilities.
    """
    print("Stereographic Projection for Mapping the Earth")
    print("=============================================")
    
    # Create a stereographic projector from the North Pole
    projector = StereographicProjection(projection_point='north')
    
    # Demo 1: Basic world map from North Pole
    print("\n1. Creating a basic world map from the North Pole...")
    fig1 = projector.plot_world_map()
    
    # Demo 2: South Pole projection
    print("\n2. Creating a map from the South Pole...")
    fig2 = StereographicProjection(projection_point='south').plot_world_map()
    
    # Demo 3: Projection from an arbitrary point
    print("\n3. Creating a map from an arbitrary point (0°N, 0°E)...")
    fig3 = StereographicProjection(projection_point=(0, 0)).plot_world_map()
    
    # Demo 4: Distortion visualization
    print("\n4. Visualizing distortion in the stereographic projection...")
    fig4 = projector.plot_world_map(show_distortion=True)
    
    # Demo 5: 3D visualization of the projection process
    print("\n5. Creating a 3D visualization of the projection process...")
    fig5 = projector.visualize_3d_projection(lat=[0, 30, 60], lon=[0, 45, 90])
    
    # Demo 6: Elevation map using stereographic projection
    print("\n6. Creating an elevation map...")
    elevation_data = download_sample_elevation()
    fig6 = projector.plot_elevation_map(elevation_data)
    
    print("\nAll demonstrations complete. Showing plots...")
    plt.show()


if __name__ == "__main__":
    main()