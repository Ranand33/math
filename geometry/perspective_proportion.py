import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, PathPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.path import Path
import math
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict
import cv2  # For more advanced perspective transformations

class LinearPerspective:
    """
    A class for creating and visualizing linear perspective systems (1-point, 2-point, and 3-point).
    """
    
    def __init__(self, width=10, height=8, horizon_height=4):
        """
        Initialize a perspective system.
        
        Args:
            width: Width of the canvas
            height: Height of the canvas
            horizon_height: Height of the horizon line
        """
        self.width = width
        self.height = height
        self.horizon_height = horizon_height
        self.vanishing_points = []
        self.perspective_type = None
        
    def set_one_point_perspective(self, vp_x=None):
        """
        Set up a one-point perspective system with a single vanishing point.
        
        Args:
            vp_x: x-coordinate of the vanishing point (centered by default)
        """
        if vp_x is None:
            vp_x = self.width / 2
            
        self.vanishing_points = [(vp_x, self.horizon_height)]
        self.perspective_type = "one-point"
        
    def set_two_point_perspective(self, vp1_x=None, vp2_x=None):
        """
        Set up a two-point perspective system with two vanishing points.
        
        Args:
            vp1_x: x-coordinate of the left vanishing point
            vp2_x: x-coordinate of the right vanishing point
        """
        if vp1_x is None:
            vp1_x = -self.width / 2
        if vp2_x is None:
            vp2_x = self.width * 1.5
            
        self.vanishing_points = [
            (vp1_x, self.horizon_height),
            (vp2_x, self.horizon_height)
        ]
        self.perspective_type = "two-point"
        
    def set_three_point_perspective(self, vp1_x=None, vp2_x=None, vp3_x=None, vp3_y=None):
        """
        Set up a three-point perspective system with three vanishing points.
        
        Args:
            vp1_x: x-coordinate of the left vanishing point
            vp2_x: x-coordinate of the right vanishing point
            vp3_x: x-coordinate of the vertical vanishing point (centered by default)
            vp3_y: y-coordinate of the vertical vanishing point (below or above the horizon)
        """
        if vp1_x is None:
            vp1_x = -self.width / 2
        if vp2_x is None:
            vp2_x = self.width * 1.5
        if vp3_x is None:
            vp3_x = self.width / 2
        if vp3_y is None:
            vp3_y = self.height * 2  # Above the canvas by default
            
        self.vanishing_points = [
            (vp1_x, self.horizon_height),
            (vp2_x, self.horizon_height),
            (vp3_x, vp3_y)
        ]
        self.perspective_type = "three-point"
        
    def draw_perspective_grid(self, num_lines=10, ax=None, color='gray', alpha=0.5):
        """
        Draw a perspective grid based on the current perspective setup.
        
        Args:
            num_lines: Number of grid lines to draw
            ax: Matplotlib axis to draw on
            color: Color of the grid lines
            alpha: Transparency of the grid lines
            
        Returns:
            Matplotlib axis with the perspective grid
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))
        
        # Draw horizon line
        ax.axhline(y=self.horizon_height, color='blue', linestyle='--', alpha=0.7, label='Horizon Line')
        
        # Draw vanishing points
        for i, vp in enumerate(self.vanishing_points):
            if 0 <= vp[0] <= self.width and 0 <= vp[1] <= self.height:
                ax.plot(vp[0], vp[1], 'ro', markersize=8, label=f'Vanishing Point {i+1}')
        
        # Draw perspective grid based on perspective type
        if self.perspective_type == "one-point":
            self._draw_one_point_grid(num_lines, ax, color, alpha)
        elif self.perspective_type == "two-point":
            self._draw_two_point_grid(num_lines, ax, color, alpha)
        elif self.perspective_type == "three-point":
            self._draw_three_point_grid(num_lines, ax, color, alpha)
            
        # Set plot limits and labels
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{self.perspective_type.capitalize()} Perspective Grid')
        ax.legend(loc='upper right')
        
        return ax
    
    def _draw_one_point_grid(self, num_lines, ax, color, alpha):
        """Helper method to draw a one-point perspective grid."""
        vp = self.vanishing_points[0]
        
        # Draw horizontal lines
        y_spacing = self.height / (num_lines + 1)
        for i in range(1, num_lines + 1):
            y = i * y_spacing
            if y != self.horizon_height:  # Skip the horizon line
                ax.axhline(y=y, color=color, alpha=alpha)
        
        # Draw vertical lines
        x_spacing = self.width / (num_lines + 1)
        for i in range(1, num_lines + 1):
            x = i * x_spacing
            ax.axvline(x=x, color=color, alpha=alpha)
        
        # Draw converging lines
        for i in range(0, num_lines + 2):
            # Starting points along the bottom of the canvas
            x_bottom = i * (self.width / (num_lines + 1))
            # Starting points along the left edge
            y_left = i * (self.height / (num_lines + 1))
            # Starting points along the right edge
            y_right = i * (self.height / (num_lines + 1))
            
            # Connect bottom edge points to vanishing point
            ax.plot([x_bottom, vp[0]], [0, vp[1]], color=color, alpha=alpha)
            # Connect top edge points to vanishing point
            ax.plot([x_bottom, vp[0]], [self.height, vp[1]], color=color, alpha=alpha)
            # Connect left edge points to vanishing point if not on horizon
            if abs(y_left - self.horizon_height) > 0.1:
                ax.plot([0, vp[0]], [y_left, vp[1]], color=color, alpha=alpha)
            # Connect right edge points to vanishing point if not on horizon
            if abs(y_right - self.horizon_height) > 0.1:
                ax.plot([self.width, vp[0]], [y_right, vp[1]], color=color, alpha=alpha)
    
    def _draw_two_point_grid(self, num_lines, ax, color, alpha):
        """Helper method to draw a two-point perspective grid."""
        vp1, vp2 = self.vanishing_points
        
        # Draw horizontal lines
        y_spacing = self.height / (num_lines + 1)
        for i in range(1, num_lines + 1):
            y = i * y_spacing
            if y != self.horizon_height:  # Skip the horizon line
                ax.axhline(y=y, color=color, alpha=alpha)
        
        # Calculate the central vertical line (usually where the object is located)
        center_x = self.width / 2
        ax.axvline(x=center_x, color=color, linestyle='-', alpha=alpha)
        
        # Draw converging lines to left vanishing point
        for i in range(0, num_lines + 2):
            # Starting points along the bottom of the canvas
            y = i * (self.height / (num_lines + 1))
            if abs(y - self.horizon_height) > 0.1:  # Skip if too close to horizon
                ax.plot([center_x, vp1[0]], [y, vp1[1]], color=color, alpha=alpha)
                ax.plot([self.width, vp1[0]], [y, vp1[1]], color=color, alpha=alpha)
        
        # Draw converging lines to right vanishing point
        for i in range(0, num_lines + 2):
            # Starting points along the bottom of the canvas
            y = i * (self.height / (num_lines + 1))
            if abs(y - self.horizon_height) > 0.1:  # Skip if too close to horizon
                ax.plot([center_x, vp2[0]], [y, vp2[1]], color=color, alpha=alpha)
                ax.plot([0, vp2[0]], [y, vp2[1]], color=color, alpha=alpha)
                
        # Draw additional vertical lines
        x_spacing = self.width / (num_lines + 1)
        for i in range(1, num_lines + 1):
            if i != int(num_lines / 2):  # Skip the center line
                x = i * x_spacing
                ax.axvline(x=x, color=color, alpha=alpha*0.7)
    
    def _draw_three_point_grid(self, num_lines, ax, color, alpha):
        """Helper method to draw a three-point perspective grid."""
        vp1, vp2, vp3 = self.vanishing_points
        
        # Draw horizontal lines (for reference)
        y_spacing = self.height / (num_lines + 1)
        for i in range(1, num_lines + 1):
            y = i * y_spacing
            if abs(y - self.horizon_height) > 0.5:  # Skip if too close to horizon
                ax.axhline(y=y, color=color, alpha=alpha*0.3, linestyle=':')
        
        # Calculate the central vertical and horizontal lines
        center_x = self.width / 2
        
        # Draw converging lines to vertical vanishing point
        # These will create the vertical edges that converge
        num_verticals = num_lines // 2
        x_spacing = self.width / (num_verticals + 1)
        for i in range(0, num_verticals + 2):
            x = i * x_spacing
            # Skip if too close to center to avoid cluttering
            if abs(x - center_x) > 0.5 or i == 0 or i == num_verticals + 1:
                # Connect bottom edge points to third vanishing point
                ax.plot([x, vp3[0]], [0, vp3[1]], color=color, alpha=alpha)
                # Connect top edge points to third vanishing point if vp3 is below
                if vp3[1] < 0:
                    ax.plot([x, vp3[0]], [self.height, vp3[1]], color=color, alpha=alpha)
        
        # For horizontal lines that converge to vp1 and vp2
        # We'll create a grid that simulates a cube or building in 3-point perspective
        levels = num_lines // 2
        for level in range(levels + 1):
            # Calculate y level (these will curve due to third vanishing point)
            y_level = level * (self.height / (levels + 1)) 
            if abs(y_level - self.horizon_height) > 0.1:
                # Connect to left vanishing point
                ax.plot([center_x, vp1[0]], [y_level, vp1[1]], color=color, alpha=alpha)
                ax.plot([self.width, vp1[0]], [y_level, vp1[1]], color=color, alpha=alpha)
                
                # Connect to right vanishing point
                ax.plot([center_x, vp2[0]], [y_level, vp2[1]], color=color, alpha=alpha)
                ax.plot([0, vp2[0]], [y_level, vp2[1]], color=color, alpha=alpha)
    
    def draw_cube_in_perspective(self, base_x, base_y, width, height, depth, ax=None, color='blue', alpha=0.5):
        """
        Draw a cube in the current perspective system.
        
        Args:
            base_x: x-coordinate of the base corner of the cube
            base_y: y-coordinate of the base corner of the cube
            width: Width of the cube in canvas units
            height: Height of the cube in canvas units
            depth: Depth of the cube in canvas units
            ax: Matplotlib axis to draw on
            color: Color of the cube
            alpha: Transparency of the cube
            
        Returns:
            Matplotlib axis with the cube drawn in perspective
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))
        
        if self.perspective_type == "one-point":
            self._draw_one_point_cube(base_x, base_y, width, height, depth, ax, color, alpha)
        elif self.perspective_type == "two-point":
            self._draw_two_point_cube(base_x, base_y, width, height, depth, ax, color, alpha)
        elif self.perspective_type == "three-point":
            self._draw_three_point_cube(base_x, base_y, width, height, depth, ax, color, alpha)
        
        # Set plot limits
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        return ax
    
    def _draw_one_point_cube(self, base_x, base_y, width, height, depth, ax, color, alpha):
        """Helper method to draw a cube in one-point perspective."""
        vp = self.vanishing_points[0]
        
        # Front face vertices
        front_vertices = [
            (base_x, base_y),                  # Bottom-left
            (base_x + width, base_y),          # Bottom-right
            (base_x + width, base_y + height), # Top-right
            (base_x, base_y + height)          # Top-left
        ]
        
        # Calculate the scale factor based on distance from vanishing point
        distance_to_vp = abs(vp[0] - (base_x + width/2))
        scale_factor = 1 - min(depth / distance_to_vp, 0.7)  # Limit for visual appeal
        
        # Calculate back face vertices (smaller due to perspective)
        back_vertices = []
        for fx, fy in front_vertices:
            # Vector from vanishing point to front vertex
            vx, vy = fx - vp[0], fy - vp[1]
            # Scale to get back vertex (depth controls how much smaller the back face is)
            bx = vp[0] + vx * scale_factor
            by = vp[1] + vy * scale_factor
            back_vertices.append((bx, by))
        
        # Draw front face (fully visible)
        front_face = Polygon(front_vertices, closed=True, fill=True, 
                             color=color, alpha=alpha, label='Front Face')
        ax.add_patch(front_face)
        
        # Connect front vertices to back vertices
        for i in range(4):
            ax.plot([front_vertices[i][0], back_vertices[i][0]], 
                    [front_vertices[i][1], back_vertices[i][1]], 
                    color=color, alpha=alpha)
        
        # Draw back face (might be partially hidden)
        back_face = Polygon(back_vertices, closed=True, fill=True, 
                            color=color, alpha=alpha*0.7, label='Back Face')
        ax.add_patch(back_face)
        
        # Determine visible side faces
        # For one-point, the left, right, top, and bottom faces may be visible
        if vp[0] > base_x + width:
            # Left face visible
            left_face = Polygon([front_vertices[0], front_vertices[3], back_vertices[3], back_vertices[0]], 
                                closed=True, fill=True, color=color, alpha=alpha*0.6)
            ax.add_patch(left_face)
        
        if vp[0] < base_x:
            # Right face visible
            right_face = Polygon([front_vertices[1], front_vertices[2], back_vertices[2], back_vertices[1]], 
                                 closed=True, fill=True, color=color, alpha=alpha*0.6)
            ax.add_patch(right_face)
        
        if vp[1] > base_y + height:
            # Bottom face visible
            bottom_face = Polygon([front_vertices[0], front_vertices[1], back_vertices[1], back_vertices[0]], 
                                  closed=True, fill=True, color=color, alpha=alpha*0.6)
            ax.add_patch(bottom_face)
        
        if vp[1] < base_y:
            # Top face visible
            top_face = Polygon([front_vertices[2], front_vertices[3], back_vertices[3], back_vertices[2]], 
                               closed=True, fill=True, color=color, alpha=alpha*0.6)
            ax.add_patch(top_face)
    
    def _draw_two_point_cube(self, base_x, base_y, width, height, depth, ax, color, alpha):
        """Helper method to draw a cube in two-point perspective."""
        vp1, vp2 = self.vanishing_points
        
        # In two-point perspective, the corner (vertical edge) is facing the viewer
        # We'll start with that corner and use the vanishing points to determine the other edges
        
        # Determine how much the edges recede based on their distance to vanishing points
        vp1_distance = abs(vp1[0] - base_x)
        vp2_distance = abs(vp2[0] - base_x)
        
        recede_factor1 = min(width / vp1_distance, 0.7)  # Limit factor for visual appeal
        recede_factor2 = min(depth / vp2_distance, 0.7)
        
        # Front vertical edge
        bottom_front = (base_x, base_y)
        top_front = (base_x, base_y + height)
        
        # Calculate horizontal edges that recede to the vanishing points
        # Bottom edges
        bottom_right = self._point_on_line(bottom_front, vp1, recede_factor1 * vp1_distance)
        bottom_left = self._point_on_line(bottom_front, vp2, recede_factor2 * vp2_distance)
        
        # Top edges
        top_right = self._point_on_line(top_front, vp1, recede_factor1 * vp1_distance)
        top_left = self._point_on_line(top_front, vp2, recede_factor2 * vp2_distance)
        
        # Back vertical edges (intersections of receding edges)
        # For simplicity, we assume these are at the same recede factors
        bottom_right_to_back = self._point_on_line(bottom_right, vp2, recede_factor2 * vp2_distance)
        bottom_left_to_back = self._point_on_line(bottom_left, vp1, recede_factor1 * vp1_distance)
        
        # We'll use the midpoint of these as our back bottom corner
        back_bottom = ((bottom_right_to_back[0] + bottom_left_to_back[0])/2, 
                       (bottom_right_to_back[1] + bottom_left_to_back[1])/2)
        
        # For top back corner, we use the same approach from the top
        top_right_to_back = self._point_on_line(top_right, vp2, recede_factor2 * vp2_distance)
        top_left_to_back = self._point_on_line(top_left, vp1, recede_factor1 * vp1_distance)
        back_top = ((top_right_to_back[0] + top_left_to_back[0])/2, 
                    (top_right_to_back[1] + top_left_to_back[1])/2)
        
        # Draw all edges
        # Vertical edges
        ax.plot([bottom_front[0], top_front[0]], [bottom_front[1], top_front[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_right[0], top_right[0]], [bottom_right[1], top_right[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_left[0], top_left[0]], [bottom_left[1], top_left[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([back_bottom[0], back_top[0]], [back_bottom[1], back_top[1]], color=color, alpha=alpha, linewidth=2)
        
        # Bottom edges
        ax.plot([bottom_front[0], bottom_right[0]], [bottom_front[1], bottom_right[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_front[0], bottom_left[0]], [bottom_front[1], bottom_left[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_right[0], back_bottom[0]], [bottom_right[1], back_bottom[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_left[0], back_bottom[0]], [bottom_left[1], back_bottom[1]], color=color, alpha=alpha, linewidth=2)
        
        # Top edges
        ax.plot([top_front[0], top_right[0]], [top_front[1], top_right[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([top_front[0], top_left[0]], [top_front[1], top_left[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([top_right[0], back_top[0]], [top_right[1], back_top[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([top_left[0], back_top[0]], [top_left[1], back_top[1]], color=color, alpha=alpha, linewidth=2)
        
        # Fill faces
        # Determine which faces are visible (simplified approach)
        faces = []
        
        # Front face (always visible in two-point)
        front_face = Polygon([bottom_front, bottom_right, top_right, top_front], 
                             closed=True, fill=True, color=color, alpha=alpha)
        ax.add_patch(front_face)
        
        # Left face
        left_face = Polygon([bottom_front, bottom_left, back_bottom, back_top, top_left, top_front], 
                            closed=True, fill=True, color=color, alpha=alpha*0.8)
        ax.add_patch(left_face)
        
        # Right face
        right_face = Polygon([bottom_front, bottom_right, back_bottom, back_top, top_right, top_front], 
                             closed=True, fill=True, color=color, alpha=alpha*0.6)
        ax.add_patch(right_face)
        
        # Top face (if viewer is below the top of the cube)
        if self.horizon_height < base_y + height:
            top_face = Polygon([top_front, top_right, back_top, top_left], 
                               closed=True, fill=True, color=color, alpha=alpha*0.7)
            ax.add_patch(top_face)
        
        # Bottom face (if viewer is above the bottom of the cube)
        if self.horizon_height > base_y:
            bottom_face = Polygon([bottom_front, bottom_right, back_bottom, bottom_left], 
                                  closed=True, fill=True, color=color, alpha=alpha*0.5)
            ax.add_patch(bottom_face)
    
    def _draw_three_point_cube(self, base_x, base_y, width, height, depth, ax, color, alpha):
        """Helper method to draw a cube in three-point perspective."""
        vp1, vp2, vp3 = self.vanishing_points
        
        # In three-point perspective, the corner (vertical edge) is facing the viewer
        # But vertical lines also converge to a third vanishing point
        
        # Determine how much the edges recede based on their distance to vanishing points
        vp1_distance = abs(vp1[0] - base_x)
        vp2_distance = abs(vp2[0] - base_x)
        vp3_distance = abs(vp3[1] - base_y)
        
        recede_factor1 = min(width / vp1_distance, 0.7)  # Limit factor for visual appeal
        recede_factor2 = min(depth / vp2_distance, 0.7)
        recede_factor3 = min(height / vp3_distance, 0.7)
        
        # Front bottom corner
        bottom_front = (base_x, base_y)
        
        # Calculate top front point (converges to vp3)
        top_front = self._point_on_line(bottom_front, vp3, recede_factor3 * vp3_distance)
        
        # Calculate horizontal edges that recede to the vanishing points
        # Bottom edges
        bottom_right = self._point_on_line(bottom_front, vp1, recede_factor1 * vp1_distance)
        bottom_left = self._point_on_line(bottom_front, vp2, recede_factor2 * vp2_distance)
        
        # Top edges
        top_right = self._point_on_line(top_front, vp1, recede_factor1 * vp1_distance)
        top_left = self._point_on_line(top_front, vp2, recede_factor2 * vp2_distance)
        
        # Back vertices (intersections of receding edges)
        # Back bottom corner
        back_bottom = self._point_on_line(bottom_right, vp2, recede_factor2 * vp2_distance)
        # Back top corner
        back_top = self._point_on_line(top_right, vp2, recede_factor2 * vp2_distance)
        
        # Draw all edges
        # Vertical edges
        ax.plot([bottom_front[0], top_front[0]], [bottom_front[1], top_front[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_right[0], top_right[0]], [bottom_right[1], top_right[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_left[0], top_left[0]], [bottom_left[1], top_left[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([back_bottom[0], back_top[0]], [back_bottom[1], back_top[1]], color=color, alpha=alpha, linewidth=2)
        
        # Bottom edges
        ax.plot([bottom_front[0], bottom_right[0]], [bottom_front[1], bottom_right[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_front[0], bottom_left[0]], [bottom_front[1], bottom_left[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_right[0], back_bottom[0]], [bottom_right[1], back_bottom[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([bottom_left[0], back_bottom[0]], [bottom_left[1], back_bottom[1]], color=color, alpha=alpha, linewidth=2)
        
        # Top edges
        ax.plot([top_front[0], top_right[0]], [top_front[1], top_right[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([top_front[0], top_left[0]], [top_front[1], top_left[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([top_right[0], back_top[0]], [top_right[1], back_top[1]], color=color, alpha=alpha, linewidth=2)
        ax.plot([top_left[0], back_top[0]], [top_left[1], back_top[1]], color=color, alpha=alpha, linewidth=2)
        
        # Fill faces
        # Determine which faces are visible (simplified approach)
        
        # Front face
        front_face = Polygon([bottom_front, bottom_right, top_right, top_front], 
                             closed=True, fill=True, color=color, alpha=alpha)
        ax.add_patch(front_face)
        
        # Left face
        left_face = Polygon([bottom_front, bottom_left, back_bottom, back_top, top_left, top_front], 
                            closed=True, fill=True, color=color, alpha=alpha*0.8)
        ax.add_patch(left_face)
        
        # Right face
        right_face = Polygon([bottom_front, bottom_right, back_bottom, back_top, top_right, top_front], 
                             closed=True, fill=True, color=color, alpha=alpha*0.6)
        ax.add_patch(right_face)
        
        # Top face
        top_face = Polygon([top_front, top_right, back_top, top_left], 
                           closed=True, fill=True, color=color, alpha=alpha*0.7)
        ax.add_patch(top_face)
        
        # Bottom face
        bottom_face = Polygon([bottom_front, bottom_right, back_bottom, bottom_left], 
                              closed=True, fill=True, color=color, alpha=alpha*0.5)
        ax.add_patch(bottom_face)
    
    def _point_on_line(self, start, end, distance):
        """
        Find a point on a line at a given distance from the start point.
        
        Args:
            start: Starting point (x, y)
            end: End point (x, y)
            distance: Distance from start
            
        Returns:
            Point (x, y) on the line
        """
        # Vector from start to end
        vx, vy = end[0] - start[0], end[1] - start[1]
        
        # Normalize the vector
        length = np.sqrt(vx**2 + vy**2)
        vx, vy = vx / length, vy / length
        
        # Find the point
        return (start[0] + vx * distance, start[1] + vy * distance)


class ProportionSystem:
    """
    A class for working with different proportion systems used in art and design.
    """
    
    def __init__(self):
        """Initialize the proportion system class."""
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # Approximately 1.618
        
    def golden_rectangle(self, width=1, height=None):
        """
        Calculate the dimensions of a golden rectangle.
        
        Args:
            width: Width of the rectangle (if height is None)
            height: Height of the rectangle (if provided, width will be calculated)
            
        Returns:
            Tuple of (width, height)
        """
        if height is None:
            # Calculate height based on width
            height = width / self.golden_ratio
        else:
            # Calculate width based on height
            width = height * self.golden_ratio
            
        return (width, height)
    
    def rule_of_thirds(self, width, height):
        """
        Calculate the division points for the rule of thirds.
        
        Args:
            width: Width of the canvas
            height: Height of the canvas
            
        Returns:
            Dictionary with horizontal and vertical division lines
        """
        return {
            'horizontal': [height / 3, 2 * height / 3],
            'vertical': [width / 3, 2 * width / 3]
        }
    
    def dynamic_symmetry(self, width, height, diagonals=True):
        """
        Calculate key points for Jay Hambidge's dynamic symmetry system.
        
        Args:
            width: Width of the canvas
            height: Height of the canvas
            diagonals: Whether to include diagonals
            
        Returns:
            Dictionary with major and minor divisions and diagonals
        """
        # Root rectangles
        root2 = np.sqrt(2)
        root3 = np.sqrt(3)
        root5 = np.sqrt(5)
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Determine the closest root rectangle
        if abs(aspect_ratio - 1) < 0.1:
            rectangle_type = "square"
        elif abs(aspect_ratio - root2) < 0.1:
            rectangle_type = "root-2"
        elif abs(aspect_ratio - root3) < 0.1:
            rectangle_type = "root-3"
        elif abs(aspect_ratio - root5) < 0.1:
            rectangle_type = "root-5"
        elif abs(aspect_ratio - self.golden_ratio) < 0.1:
            rectangle_type = "golden"
        else:
            rectangle_type = "custom"
        
        # For any rectangle, we calculate the division points
        division_points = {
            'horizontal': [height * i / 4 for i in range(1, 4)],
            'vertical': [width * i / 4 for i in range(1, 4)],
            'rectangle_type': rectangle_type
        }
        
        if diagonals:
            # Add diagonal lines
            division_points['diagonals'] = [
                [(0, 0), (width, height)],  # Bottom-left to top-right
                [(width, 0), (0, height)]   # Bottom-right to top-left
            ]
            
            # Add reciprocal diagonals (from corners to midpoints)
            division_points['reciprocals'] = [
                [(0, 0), (width, height/2)],      # Bottom-left to middle-right
                [(width, 0), (0, height/2)],      # Bottom-right to middle-left
                [(0, height), (width, height/2)], # Top-left to middle-right
                [(width, height), (0, height/2)]  # Top-right to middle-left
            ]
        
        return division_points
    
    def divine_proportion_spiral(self, center_x, center_y, max_radius, num_turns=4):
        """
        Calculate points for a golden ratio spiral.
        
        Args:
            center_x: x-coordinate of the spiral center
            center_y: y-coordinate of the spiral center
            max_radius: Maximum radius of the spiral
            num_turns: Number of spiral turns
            
        Returns:
            List of (x, y) points forming the spiral
        """
        points = []
        
        # Golden angle is derived from the golden ratio
        golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees in radians
        
        # Generate the spiral points
        for i in range(num_turns * 100):
            # Calculate the radius and angle for this point
            radius = max_radius * np.sqrt(i / (num_turns * 100))
            angle = i * golden_angle
            
            # Convert polar coordinates to cartesian
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            points.append((x, y))
        
        return points
    
    def fibonacci_grid(self, width, height, start_size=None):
        """
        Generate a grid of Fibonacci rectangles that approximates a golden spiral.
        
        Args:
            width: Width of the canvas
            height: Height of the canvas
            start_size: Starting size for the smallest square (if None, calculated based on canvas)
            
        Returns:
            List of rectangles [(x, y, width, height)]
        """
        # Fibonacci sequence
        fibonacci = [1, 1]
        for i in range(10):
            fibonacci.append(fibonacci[-1] + fibonacci[-2])
        
        # Calculate the starting size if not provided
        if start_size is None:
            # Size the grid to fit within the canvas
            total_size = fibonacci[-1]
            scale = min(width, height) / total_size
            start_size = scale
        
        # Generate the rectangles
        rectangles = []
        
        # Start in the bottom-left corner
        current_x, current_y = 0, 0
        
        # Direction of growth (0: right, 1: up, 2: left, 3: down)
        direction = 0
        
        for i in range(len(fibonacci) - 2):
            # Get the size of the current rectangle
            size = fibonacci[i] * start_size
            
            # Calculate rectangle based on direction
            if direction == 0:  # Right
                rectangle = (current_x, current_y, size, size)
                current_x += size
            elif direction == 1:  # Up
                rectangle = (current_x - size, current_y, size, size)
                current_y += size
            elif direction == 2:  # Left
                rectangle = (current_x - size, current_y - size, size, size)
                current_x -= size
            elif direction == 3:  # Down
                rectangle = (current_x, current_y - size, size, size)
                current_y -= size
            
            rectangles.append(rectangle)
            
            # Update direction (clockwise)
            direction = (direction + 1) % 4
        
        return rectangles
    
    def draw_proportional_grid(self, width, height, grid_type='golden', ax=None):
        """
        Draw a proportional grid on a matplotlib axis.
        
        Args:
            width: Width of the canvas
            height: Height of the canvas
            grid_type: Type of proportional grid ('golden', 'thirds', 'dynamic', 'fibonacci')
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib axis with the grid drawn
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(width/2, height/2))
        
        if grid_type == 'golden':
            self._draw_golden_grid(width, height, ax)
        elif grid_type == 'thirds':
            self._draw_thirds_grid(width, height, ax)
        elif grid_type == 'dynamic':
            self._draw_dynamic_grid(width, height, ax)
        elif grid_type == 'fibonacci':
            self._draw_fibonacci_grid(width, height, ax)
        
        # Set plot limits and properties
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title(f'{grid_type.capitalize()} Proportion Grid')
        
        return ax
    
    def _draw_golden_grid(self, width, height, ax):
        """Helper method to draw a golden ratio grid."""
        # Draw canvas rectangle
        ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='black'))
        
        # Draw golden ratio divisions
        # Vertical division
        golden_x = width / self.golden_ratio
        ax.axvline(x=golden_x, color='gold', linestyle='--', alpha=0.7)
        
        # Alternative vertical division
        alt_golden_x = width - golden_x
        ax.axvline(x=alt_golden_x, color='gold', linestyle=':', alpha=0.5)
        
        # Horizontal division
        golden_y = height / self.golden_ratio
        ax.axhline(y=golden_y, color='gold', linestyle='--', alpha=0.7)
        
        # Alternative horizontal division
        alt_golden_y = height - golden_y
        ax.axhline(y=alt_golden_y, color='gold', linestyle=':', alpha=0.5)
        
        # Add golden spiral
        spiral_points = self.divine_proportion_spiral(0, 0, min(width, height), num_turns=3)
        spiral_x, spiral_y = zip(*spiral_points)
        ax.plot(spiral_x, spiral_y, 'r-', alpha=0.5, label='Golden Spiral')
        
        # Add guidelines
        ax.text(golden_x + 5, 5, f'Golden Division: {golden_x:.1f}', fontsize=8)
        ax.text(5, golden_y + 5, f'Golden Division: {golden_y:.1f}', fontsize=8)
        
        ax.legend()
    
    def _draw_thirds_grid(self, width, height, ax):
        """Helper method to draw a rule of thirds grid."""
        # Draw canvas rectangle
        ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='black'))
        
        # Get rule of thirds divisions
        thirds = self.rule_of_thirds(width, height)
        
        # Draw horizontal division lines
        for y in thirds['horizontal']:
            ax.axhline(y=y, color='blue', linestyle='--', alpha=0.7)
            ax.text(5, y + 5, f'Horizontal Third: {y:.1f}', fontsize=8)
        
        # Draw vertical division lines
        for x in thirds['vertical']:
            ax.axvline(x=x, color='blue', linestyle='--', alpha=0.7)
            ax.text(x + 5, 5, f'Vertical Third: {x:.1f}', fontsize=8)
        
        # Mark the four power points (intersections)
        for x in thirds['vertical']:
            for y in thirds['horizontal']:
                ax.plot(x, y, 'ro', markersize=8, alpha=0.7)
        
        # Add label
        ax.text(width/2, height - 20, "Rule of Thirds", fontsize=12, ha='center')
    
    def _draw_dynamic_grid(self, width, height, ax):
        """Helper method to draw a dynamic symmetry grid."""
        # Draw canvas rectangle
        ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='black'))
        
        # Get dynamic symmetry divisions
        dynamic = self.dynamic_symmetry(width, height)
        
        # Draw horizontal and vertical divisions
        for y in dynamic['horizontal']:
            ax.axhline(y=y, color='green', linestyle='--', alpha=0.5)
        
        for x in dynamic['vertical']:
            ax.axvline(x=x, color='green', linestyle='--', alpha=0.5)
        
        # Draw diagonals
        if 'diagonals' in dynamic:
            for (x1, y1), (x2, y2) in dynamic['diagonals']:
                ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.5)
        
        # Draw reciprocals
        if 'reciprocals' in dynamic:
            for (x1, y1), (x2, y2) in dynamic['reciprocals']:
                ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.3)
        
        # Add information about the rectangle type
        ax.text(width/2, height - 20, f"Dynamic Symmetry - {dynamic['rectangle_type'].capitalize()} Rectangle", 
                fontsize=12, ha='center')
    
    def _draw_fibonacci_grid(self, width, height, ax):
        """Helper method to draw a Fibonacci grid."""
        # Draw canvas rectangle
        ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='black'))
        
        # Get Fibonacci rectangles
        rectangles = self.fibonacci_grid(width, height)
        
        # Draw the rectangles
        colors = plt.cm.viridis(np.linspace(0, 1, len(rectangles)))
        
        for i, (x, y, w, h) in enumerate(rectangles):
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=colors[i], linewidth=2))
            ax.text(x + w/2, y + h/2, str(i+1), fontsize=10, ha='center', va='center')
        
        # Calculate spiral points through the corner of each square
        spiral_points = []
        for rect in rectangles:
            # Add the center point of each square
            x, y, w, h = rect
            spiral_points.append((x + w/2, y + h/2))
        
        # Draw the spiral
        if spiral_points:
            spiral_x, spiral_y = zip(*spiral_points)
            ax.plot(spiral_x, spiral_y, 'r-', alpha=0.7, label='Fibonacci Spiral')
        
        ax.legend()
        ax.text(width/2, height - 20, "Fibonacci Grid", fontsize=12, ha='center')


class HumanProportions:
    """
    A class for modeling and visualizing human body proportions.
    """
    
    def __init__(self):
        """Initialize the human proportions model."""
        # Dictionary of various proportion systems
        self.proportion_systems = {
            'classic': {
                'description': 'Classic 8-head Canon (commonly used in art)',
                'head_ratio': 8,  # Body is 8 head heights
                'landmarks': {
                    'eyes': 0.5,          # Eyes at mid-head
                    'shoulders': 2.0,      # Shoulders at 2 head heights
                    'nipples': 2.5,
                    'elbows': 3.0,
                    'navel': 3.5,
                    'hips': 4.0,
                    'wrists': 4.5,
                    'groin': 4.0,
                    'fingertips': 5.0,
                    'knees': 5.5,
                    'calves': 6.5,
                    'ankles': 7.25,
                    'soles': 8.0
                }
            },
            'vitruvian': {
                'description': 'Vitruvian Man proportions by Leonardo da Vinci',
                'head_ratio': 8,
                'landmarks': {
                    'eyes': 0.5,
                    'shoulders': 2.0,
                    'nipples': 2.5,
                    'navel': 3.0,  # Navel as center of the body in Vitruvian man
                    'elbows': 3.5,
                    'hips': 4.0,
                    'wrists': 4.25,
                    'groin': 4.0,
                    'fingertips': 4.5,
                    'knees': 5.5,
                    'calves': 6.5,
                    'ankles': 7.25,
                    'soles': 8.0
                },
                # Special proportions for the Vitruvian man
                'special': {
                    'height_equals_armspan': True,  # Height equals armspan
                    'navel_to_soles_golden_ratio': True  # Distance from navel to soles forms golden ratio with height
                }
            },
            'realistic': {
                'description': 'Realistic average adult proportions',
                'head_ratio': 7.5,  # Slightly shorter than the idealized classic canon
                'landmarks': {
                    'eyes': 0.5,
                    'shoulders': 1.9,
                    'nipples': 2.4,
                    'elbows': 3.0,
                    'navel': 3.3,
                    'hips': 3.8,
                    'wrists': 4.3,
                    'groin': 3.8,
                    'fingertips': 4.7,
                    'knees': 5.3,
                    'calves': 6.3,
                    'ankles': 7.0,
                    'soles': 7.5
                }
            },
            'child': {
                'description': 'Child proportions (around 6-8 years old)',
                'head_ratio': 6,  # Children have larger heads relative to body
                'landmarks': {
                    'eyes': 0.5,
                    'shoulders': 1.8,
                    'nipples': 2.2,
                    'elbows': 2.7,
                    'navel': 3.0,
                    'hips': 3.3,
                    'wrists': 3.8,
                    'groin': 3.3,
                    'fingertips': 4.2,
                    'knees': 4.5,
                    'calves': 5.2,
                    'ankles': 5.7,
                    'soles': 6.0
                }
            },
            'heroic': {
                'description': 'Heroic/Idealized proportions (Greek statues, superheroes)',
                'head_ratio': 8.5,  # Exaggerated height for heroic appearance
                'landmarks': {
                    'eyes': 0.5,
                    'shoulders': 2.1,      # Broader shoulders
                    'nipples': 2.6,
                    'elbows': 3.2,
                    'navel': 3.6,
                    'hips': 4.2,
                    'wrists': 4.6,
                    'groin': 4.2,
                    'fingertips': 5.0,
                    'knees': 5.7,
                    'calves': 6.8,
                    'ankles': 7.5,
                    'soles': 8.5
                }
            }
        }
        
        # Golden ratio used in some proportion systems
        self.golden_ratio = (1 + np.sqrt(5)) / 2
    
    def calculate_proportions(self, height, system='classic'):
        """
        Calculate actual measurements for a given height and proportion system.
        
        Args:
            height: Total height in any unit (cm, inches, etc.)
            system: Proportion system to use
            
        Returns:
            Dictionary with measurements for each body landmark
        """
        if system not in self.proportion_systems:
            raise ValueError(f"Unknown proportion system: {system}")
        
        proportion_data = self.proportion_systems[system]
        head_ratio = proportion_data['head_ratio']
        landmarks = proportion_data['landmarks']
        
        # Calculate head height
        head_height = height / head_ratio
        
        # Calculate measurements for each landmark
        measurements = {}
        for landmark, ratio in landmarks.items():
            measurements[landmark] = ratio * head_height
        
        # Add total height
        measurements['total_height'] = height
        measurements['head_height'] = head_height
        
        return measurements
    
    def draw_figure(self, height, system='classic', gender='neutral', pose='standard', ax=None):
        """
        Draw a human figure based on the specified proportion system.
        
        Args:
            height: Total height in any unit
            system: Proportion system to use
            gender: 'male', 'female', or 'neutral'
            pose: Pose of the figure ('standard', 'arms_up', etc.)
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib axis with the human figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 10))
        
        # Calculate proportions
        proportions = self.calculate_proportions(height, system)
        
        # Define the proportional width based on gender
        if gender == 'male':
            shoulder_width = proportions['head_height'] * 2.5
            hip_width = proportions['head_height'] * 1.8
        elif gender == 'female':
            shoulder_width = proportions['head_height'] * 2.0
            hip_width = proportions['head_height'] * 2.0
        else:  # neutral
            shoulder_width = proportions['head_height'] * 2.2
            hip_width = proportions['head_height'] * 1.9
        
        # Adjust other widths
        head_width = proportions['head_height'] * 0.75
        neck_width = head_width * 0.4
        waist_width = hip_width * 0.8
        
        # Draw figure centered at x=0
        self._draw_figure_elements(proportions, shoulder_width, hip_width, head_width, 
                                   neck_width, waist_width, gender, pose, ax)
        
        # Draw measurement lines on the left side
        self._draw_measurement_lines(proportions, ax)
        
        # Set plot properties
        ax.set_xlim(-shoulder_width, shoulder_width)
        ax.set_ylim(0, height * 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title(f'Human Proportions - {system.capitalize()} System')
        
        return ax
    
    def _draw_figure_elements(self, proportions, shoulder_width, hip_width, head_width, 
                              neck_width, waist_width, gender, pose, ax):
        """Helper method to draw the elements of the human figure."""
        # Head
        head_height = proportions['head_height']
        head_center = head_width / 2
        head = Circle((0, height - head_height/2), head_width/2, fill=False, edgecolor='black')
        ax.add_patch(head)
        
        # Eyes
        eye_y = height - proportions['eyes'] * head_height
        eye_spacing = head_width * 0.3
        left_eye = Circle((-eye_spacing/2, eye_y), head_width*0.05, fill=True, edgecolor='black')
        right_eye = Circle((eye_spacing/2, eye_y), head_width*0.05, fill=True, edgecolor='black')
        ax.add_patch(left_eye)
        ax.add_patch(right_eye)
        
        # Neck
        neck_top = height - head_height
        neck_bottom = height - proportions['shoulders']
        neck_line = np.array([
            [-neck_width/2, neck_top],
            [-neck_width/2, neck_bottom],
            [neck_width/2, neck_bottom],
            [neck_width/2, neck_top]
        ])
        ax.add_patch(Polygon(neck_line, closed=True, fill=False, edgecolor='black'))
        
        # Torso outline
        shoulder_y = height - proportions['shoulders']
        nipple_y = height - proportions['nipples']
        navel_y = height - proportions['navel']
        hip_y = height - proportions['hips']
        
        # Draw torso based on gender
        if gender == 'male':
            torso_line = np.array([
                [-shoulder_width/2, shoulder_y],  # Left shoulder
                [-waist_width/2, navel_y],        # Left waist
                [-hip_width/2, hip_y],            # Left hip
                [hip_width/2, hip_y],             # Right hip
                [waist_width/2, navel_y],         # Right waist
                [shoulder_width/2, shoulder_y]    # Right shoulder
            ])
        elif gender == 'female':
            breast_width = shoulder_width * 0.8
            breast_y = height - proportions['nipples']
            torso_line = np.array([
                [-shoulder_width/2, shoulder_y],  # Left shoulder
                [-breast_width/2, breast_y],      # Left breast
                [-waist_width/2, navel_y],        # Left waist
                [-hip_width/2, hip_y],            # Left hip
                [hip_width/2, hip_y],             # Right hip
                [waist_width/2, navel_y],         # Right waist
                [breast_width/2, breast_y],       # Right breast
                [shoulder_width/2, shoulder_y]    # Right shoulder
            ])
        else:  # neutral
            torso_line = np.array([
                [-shoulder_width/2, shoulder_y],  # Left shoulder
                [-waist_width/2, navel_y],        # Left waist
                [-hip_width/2, hip_y],            # Left hip
                [hip_width/2, hip_y],             # Right hip
                [waist_width/2, navel_y],         # Right waist
                [shoulder_width/2, shoulder_y]    # Right shoulder
            ])
        
        ax.add_patch(Polygon(torso_line, closed=True, fill=False, edgecolor='black'))
        
        # Arms
        elbow_y = height - proportions['elbows']
        wrist_y = height - proportions['wrists']
        fingertip_y = height - proportions['fingertips']
        
        # Adjust arm position based on pose
        if pose == 'standard':
            # Standard pose (arms at sides)
            # Left arm
            ax.plot([-shoulder_width/2, -shoulder_width/2, -shoulder_width/2 - head_width/2, -shoulder_width/2 - head_width/3],
                    [shoulder_y, elbow_y, wrist_y, fingertip_y], 'k-')
            # Right arm
            ax.plot([shoulder_width/2, shoulder_width/2, shoulder_width/2 + head_width/2, shoulder_width/2 + head_width/3],
                    [shoulder_y, elbow_y, wrist_y, fingertip_y], 'k-')
        elif pose == 'arms_up':
            # Arms raised up
            # Left arm
            ax.plot([-shoulder_width/2, -shoulder_width/2 - head_width, -shoulder_width/2 - head_width*1.5],
                    [shoulder_y, shoulder_y + head_height*2, shoulder_y + head_height*3], 'k-')
            # Right arm
            ax.plot([shoulder_width/2, shoulder_width/2 + head_width, shoulder_width/2 + head_width*1.5],
                    [shoulder_y, shoulder_y + head_height*2, shoulder_y + head_height*3], 'k-')
        else:
            # Default to standard pose
            # Left arm
            ax.plot([-shoulder_width/2, -shoulder_width/2, -shoulder_width/2 - head_width/2, -shoulder_width/2 - head_width/3],
                    [shoulder_y, elbow_y, wrist_y, fingertip_y], 'k-')
            # Right arm
            ax.plot([shoulder_width/2, shoulder_width/2, shoulder_width/2 + head_width/2, shoulder_width/2 + head_width/3],
                    [shoulder_y, elbow_y, wrist_y, fingertip_y], 'k-')
        
        # Legs
        knee_y = height - proportions['knees']
        ankle_y = height - proportions['ankles']
        sole_y = height - proportions['soles']
        
        # Left leg
        ax.plot([-hip_width/4, -hip_width/4, -hip_width/4, -hip_width/4],
                [hip_y, knee_y, ankle_y, sole_y], 'k-')
        # Right leg
        ax.plot([hip_width/4, hip_width/4, hip_width/4, hip_width/4],
                [hip_y, knee_y, ankle_y, sole_y], 'k-')
        
        # Feet
        foot_length = head_width * 0.8
        # Left foot
        ax.plot([-hip_width/4, -hip_width/4 + foot_length], [sole_y, sole_y], 'k-')
        # Right foot
        ax.plot([hip_width/4, hip_width/4 + foot_length], [sole_y, sole_y], 'k-')
    
    def _draw_measurement_lines(self, proportions, ax):
        """Helper method to draw measurement lines and labels."""
        height = proportions['total_height']
        head_height = proportions['head_height']
        offset = -head_height * 2  # Offset for measurement lines
        
        # Draw vertical line for measurements
        ax.plot([offset, offset], [0, height], 'k-', linewidth=1)
        
        # Draw horizontal lines and labels for key landmarks
        for landmark, value in proportions.items():
            if landmark not in ['total_height', 'head_height']:
                y = height - value
                ax.plot([offset, offset - head_height/2], [y, y], 'k-', linewidth=1)
                ax.text(offset - head_height/2 - 0.1, y, f"{landmark}: {value:.2f}", 
                        fontsize=8, ha='right', va='center')
        
        # Add head height label
        ax.text(offset - head_height/2 - 0.1, height - head_height/2, 
                f"Head Height: {head_height:.2f}", fontsize=8, ha='right', va='center')
        
        # Add total height label
        ax.text(offset - head_height/2 - 0.1, height/2, 
                f"Total Height: {height:.2f}", fontsize=10, ha='right', va='center', 
                rotation=90, weight='bold')
    
    def compare_proportion_systems(self, height, ax=None):
        """
        Draw multiple proportion systems side by side for comparison.
        
        Args:
            height: Total height in any unit
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib figure with the comparison
        """
        systems = list(self.proportion_systems.keys())
        
        if ax is None:
            fig, axes = plt.subplots(1, len(systems), figsize=(4 * len(systems), 10))
        else:
            axes = [ax] * len(systems)
            fig = ax.figure
        
        for i, system in enumerate(systems):
            self.draw_figure(height, system=system, gender='neutral', pose='standard', ax=axes[i])
            axes[i].set_title(f"{system.capitalize()}\n{self.proportion_systems[system]['description']}")
            
            # Add head count lines
            self._add_head_count_lines(height, system, axes[i])
        
        plt.tight_layout()
        fig.suptitle(f"Comparison of Human Proportion Systems (Height: {height} units)", 
                     fontsize=16, y=1.02)
        
        return fig
    
    def _add_head_count_lines(self, height, system, ax):
        """Helper method to add head count reference lines."""
        proportions = self.calculate_proportions(height, system)
        head_height = proportions['head_height']
        head_ratio = self.proportion_systems[system]['head_ratio']
        
        # Draw horizontal lines for each head height
        for i in range(1, int(head_ratio) + 1):
            y = height - i * head_height
            ax.axhline(y=y, color='red', linestyle=':', alpha=0.5)
            
            # Add head count label
            ax.text(-head_height, y, f"{i}", fontsize=8, color='red', 
                    ha='center', va='center')
    
    def draw_vitruvian_man(self, height=10, ax=None):
        """
        Draw a representation of Leonardo da Vinci's Vitruvian Man.
        
        Args:
            height: Total height in any unit
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib axis with the Vitruvian Man drawing
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Calculate proportions using the Vitruvian system
        proportions = self.calculate_proportions(height, 'vitruvian')
        head_height = proportions['head_height']
        
        # Draw the circle and square
        circle_radius = height / 2
        square_size = height
        
        circle = Circle((0, height/2), circle_radius, fill=False, edgecolor='blue', alpha=0.5)
        square = Rectangle((-height/2, 0), square_size, square_size, fill=False, edgecolor='green', alpha=0.5)
        
        ax.add_patch(circle)
        ax.add_patch(square)
        
        # Draw the figure in standard pose
        self.draw_figure(height, system='vitruvian', gender='male', pose='standard', ax=ax)
        
        # Draw the figure in spread-eagle pose (arms and legs spread)
        shoulder_width = head_height * 2.5
        hip_width = head_height * 1.8
        spread_limb_length = height / 2  # Arms and legs fully extended
        
        # Center of the figure
        center_x, center_y = 0, height/2
        
        # Draw spread arms and legs as lines
        # Arms
        ax.plot([center_x, center_x - spread_limb_length], [center_y, center_y], 'k-', alpha=0.7)
        ax.plot([center_x, center_x + spread_limb_length], [center_y, center_y], 'k-', alpha=0.7)
        
        # Legs
        groin_y = height - proportions['groin']
        leg_angle = 45 * np.pi / 180  # 45 degrees in radians
        leg_length = height / 2.5
        
        ax.plot([center_x, center_x - leg_length * np.cos(leg_angle)], 
                [groin_y, groin_y - leg_length * np.sin(leg_angle)], 'k-', alpha=0.7)
        ax.plot([center_x, center_x + leg_length * np.cos(leg_angle)], 
                [groin_y, groin_y - leg_length * np.sin(leg_angle)], 'k-', alpha=0.7)
        
        # Add title and information
        ax.set_title("The Vitruvian Man\nBased on Leonardo da Vinci's Drawing", fontsize=14)
        
        # Add explanatory text
        description = (
            "The Vitruvian Man demonstrates ideal human proportions:\n"
            "- The height equals the arm span (square)\n"
            "- The navel is at the center of the circle\n"
            "- Body height is 8 head heights\n"
            "- Extended arms and legs touch the circle\n"
            "- The body fits perfectly in both circle and square"
        )
        
        ax.text(0, -height*0.05, description, fontsize=10, ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Set plot properties
        ax.set_xlim(-height*0.75, height*0.75)
        ax.set_ylim(-height*0.1, height*1.1)
        ax.set_aspect('equal')
        
        return ax


class ArchitecturalProportions:
    """
    A class for modeling and visualizing architectural proportions.
    """
    
    def __init__(self):
        """Initialize the architectural proportions model."""
        # Golden ratio
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Dictionary of architectural proportion systems
        self.proportion_systems = {
            'classical': {
                'description': 'Classical Greek and Roman Proportions',
                'column_proportions': {
                    'doric': {'diameter_to_height': 1/6.5, 'capital_height': 1/10},
                    'ionic': {'diameter_to_height': 1/9, 'capital_height': 1/9},
                    'corinthian': {'diameter_to_height': 1/10, 'capital_height': 1/8}
                },
                'entablature_proportions': {
                    'architrave': 1/10,  # Fraction of total height
                    'frieze': 1/10,
                    'cornice': 1/8
                }
            },
            'renaissance': {
                'description': 'Renaissance Proportions (Palladio)',
                'room_proportions': [
                    {'width': 1, 'length': 1},                      # Square
                    {'width': 1, 'length': 4/3},                    # 3:4
                    {'width': 1, 'length': 3/2},                    # 2:3
                    {'width': 1, 'length': self.golden_ratio},      # Golden ratio
                    {'width': 1, 'length': 2}                       # 1:2
                ],
                'elevation_proportions': {
                    'base': 1/8,         # Fraction of total height
                    'first_story': 3/8,
                    'second_story': 2/8,
                    'attic': 1/8
                }
            },
            'modernist': {
                'description': 'Modernist Proportions (Le Corbusier\'s Modulor)',
                'blue_series': [0.13, 0.21, 0.34, 0.55, 0.89, 1.44, 2.33, 3.77],  # Modulor blue series
                'red_series': [0.15, 0.25, 0.41, 0.66, 1.07, 1.73, 2.80, 4.53],   # Modulor red series
                'key_dimensions': {
                    'ceiling_height': 2.26,  # Typical Modulor ceiling height (m)
                    'door_height': 2.20,     # Typical door height (m)
                    'doorknob_height': 1.13,  # Typical doorknob height (m)
                    'table_height': 0.70,    # Typical table height (m)
                    'chair_height': 0.43     # Typical chair height (m)
                }
            },
            'japanese': {
                'description': 'Traditional Japanese Proportions',
                'tatami_size': {'width': 0.9, 'length': 1.8},  # Standard tatami size (m)
                'room_sizes': [
                    {'name': '4.5 mats', 'size': 4.5},
                    {'name': '6 mats', 'size': 6},
                    {'name': '8 mats', 'size': 8},
                    {'name': '10 mats', 'size': 10},
                    {'name': '12 mats', 'size': 12}
                ],
                'elements': {
                    'ceiling_height': 2.4,  # Typical ceiling height (m)
                    'door_height': 1.8,     # Typical door height (m)
                    'tatami_thickness': 0.05,  # Tatami thickness (m)
                    'tokonoma_width': 2     # Typical tokonoma (alcove) width (m)
                }
            }
        }
    
    def draw_classical_orders(self, height=10, ax=None):
        """
        Draw the classical architectural orders with proper proportions.
        
        Args:
            height: Total height of the column and entablature
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib figure with the classical orders
        """
        orders = ['doric', 'ionic', 'corinthian']
        
        if ax is None:
            fig, axes = plt.subplots(1, len(orders), figsize=(15, 10))
        else:
            axes = [ax] * len(orders)
            fig = ax.figure
        
        classical = self.proportion_systems['classical']
        
        for i, order in enumerate(orders):
            # Get proportions for this order
            column_prop = classical['column_proportions'][order]
            entablature_prop = classical['entablature_proportions']
            
            # Calculate dimensions
            diameter = height * column_prop['diameter_to_height']
            
            # Calculate entablature heights
            entablature_height = height * 0.2  # Entablature is typically 1/5 of total height
            column_height = height - entablature_height
            
            architrave_height = entablature_height * entablature_prop['architrave'] / sum(entablature_prop.values())
            frieze_height = entablature_height * entablature_prop['frieze'] / sum(entablature_prop.values())
            cornice_height = entablature_height * entablature_prop['cornice'] / sum(entablature_prop.values())
            
            # Calculate capital height
            capital_height = column_height * column_prop['capital_height']
            shaft_height = column_height - capital_height
            
            # Base width is slightly larger than diameter
            base_width = diameter * 1.2
            
            # Draw the column and entablature
            self._draw_column(axes[i], order, diameter, base_width, 
                              column_height, capital_height, shaft_height,
                              architrave_height, frieze_height, cornice_height)
            
            # Set plot properties
            axes[i].set_xlim(-diameter*4, diameter*4)
            axes[i].set_ylim(0, height * 1.1)
            axes[i].set_aspect('equal')
            axes[i].set_title(f"{order.capitalize()} Order")
            
            # Remove tick labels for cleaner look
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        plt.tight_layout()
        fig.suptitle("Classical Architectural Orders", fontsize=16, y=1.02)
        
        return fig
    
    def _draw_column(self, ax, order, diameter, base_width, column_height, 
                     capital_height, shaft_height, architrave_height, 
                     frieze_height, cornice_height):
        """Helper method to draw a classical column with proper proportions."""
        # Draw the base/plinth
        plinth_height = diameter * 0.5
        base = Rectangle((-base_width/2, 0), base_width, plinth_height, 
                         fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(base)
        
        # Draw the shaft
        # Columns have entasis (slight bulge in the middle)
        entasis_factor = 0.02  # 2% bulge
        num_points = 30
        heights = np.linspace(0, shaft_height, num_points)
        width_factors = 1 - entasis_factor * np.sin(np.pi * heights / shaft_height)
        
        # Left side of shaft
        shaft_left_x = -diameter/2 * width_factors
        shaft_left_y = heights + plinth_height
        
        # Right side of shaft
        shaft_right_x = diameter/2 * width_factors
        shaft_right_y = heights + plinth_height
        
        # Create shaft polygon
        shaft_x = np.concatenate([shaft_left_x, shaft_right_x[::-1]])
        shaft_y = np.concatenate([shaft_left_y, shaft_right_y[::-1]])
        
        shaft = plt.Polygon(np.column_stack([shaft_x, shaft_y]), 
                            closed=True, fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(shaft)
        
        # Draw fluting (vertical grooves) if not Doric
        if order != 'doric':
            self._draw_fluting(ax, order, diameter, plinth_height, shaft_height)
        
        # Draw the capital
        capital_bottom_y = plinth_height + shaft_height
        capital_top_y = capital_bottom_y + capital_height
        
        if order == 'doric':
            self._draw_doric_capital(ax, diameter, capital_bottom_y, capital_height)
        elif order == 'ionic':
            self._draw_ionic_capital(ax, diameter, capital_bottom_y, capital_height)
        else:  # corinthian
            self._draw_corinthian_capital(ax, diameter, capital_bottom_y, capital_height)
        
        # Draw the entablature
        entablature_bottom_y = capital_top_y
        entablature_width = diameter * 2
        
        # Architrave
        architrave = Rectangle((-entablature_width/2, entablature_bottom_y), 
                              entablature_width, architrave_height, 
                              fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(architrave)
        
        # Frieze
        frieze_bottom_y = entablature_bottom_y + architrave_height
        frieze = Rectangle((-entablature_width/2, frieze_bottom_y), 
                          entablature_width, frieze_height, 
                          fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(frieze)
        
        # Add decorative elements to frieze if Doric
        if order == 'doric':
            self._draw_doric_frieze(ax, diameter, frieze_bottom_y, frieze_height)
        
        # Cornice
        cornice_bottom_y = frieze_bottom_y + frieze_height
        cornice_width = entablature_width * 1.2  # Cornice projects further
        cornice = Rectangle((-cornice_width/2, cornice_bottom_y), 
                           cornice_width, cornice_height, 
                           fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(cornice)
    
    def _draw_fluting(self, ax, order, diameter, base_y, shaft_height):
        """Helper method to draw column fluting (vertical grooves)."""
        num_flutes = 24 if order == 'ionic' else 20
        
        for i in range(num_flutes):
            angle = 2 * np.pi * i / num_flutes
            x = diameter/2 * 0.9 * np.cos(angle)
            
            # Only draw flutes visible from front
            if x > -diameter/2 and x < diameter/2:
                ax.plot([x, x], [base_y, base_y + shaft_height], 'k-', linewidth=0.5, alpha=0.5)
    
    def _draw_doric_capital(self, ax, diameter, bottom_y, height):
        """Helper method to draw a Doric capital."""
        # Echinus (circular cushion)
        echinus_height = height * 0.6
        echinus_width = diameter * 1.2
        
        # Draw as a trapezoid approximating the curve
        echinus_x = [-diameter/2, -echinus_width/2, echinus_width/2, diameter/2]
        echinus_y = [bottom_y, bottom_y + echinus_height, bottom_y + echinus_height, bottom_y]
        
        echinus = plt.Polygon(np.column_stack([echinus_x, echinus_y]), 
                             closed=True, fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(echinus)
        
        # Abacus (square slab on top)
        abacus_height = height * 0.4
        abacus_width = diameter * 1.5
        
        abacus = Rectangle((-abacus_width/2, bottom_y + echinus_height), 
                          abacus_width, abacus_height, 
                          fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(abacus)
    
    def _draw_ionic_capital(self, ax, diameter, bottom_y, height):
        """Helper method to draw an Ionic capital."""
        # Echinus with egg-and-dart pattern
        echinus_height = height * 0.3
        echinus_width = diameter * 1.2
        
        echinus = Rectangle((-echinus_width/2, bottom_y), 
                           echinus_width, echinus_height, 
                           fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(echinus)
        
        # Volutes (spiral scrolls)
        volute_width = diameter * 0.8
        volute_height = height * 0.7
        
        # Left volute
        left_volute_center = (-diameter/2, bottom_y + echinus_height + volute_height/2)
        self._draw_spiral(ax, left_volute_center, volute_width/2, volute_height/2, 2.5, True)
        
        # Right volute
        right_volute_center = (diameter/2, bottom_y + echinus_height + volute_height/2)
        self._draw_spiral(ax, right_volute_center, volute_width/2, volute_height/2, 2.5, False)
        
        # Abacus (thin slab on top)
        abacus_height = height * 0.1
        abacus_width = diameter * 1.5
        
        abacus = Rectangle((-abacus_width/2, bottom_y + echinus_height + volute_height), 
                          abacus_width, abacus_height, 
                          fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(abacus)
    
    def _draw_corinthian_capital(self, ax, diameter, bottom_y, height):
        """Helper method to draw a Corinthian capital."""
        # Base of the capital
        base_width = diameter * 1.1
        
        # Capital widens toward the top (bell shape)
        top_width = diameter * 1.8
        
        # Create bell shape
        bell_x = [-base_width/2, -top_width/2, top_width/2, base_width/2]
        bell_y = [bottom_y, bottom_y + height*0.9, bottom_y + height*0.9, bottom_y]
        
        bell = plt.Polygon(np.column_stack([bell_x, bell_y]), 
                          closed=True, fill=True, edgecolor='black', facecolor='lightgray', alpha=0.5)
        ax.add_patch(bell)
        
        # Acanthus leaves (simplified representation)
        # First row of leaves
        for x_pos in np.linspace(-diameter/2, diameter/2, 5):
            leaf_height = height * 0.4
            leaf_width = diameter * 0.3
            
            leaf_x = [x_pos - leaf_width/2, x_pos, x_pos + leaf_width/2]
            leaf_y = [bottom_y, bottom_y + leaf_height, bottom_y]
            
            leaf = plt.Polygon(np.column_stack([leaf_x, leaf_y]), 
                              closed=True, fill=False, edgecolor='black')
            ax.add_patch(leaf)
        
        # Second row of leaves (higher and offset)
        for x_pos in np.linspace(-diameter/2 + diameter*0.15, diameter/2 - diameter*0.15, 3):
            leaf_height = height * 0.6
            leaf_width = diameter * 0.3
            
            leaf_x = [x_pos - leaf_width/2, x_pos, x_pos + leaf_width/2]
            leaf_y = [bottom_y + height*0.2, bottom_y + height*0.2 + leaf_height, bottom_y + height*0.2]
            
            leaf = plt.Polygon(np.column_stack([leaf_x, leaf_y]), 
                              closed=True, fill=False, edgecolor='black')
            ax.add_patch(leaf)
        
        # Volutes at top corners
        volute_size = diameter * 0.2
        
        # Left volute
        left_volute_center = (-top_width/2 + volute_size/2, bottom_y + height*0.85)
        self._draw_spiral(ax, left_volute_center, volute_size, volute_size, 2, True)
        
        # Right volute
        right_volute_center = (top_width/2 - volute_size/2, bottom_y + height*0.85)
        self._draw_spiral(ax, right_volute_center, volute_size, volute_size, 2, False)
        
        # Abacus
        abacus_height = height * 0.1
        abacus_width = top_width * 1.1
        
        abacus = Rectangle((-abacus_width/2, bottom_y + height*0.9), 
                          abacus_width, abacus_height, 
                          fill=True, edgecolor='black', facecolor='lightgray')
        ax.add_patch(abacus)
    
    def _draw_doric_frieze(self, ax, diameter, bottom_y, height):
        """Helper method to draw triglyph and metope pattern on a Doric frieze."""
        # Define proportions
        frieze_width = diameter * 2
        triglyph_width = frieze_width / 8
        metope_width = frieze_width / 4
        
        # Draw triglyphs (vertical channeled blocks)
        for x_pos in [-frieze_width/4, frieze_width/4]:
            # Triglyph background
            triglyph = Rectangle((x_pos - triglyph_width/2, bottom_y), 
                               triglyph_width, height, 
                               fill=True, edgecolor='black', facecolor='darkgray')
            ax.add_patch(triglyph)
            
            # Vertical grooves
            for offset in [-triglyph_width/4, 0, triglyph_width/4]:
                ax.plot([x_pos + offset, x_pos + offset], 
                        [bottom_y, bottom_y + height], 'k-', linewidth=1.5)
    
    def _draw_spiral(self, ax, center, width, height, turns, clockwise=True):
        """Helper method to draw a spiral (for Ionic and Corinthian volutes)."""
        t = np.linspace(0, turns * 2 * np.pi, 100)
        
        # Create spiral with decreasing radius
        radius = np.exp(-0.2 * t) * width
        
        if clockwise:
            x = center[0] + radius * np.cos(t)
            y = center[1] + radius * np.sin(t)
        else:
            x = center[0] + radius * np.cos(-t)
            y = center[1] + radius * np.sin(-t)
        
        ax.plot(x, y, 'k-', linewidth=1.5)
    
    def draw_renaissance_proportions(self, width=10, ax=None):
        """
        Draw Palladian villa elevations and plans with Renaissance proportions.
        
        Args:
            width: Width of the villa in any unit
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib figure with Renaissance proportions
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        else:
            ax1 = ax
            ax2 = ax
            fig = ax.figure
        
        # Get Renaissance proportions
        renaissance = self.proportion_systems['renaissance']
        
        # Draw the villa elevation (front view)
        self._draw_renaissance_elevation(ax1, width, renaissance['elevation_proportions'])
        
        # Draw the villa plan (top view)
        self._draw_renaissance_plan(ax2, width, renaissance['room_proportions'])
        
        # Set titles
        ax1.set_title("Renaissance Villa Elevation", fontsize=14)
        ax2.set_title("Renaissance Villa Plan", fontsize=14)
        
        # Add information about proportions
        self._add_renaissance_info(fig)
        
        return fig
    
    def _draw_renaissance_elevation(self, ax, width, proportions):
        """Helper method to draw a Renaissance elevation with proper proportions."""
        # Calculate heights
        base_height = width * proportions['base']
        first_story_height = width * proportions['first_story']
        second_story_height = width * proportions['second_story']
        attic_height = width * proportions['attic']
        
        total_height = base_height + first_story_height + second_story_height + attic_height
        
        # Draw the main building body
        main_building = Rectangle((0, 0), width, total_height, 
                                 fill=True, edgecolor='black', facecolor='bisque')
        ax.add_patch(main_building)
        
        # Draw base/foundation
        ax.axhline(y=base_height, color='black', linestyle='-', linewidth=1)
        
        # Draw first story details
        first_story_bottom = base_height
        first_story_top = first_story_bottom + first_story_height
        
        # Central door
        door_width = width / 5
        door_height = first_story_height * 0.8
        door = Rectangle((width/2 - door_width/2, first_story_bottom), 
                         door_width, door_height, 
                         fill=True, edgecolor='black', facecolor='saddlebrown')
        ax.add_patch(door)
        
        # Windows
        window_width = width / 8
        window_height = first_story_height / 2
        window_y = first_story_bottom + first_story_height/2 - window_height/2
        
        for x_pos in [width/4 - window_width/2, width*3/4 - window_width/2]:
            window = Rectangle((x_pos, window_y), 
                              window_width, window_height, 
                              fill=True, edgecolor='black', facecolor='lightblue')
            ax.add_patch(window)
        
        # Draw floor division between first and second story
        ax.axhline(y=first_story_top, color='black', linestyle='-', linewidth=1)
        
        # Draw second story details
        second_story_bottom = first_story_top
        second_story_top = second_story_bottom + second_story_height
        
        # Second story windows
        s_window_width = width / 8
        s_window_height = second_story_height / 2
        s_window_y = second_story_bottom + second_story_height/2 - s_window_height/2
        
        for x_pos in [width/4 - s_window_width/2, width/2 - s_window_width/2, width*3/4 - s_window_width/2]:
            window = Rectangle((x_pos, s_window_y), 
                              window_width, s_window_height, 
                              fill=True, edgecolor='black', facecolor='lightblue')
            ax.add_patch(window)
        
        # Draw floor division between second story and attic
        ax.axhline(y=second_story_top, color='black', linestyle='-', linewidth=1)
        
        # Draw attic details
        attic_bottom = second_story_top
        
        # Attic windows (smaller)
        a_window_width = width / 12
        a_window_height = attic_height / 2
        a_window_y = attic_bottom + attic_height/2 - a_window_height/2
        
        for x_pos in [width/4 - a_window_width/2, width/2 - a_window_width/2, width*3/4 - a_window_width/2]:
            window = Rectangle((x_pos, a_window_y), 
                              a_window_width, a_window_height, 
                              fill=True, edgecolor='black', facecolor='lightblue')
            ax.add_patch(window)
        
        # Draw the roof
        roof_height = width / 6
        roof = plt.Polygon([[0, total_height], [width, total_height], [width/2, total_height + roof_height]], 
                          closed=True, fill=True, edgecolor='black', facecolor='brown')
        ax.add_patch(roof)
        
        # Add dimension lines and labels
        # Base height
        ax.annotate("", xy=(width*1.1, 0), xytext=(width*1.1, base_height),
                   arrowprops=dict(arrowstyle="<->", color='red'))
        ax.text(width*1.15, base_height/2, f"Base\n1/8", color='red', ha='left', va='center')
        
        # First story height
        ax.annotate("", xy=(width*1.1, base_height), xytext=(width*1.1, first_story_top),
                   arrowprops=dict(arrowstyle="<->", color='red'))
        ax.text(width*1.15, base_height + first_story_height/2, f"First Story\n3/8", color='red', ha='left', va='center')
        
        # Second story height
        ax.annotate("", xy=(width*1.1, first_story_top), xytext=(width*1.1, second_story_top),
                   arrowprops=dict(arrowstyle="<->", color='red'))
        ax.text(width*1.15, first_story_top + second_story_height/2, f"Second Story\n2/8", color='red', ha='left', va='center')
        
        # Attic height
        ax.annotate("", xy=(width*1.1, second_story_top), xytext=(width*1.1, total_height),
                   arrowprops=dict(arrowstyle="<->", color='red'))
        ax.text(width*1.15, second_story_top + attic_height/2, f"Attic\n1/8", color='red', ha='left', va='center')
        
        # Total height
        ax.annotate("", xy=(width*1.3, 0), xytext=(width*1.3, total_height),
                   arrowprops=dict(arrowstyle="<->", color='black', linewidth=2))
        ax.text(width*1.35, total_height/2, f"Total Height\n= Width", color='black', ha='left', va='center', fontweight='bold')
        
        # Set plot properties
        ax.set_xlim(-width*0.1, width*1.5)
        ax.set_ylim(-width*0.1, total_height + roof_height + width*0.1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_renaissance_plan(self, ax, width, room_proportions):
        """Helper method to draw a Renaissance plan with proper proportions."""
        # Total width equals the elevation width
        # Create a layout of rooms based on the Palladian style
        
        # Total building depth
        depth = width * self.golden_ratio  # Using golden ratio for overall proportion
        
        # Main building outline
        main_building = Rectangle((0, 0), width, depth, 
                                 fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(main_building)
        
        # Create a grid of rooms (simplified Palladian layout)
        grid_width = 3
        grid_height = 3
        
        cell_width = width / grid_width
        cell_height = depth / grid_height
        
        # Draw the grid
        for i in range(grid_width + 1):
            x = i * cell_width
            ax.plot([x, x], [0, depth], 'k-', linewidth=1, alpha=0.5)
        
        for i in range(grid_height + 1):
            y = i * cell_height
            ax.plot([0, width], [y, y], 'k-', linewidth=1, alpha=0.5)
        
        # Draw rooms with different proportions
        room_colors = plt.cm.Pastel1(np.linspace(0, 1, len(room_proportions)))
        
        # Central grand hall (2x1 cells)
        hall_x = cell_width
        hall_y = cell_height
        hall_width = cell_width
        hall_height = cell_height
        
        hall = Rectangle((hall_x, hall_y), hall_width, hall_height, 
                        fill=True, edgecolor='black', facecolor=room_colors[0], alpha=0.7)
        ax.add_patch(hall)
        ax.text(hall_x + hall_width/2, hall_y + hall_height/2, "Grand Hall\n(1:1)", 
                ha='center', va='center', fontsize=10)
        
        # Left wing room (golden ratio)
        left_x = 0
        left_y = cell_height
        left_width = cell_width
        left_height = cell_height
        
        left_room = Rectangle((left_x, left_y), left_width, left_height, 
                             fill=True, edgecolor='black', facecolor=room_colors[3], alpha=0.7)
        ax.add_patch(left_room)
        ax.text(left_x + left_width/2, left_y + left_height/2, f"Golden Ratio\n(1:{self.golden_ratio:.2f})", 
                ha='center', va='center', fontsize=10)
        
        # Right wing room (2:3 ratio)
        right_x = 2 * cell_width
        right_y = cell_height
        right_width = cell_width
        right_height = cell_height
        
        right_room = Rectangle((right_x, right_y), right_width, right_height, 
                              fill=True, edgecolor='black', facecolor=room_colors[2], alpha=0.7)
        ax.add_patch(right_room)
        ax.text(right_x + right_width/2, right_y + right_height/2, "Ratio 2:3", 
                ha='center', va='center', fontsize=10)
        
        # Top reception room (1:2 ratio)
        top_x = cell_width
        top_y = 0
        top_width = cell_width
        top_height = cell_height
        
        top_room = Rectangle((top_x, top_y), top_width, top_height, 
                            fill=True, edgecolor='black', facecolor=room_colors[4], alpha=0.7)
        ax.add_patch(top_room)
        ax.text(top_x + top_width/2, top_y + top_height/2, "Ratio 1:2", 
                ha='center', va='center', fontsize=10)
        
        # Bottom salon room (3:4 ratio)
        bottom_x = cell_width
        bottom_y = 2 * cell_height
        bottom_width = cell_width
        bottom_height = cell_height
        
        bottom_room = Rectangle((bottom_x, bottom_y), bottom_width, bottom_height, 
                               fill=True, edgecolor='black', facecolor=room_colors[1], alpha=0.7)
        ax.add_patch(bottom_room)
        ax.text(bottom_x + bottom_width/2, bottom_y + bottom_height/2, "Ratio 3:4", 
                ha='center', va='center', fontsize=10)
        
        # Add key room proportion information
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=room_colors[0], alpha=0.7, label='Square (1:1)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=room_colors[1], alpha=0.7, label='Ratio 3:4'),
            plt.Rectangle((0, 0), 1, 1, facecolor=room_colors[2], alpha=0.7, label='Ratio 2:3'),
            plt.Rectangle((0, 0), 1, 1, facecolor=room_colors[3], alpha=0.7, label=f'Golden Ratio (1:{self.golden_ratio:.2f})'),
            plt.Rectangle((0, 0), 1, 1, facecolor=room_colors[4], alpha=0.7, label='Ratio 1:2')
        ]
        
        # Place legend outside the plan
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1), title="Room Proportions")
        
        # Add dimension lines
        ax.annotate("", xy=(0, depth*1.1), xytext=(width, depth*1.1),
                   arrowprops=dict(arrowstyle="<->", color='black'))
        ax.text(width/2, depth*1.15, f"Width = {width}", ha='center', va='bottom')
        
        ax.annotate("", xy=(width*1.1, 0), xytext=(width*1.1, depth),
                   arrowprops=dict(arrowstyle="<->", color='black'))
        ax.text(width*1.15, depth/2, f"Depth = Width × Golden Ratio\n= {width} × {self.golden_ratio:.2f} = {depth:.2f}", 
                ha='left', va='center')
        
        # Set plot properties
        ax.set_xlim(-width*0.1, width*1.6)
        ax.set_ylim(-depth*0.1, depth*1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _add_renaissance_info(self, fig):
        """Helper method to add information about Renaissance proportions to the figure."""
        description = (
            "Renaissance Proportional Systems\n\n"
            "In Renaissance architecture, particularly in the works of Andrea Palladio (1508-1580),\n"
            "proportions were carefully calculated according to mathematical ratios:\n\n"
            "• Building elevations often used simple fractions (1/8, 3/8, 2/8, 1/8)\n"
            "• Room proportions used harmonious ratios: 1:1 (square), 3:4, 2:3, Golden Ratio, and 1:2\n"
            "• The Golden Ratio (φ ≈ 1.618) was particularly valued for its aesthetic properties\n"
            "• Overall building dimensions were often based on simple whole-number ratios\n\n"
            "Palladio's villas exemplify these proportional systems, creating harmony through\n"
            "mathematical relationships between various architectural elements."
        )
        
        fig.text(0.5, 0.01, description, fontsize=10, ha='center', va='bottom',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round'))
    
    def draw_modernist_proportions(self, height=2.26, ax=None):
        """
        Draw a visualization of Le Corbusier's Modulor proportional system.
        
        Args:
            height: Reference height in meters (typically ceiling height, 2.26m)
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib figure with modernist proportions
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.figure
        
        # Get Modulor proportions
        modernist = self.proportion_systems['modernist']
        
        # Draw the human figure (simplified)
        self._draw_modulor_figure(ax, height, modernist)
        
        # Draw the key architectural elements
        self._draw_modulor_architecture(ax, height, modernist)
        
        # Add spiral and proportion lines
        self._draw_modulor_proportions(ax, height, modernist)
        
        # Add information about the system
        description = (
            "Le Corbusier's Modulor System (1948)\n\n"
            "The Modulor is a proportional system based on human measurements,\n"
            "the Golden Ratio, and the Fibonacci sequence.\n\n"
            "• Blue Series: 13, 21, 34, 55, 89, 144, 233, 377... cm\n"
            "• Red Series: 15, 25, 41, 66, 107, 173, 280, 453... cm\n\n"
            "These measurements relate to different parts of the human body\n"
            "and create a harmonious system for architectural proportions."
        )
        
        ax.text(height*1.5, height*0.3, description, fontsize=10, ha='left', va='top',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Set plot properties
        ax.set_xlim(-height*0.5, height*3.5)
        ax.set_ylim(-height*0.1, height*2.5)
        ax.set_aspect('equal')
        ax.set_title("Le Corbusier's Modulor Proportional System", fontsize=14)
        ax.axis('off')
        
        return fig
    
    def _draw_modulor_figure(self, ax, height, modernist):
        """Helper method to draw a simplified human figure with Modulor proportions."""
        # Calculate dimensions
        # A person with height of 1.83m (6 feet) was the basis for Modulor
        figure_height = 1.83
        
        # Calculate scale factor
        scale = height / modernist['key_dimensions']['ceiling_height']
        
        # Draw floor line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        
        # Draw ceiling line
        ceiling_height = modernist['key_dimensions']['ceiling_height'] * scale
        ax.axhline(y=ceiling_height, color='blue', linestyle='--', linewidth=2, alpha=0.5)
        ax.text(-0.1, ceiling_height, f"{ceiling_height:.2f}m", ha='right', va='center', color='blue')
        
        # Create human figure (simple line drawing)
        figure_x = height * 0.5
        
        # Head
        head_radius = figure_height * 0.07 * scale
        head = Circle((figure_x, figure_height * scale), head_radius, 
                     fill=False, edgecolor='black', linestyle='-')
        ax.add_patch(head)
        
        # Body
        body_top = figure_height * scale - head_radius
        body_bottom = figure_height * 0.5 * scale
        ax.plot([figure_x, figure_x], [body_top, body_bottom], 'k-', linewidth=2)
        
        # Arms
        shoulder_y = figure_height * 0.8 * scale
        arm_length = figure_height * 0.25 * scale
        arm_angle = 15 * np.pi / 180  # 15 degrees from horizontal
        
        # Left arm
        l_arm_x = figure_x - arm_length * np.cos(arm_angle)
        l_arm_y = shoulder_y - arm_length * np.sin(arm_angle)
        ax.plot([figure_x, l_arm_x], [shoulder_y, l_arm_y], 'k-', linewidth=2)
        
        # Right arm (raised to ceiling)
        ax.plot([figure_x, figure_x], [shoulder_y, ceiling_height], 'k-', linewidth=2)
        
        # Legs
        hip_y = figure_height * 0.5 * scale
        leg_length = figure_height * 0.5 * scale
        
        # Left leg
        l_leg_x = figure_x - leg_length * 0.2
        ax.plot([figure_x, l_leg_x], [hip_y, 0], 'k-', linewidth=2)
        
        # Right leg
        r_leg_x = figure_x + leg_length * 0.2
        ax.plot([figure_x, r_leg_x], [hip_y, 0], 'k-', linewidth=2)
        
        # Mark key heights
        # Navel (natural section point at Golden Ratio)
        navel_height = figure_height * (1 - 1/self.golden_ratio) * scale
        ax.plot([figure_x - head_radius*2, figure_x + head_radius*2], 
                [navel_height, navel_height], 'r-', linewidth=1.5)
        ax.text(figure_x - head_radius*2.5, navel_height, "Navel\n(Φ section)", 
                ha='right', va='center', color='red', fontsize=9)
        
        # Height of raised hand
        raised_hand_height = ceiling_height
        ax.plot([figure_x - head_radius*2, figure_x + head_radius*2], 
                [raised_hand_height, raised_hand_height], 'b-', linewidth=1.5)
        ax.text(figure_x - head_radius*2.5, raised_hand_height, "Raised Hand\n(226cm)", 
                ha='right', va='center', color='blue', fontsize=9)
        
        # Height of person
        ax.plot([figure_x - head_radius*2, figure_x + head_radius*2], 
                [figure_height * scale, figure_height * scale], 'g-', linewidth=1.5)
        ax.text(figure_x - head_radius*2.5, figure_height * scale, "Height\n(183cm)", 
                ha='right', va='center', color='green', fontsize=9)
    
    def _draw_modulor_architecture(self, ax, height, modernist):
        """Helper method to draw architectural elements with Modulor proportions."""
        # Calculate scale factor
        scale = height / modernist['key_dimensions']['ceiling_height']
        
        # Starting position for architectural elements
        start_x = height * 1.2
        
        # Draw wall section
        wall_height = modernist['key_dimensions']['ceiling_height'] * scale
        wall_width = height * 1.5
        
        wall = Rectangle((start_x, 0), wall_width, wall_height, 
                        fill=True, edgecolor='black', facecolor='lavender', alpha=0.5)
        ax.add_patch(wall)
        
        # Door
        door_height = modernist['key_dimensions']['door_height'] * scale
        door_width = wall_width * 0.25
        
        door = Rectangle((start_x + wall_width*0.1, 0), door_width, door_height, 
                        fill=True, edgecolor='black', facecolor='saddlebrown')
        ax.add_patch(door)
        
        # Doorknob
        doorknob_height = modernist['key_dimensions']['doorknob_height'] * scale
        doorknob = Circle((start_x + wall_width*0.1 + door_width*0.8, doorknob_height), 
                         door_width*0.05, fill=True, edgecolor='black', facecolor='gold')
        ax.add_patch(doorknob)
        
        # Window
        window_height = wall_height * 0.4
        window_width = wall_width * 0.3
        window_bottom = wall_height - window_height - wall_height * 0.1
        
        window = Rectangle((start_x + wall_width*0.6, window_bottom), 
                          window_width, window_height, 
                          fill=True, edgecolor='black', facecolor='lightblue')
        ax.add_patch(window)
        
        # Furniture
        # Table
        table_height = modernist['key_dimensions']['table_height'] * scale
        table_width = wall_width * 0.4
        table_x = start_x + wall_width*0.4
        
        table = Rectangle((table_x, table_height), table_width, table_height*0.1, 
                        fill=True, edgecolor='black', facecolor='sienna')
        ax.add_patch(table)
        
        # Table legs
        leg_width = table_width * 0.05
        for leg_x in [table_x + leg_width, table_x + table_width - leg_width*2]:
            leg = Rectangle((leg_x, 0), leg_width, table_height, 
                          fill=True, edgecolor='black', facecolor='sienna')
            ax.add_patch(leg)
        
        # Chair
        chair_height = modernist['key_dimensions']['chair_height'] * scale
        chair_width = table_width * 0.4
        chair_x = table_x + table_width * 0.3
        
        chair_seat = Rectangle((chair_x, chair_height), chair_width, chair_height*0.1, 
                             fill=True, edgecolor='black', facecolor='sienna')
        ax.add_patch(chair_seat)
        
        chair_back = Rectangle((chair_x, chair_height + chair_height*0.1), 
                              chair_width, chair_height*0.6, 
                              fill=True, edgecolor='black', facecolor='sienna', alpha=0.7)
        ax.add_patch(chair_back)
        
        # Chair legs
        chair_leg_width = chair_width * 0.1
        for leg_x in [chair_x + chair_leg_width, chair_x + chair_width - chair_leg_width*2]:
            leg = Rectangle((leg_x, 0), chair_leg_width, chair_height, 
                          fill=True, edgecolor='black', facecolor='sienna')
            ax.add_patch(leg)
        
        # Add dimension lines and labels for key heights
        self._add_modulor_dimension_lines(ax, start_x + wall_width + wall_width*0.1, scale, modernist)
    
    def _add_modulor_dimension_lines(self, ax, x_pos, scale, modernist):
        """Helper method to add dimension lines for Modulor heights."""
        # Key heights from Modulor
        heights = {
            'Chair Height': modernist['key_dimensions']['chair_height'],
            'Table Height': modernist['key_dimensions']['table_height'],
            'Doorknob Height': modernist['key_dimensions']['doorknob_height'],
            'Door Height': modernist['key_dimensions']['door_height'],
            'Ceiling Height': modernist['key_dimensions']['ceiling_height']
        }
        
        # Draw dimension lines
        for i, (name, height) in enumerate(heights.items()):
            # Scale height to drawing
            scaled_height = height * scale
            
            # Draw horizontal line at this height
            ax.plot([x_pos, x_pos + 0.1], [scaled_height, scaled_height], 'k-', linewidth=1)
            
            # Add label
            if 'Blue' in name:
                color = 'blue'
            elif 'Red' in name:
                color = 'red'
            else:
                color = 'black'
                
            ax.text(x_pos + 0.15, scaled_height, f"{name}: {height}m", 
                   ha='left', va='center', color=color, fontsize=9)
    
    def _draw_modulor_proportions(self, ax, height, modernist):
        """Helper method to draw Modulor proportional system lines and spirals."""
        # Calculate scale factor
        scale = height / modernist['key_dimensions']['ceiling_height']
        
        # Convert blue and red series to scaled heights
        blue_series = [h * scale for h in modernist['blue_series']]
        red_series = [h * scale for h in modernist['red_series']]
        
        # Starting point for the spiral
        start_x = height * 2.6
        start_y = 0
        
        # Draw golden spiral based on Fibonacci-like sequence
        # This is a simplified representation of the Modulor spiral
        rect_size = height * 0.3
        self._draw_fibonacci_spiral(ax, start_x, start_y, rect_size)
        
        # Add label
        ax.text(start_x + rect_size*1.5, start_y + rect_size*2.5, 
               "Modulor Spiral\nbased on Golden Ratio", 
               ha='center', va='center', fontsize=10)
        
        # Draw blue and red series marks
        blue_x = height * 0.2
        red_x = height * 0.4
        
        # Draw vertical scale line
        ax.plot([blue_x, blue_x], [0, blue_series[-1]], 'b-', linewidth=1)
        ax.plot([red_x, red_x], [0, red_series[-1]], 'r-', linewidth=1)
        
        # Add markers for each value in the series
        for i, h in enumerate(blue_series):
            ax.plot([blue_x - 0.05, blue_x + 0.05], [h, h], 'b-', linewidth=2)
            
            # Only label some points to avoid clutter
            if i % 2 == 0:
                ax.text(blue_x - 0.1, h, f"{modernist['blue_series'][i]}", 
                       ha='right', va='center', color='blue', fontsize=8)
        
        for i, h in enumerate(red_series):
            ax.plot([red_x - 0.05, red_x + 0.05], [h, h], 'r-', linewidth=2)
            
            # Only label some points to avoid clutter
            if i % 2 == 0:
                ax.text(red_x + 0.1, h, f"{modernist['red_series'][i]}", 
                       ha='left', va='center', color='red', fontsize=8)
        
        # Add labels for the series
        ax.text(blue_x, blue_series[-1] + 0.1, "Blue Series", 
               ha='center', va='bottom', color='blue', fontsize=10)
        ax.text(red_x, red_series[-1] + 0.1, "Red Series", 
               ha='center', va='bottom', color='red', fontsize=10)
    
    def _draw_fibonacci_spiral(self, ax, start_x, start_y, rect_size):
        """Helper method to draw a Fibonacci spiral (representative of Modulor)."""
        # Fibonacci sequence (simplified)
        fib = [1, 1, 2, 3, 5, 8, 13]
        
        # Draw rectangles forming the spiral
        x, y = start_x, start_y
        width = rect_size
        
        # Direction: 0=right, 1=up, 2=left, 3=down
        direction = 0
        
        # Draw spiral segments
        for i in range(len(fib) - 1):
            size = fib[i] * width / fib[-2]  # Scale to fit in overall size
            
            if direction == 0:  # Right
                rect = Rectangle((x, y), size, size, fill=False, edgecolor='purple', alpha=0.7)
                ax.add_patch(rect)
                x += size
            elif direction == 1:  # Up
                rect = Rectangle((x - size, y), size, size, fill=False, edgecolor='purple', alpha=0.7)
                ax.add_patch(rect)
                y += size
            elif direction == 2:  # Left
                rect = Rectangle((x - size, y - size), size, size, fill=False, edgecolor='purple', alpha=0.7)
                ax.add_patch(rect)
                x -= size
            elif direction == 3:  # Down
                rect = Rectangle((x, y - size), size, size, fill=False, edgecolor='purple', alpha=0.7)
                ax.add_patch(rect)
                y -= size
            
            direction = (direction + 1) % 4
        
        # Draw the actual spiral curve
        t = np.linspace(0, 2*np.pi, 100)
        spiral_x = []
        spiral_y = []
        
        for theta in np.linspace(0, 2*np.pi * 2, 100):
            # Logarithmic spiral formula
            r = width * np.exp(0.2 * theta)
            x = start_x + rect_size + r * np.cos(theta)
            y = start_y + rect_size + r * np.sin(theta)
            spiral_x.append(x)
            spiral_y.append(y)
        
        ax.plot(spiral_x, spiral_y, 'purple', linewidth=2, alpha=0.7)
    
    def draw_japanese_proportions(self, width=9, ax=None):
        """
        Draw a traditional Japanese room layout with tatami proportions.
        
        Args:
            width: Width of the room in any unit (typically meters)
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib figure with Japanese proportions
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.figure
        
        # Get Japanese proportions
        japanese = self.proportion_systems['japanese']
        
        # Calculate tatami size
        tatami = japanese['tatami_size']
        tatami_width_scaled = tatami['width'] * (width / 9)  # Scale to fit in given width
        tatami_length_scaled = tatami['length'] * (width / 9)
        
        # Choose a typical room size (8 tatami)
        room_size = 8
        
        # Draw the room with tatami
        self._draw_tatami_room(ax, tatami_width_scaled, tatami_length_scaled, room_size)
        
        # Add tokonoma (decorative alcove)
        self._draw_tokonoma(ax, tatami_width_scaled, tatami_length_scaled, room_size)
        
        # Add traditional elements
        self._draw_japanese_elements(ax, tatami_width_scaled, tatami_length_scaled, room_size)
        
        # Add information about tatami proportions
        self._add_tatami_info(ax, tatami_width_scaled, tatami_length_scaled, room_size, japanese)
        
        # Set plot properties
        room_width = tatami_width_scaled * 3
        room_height = tatami_length_scaled * 3
        
        ax.set_xlim(-room_width*0.1, room_width*1.5)
        ax.set_ylim(-room_height*0.1, room_height*1.2)
        ax.set_aspect('equal')
        ax.set_title("Traditional Japanese Proportions - 8 Tatami Room", fontsize=14)
        ax.axis('off')
        
        return fig
    
    def _draw_tatami_room(self, ax, tatami_width, tatami_length, room_size):
        """Helper method to draw a Japanese room with tatami mats."""
        # Tatami layouts vary by room size
        # For an 8-tatami room, a common layout is a 3x3 grid with one tatami removed
        
        tatami_color = 'khaki'
        border_color = 'saddlebrown'
        
        # Draw tatami mats
        if room_size == 8:
            # 8-tatami layout
            positions = [
                # Horizontal tatami (1:2 ratio)
                (0, 0),
                (0, tatami_width),
                (tatami_length, 0),
                (tatami_length, tatami_width),
                # Vertical tatami (2:1 ratio)
                (0, tatami_width*2),
                (tatami_width, tatami_width*2),
                (tatami_length, tatami_width*2),
                (tatami_length + tatami_width, 0)
            ]
            
            orientations = [
                'horizontal', 'horizontal', 'horizontal', 'horizontal',
                'vertical', 'vertical', 'vertical', 'vertical'
            ]
        elif room_size == 6:
            # 6-tatami layout
            positions = [
                # Horizontal tatami (1:2 ratio)
                (0, 0),
                (0, tatami_width),
                (tatami_length, 0),
                # Vertical tatami (2:1 ratio)
                (0, tatami_width*2),
                (tatami_width, tatami_width*2),
                (tatami_length, tatami_width*2)
            ]
            
            orientations = [
                'horizontal', 'horizontal', 'horizontal',
                'vertical', 'vertical', 'vertical'
            ]
        elif room_size == 4:
            # 4.5-tatami layout (4 full tatami + half tatami in center)
            positions = [
                # Horizontal tatami (1:2 ratio)
                (0, 0),
                (0, tatami_width + tatami_width/2),
                # Vertical tatami (2:1 ratio)
                (tatami_width/2, tatami_width/2),
                (tatami_length, 0),
                (tatami_length, tatami_width + tatami_width/2)
            ]
            
            orientations = [
                'horizontal', 'horizontal',
                'center_half',
                'vertical', 'vertical'
            ]
        else:
            # Default to 4.5-tatami layout
            positions = [
                # Horizontal tatami (1:2 ratio)
                (0, 0),
                (0, tatami_width + tatami_width/2),
                # Vertical tatami (2:1 ratio)
                (tatami_width/2, tatami_width/2),
                (tatami_length, 0),
                (tatami_length, tatami_width + tatami_width/2)
            ]
            
            orientations = [
                'horizontal', 'horizontal',
                'center_half',
                'vertical', 'vertical'
            ]
        
        # Draw the tatami mats
        for i, (pos, orientation) in enumerate(zip(positions, orientations)):
            if orientation == 'horizontal':
                width = tatami_length
                height = tatami_width
            elif orientation == 'vertical':
                width = tatami_width
                height = tatami_length
            elif orientation == 'center_half':
                width = tatami_width
                height = tatami_width
            
            # Draw tatami mat
            tatami = Rectangle(pos, width, height, 
                             fill=True, edgecolor=border_color, facecolor=tatami_color, 
                             linewidth=2, alpha=0.8)
            ax.add_patch(tatami)
            
            # Add tatami border lines
            inner_border = Rectangle((pos[0] + width*0.05, pos[1] + height*0.05), 
                                   width*0.9, height*0.9, 
                                   fill=False, edgecolor=border_color, 
                                   linewidth=1, alpha=0.6)
            ax.add_patch(inner_border)
            
            # Add tatami number
            ax.text(pos[0] + width/2, pos[1] + height/2, str(i+1), 
                   ha='center', va='center', fontsize=10, color=border_color)
        
        # Draw room outline
        if room_size == 8:
            room_width = tatami_length + tatami_width
            room_height = tatami_width * 3
        elif room_size == 6:
            room_width = tatami_length
            room_height = tatami_width * 3
        elif room_size == 4:
            room_width = tatami_length + tatami_width/2
            room_height = tatami_width * 2 + tatami_width/2
        else:
            room_width = tatami_length + tatami_width/2
            room_height = tatami_width * 2 + tatami_width/2
        
        room_outline = Rectangle((0, 0), room_width, room_height, 
                               fill=False, edgecolor='black', linewidth=3)
        ax.add_patch(room_outline)
    
    def _draw_tokonoma(self, ax, tatami_width, tatami_length, room_size):
        """Helper method to draw a tokonoma (decorative alcove) in the Japanese room."""
        # Tokonoma position depends on room layout
        if room_size == 8:
            # For 8-tatami room, tokonoma often replaces one tatami
            tokonoma_x = tatami_length + tatami_width
            tokonoma_y = tatami_width
            tokonoma_width = tatami_width
            tokonoma_height = tatami_length
        elif room_size == 6:
            # For 6-tatami room, tokonoma often at the end
            tokonoma_x = tatami_length
            tokonoma_y = tatami_width
            tokonoma_width = tatami_width/2
            tokonoma_height = tatami_width
        else:
            # Default position
            tokonoma_x = tatami_length
            tokonoma_y = tatami_width * 1.5
            tokonoma_width = tatami_width/2
            tokonoma_height = tatami_width
        
        # Draw tokonoma
        tokonoma = Rectangle((tokonoma_x, tokonoma_y), tokonoma_width, tokonoma_height, 
                           fill=True, edgecolor='black', facecolor='lavender', 
                           linewidth=2, alpha=0.7)
        ax.add_patch(tokonoma)
        
        # Draw tokonoma platform (slightly raised)
        platform_height = tokonoma_height * 0.1
        platform = Rectangle((tokonoma_x, tokonoma_y), tokonoma_width, platform_height, 
                           fill=True, edgecolor='black', facecolor='sienna', 
                           linewidth=1, alpha=0.8)
        ax.add_patch(platform)
        
        # Draw a simple scroll in the tokonoma
        scroll_width = tokonoma_width * 0.4
        scroll_height = tokonoma_height * 0.7
        scroll_x = tokonoma_x + tokonoma_width/2 - scroll_width/2
        scroll_y = tokonoma_y + platform_height + (tokonoma_height - platform_height - scroll_height)/2
        
        scroll = Rectangle((scroll_x, scroll_y), scroll_width, scroll_height, 
                         fill=True, edgecolor='black', facecolor='white', 
                         linewidth=1, alpha=0.9)
        ax.add_patch(scroll)
        
        # Add a label
        ax.text(tokonoma_x + tokonoma_width/2, tokonoma_y - 0.1, "Tokonoma\n(Alcove)", 
               ha='center', va='top', fontsize=9)
    
    def _draw_japanese_elements(self, ax, tatami_width, tatami_length, room_size):
        """Helper method to draw traditional Japanese room elements."""
        # Calculate room dimensions
        if room_size == 8:
            room_width = tatami_length + tatami_width
            room_height = tatami_width * 3
        elif room_size == 6:
            room_width = tatami_length
            room_height = tatami_width * 3
        elif room_size == 4:
            room_width = tatami_length + tatami_width/2
            room_height = tatami_width * 2 + tatami_width/2
        else:
            room_width = tatami_length + tatami_width/2
            room_height = tatami_width * 2 + tatami_width/2
        
        # Draw shoji (paper sliding doors)
        # Typically on one or two sides of the room
        shoji_width = tatami_width
        shoji_height = tatami_width * 0.1
        
        # Bottom shoji
        bottom_shoji = Rectangle((room_width/4, -shoji_height), room_width/2, shoji_height, 
                               fill=True, edgecolor='black', facecolor='white', 
                               linewidth=1, alpha=0.9)
        ax.add_patch(bottom_shoji)
        
        # Draw shoji grid pattern
        grid_spacing = shoji_width / 6
        for i in range(1, 6):
            # Vertical grid lines
            ax.plot([room_width/4 + i*grid_spacing, room_width/4 + i*grid_spacing], 
                   [-shoji_height, 0], 'k-', linewidth=0.5, alpha=0.5)
            
            # Horizontal grid lines
            ax.plot([room_width/4, room_width/4 + room_width/2], 
                   [-shoji_height/2, -shoji_height/2], 'k-', linewidth=0.5, alpha=0.5)
        
        # Label
        ax.text(room_width/2, -shoji_height - 0.05, "Shoji (Sliding Doors)", 
               ha='center', va='top', fontsize=9)
        
        # Draw zabuton (floor cushions)
        zabuton_size = tatami_width * 0.4
        
        # Place a few zabuton in the room
        zabuton_positions = [
            (tatami_length / 2, tatami_width / 2),
            (tatami_length / 2, tatami_width * 1.5),
            (tatami_length * 0.8, tatami_width)
        ]
        
        for pos in zabuton_positions:
            zabuton = Rectangle((pos[0] - zabuton_size/2, pos[1] - zabuton_size/2), 
                              zabuton_size, zabuton_size, 
                              fill=True, edgecolor='black', facecolor='royalblue', 
                              linewidth=1, alpha=0.7)
            ax.add_patch(zabuton)
        
        # Draw chabudai (low table)
        table_width = tatami_width * 0.8
        table_height = tatami_width * 0.5
        table_x = tatami_length / 2 - table_width / 2
        table_y = tatami_width - table_height / 2
        
        table = Rectangle((table_x, table_y), table_width, table_height, 
                        fill=True, edgecolor='black', facecolor='sienna', 
                        linewidth=1, alpha=0.7)
        ax.add_patch(table)
        
        # Label
        ax.text(table_x + table_width/2, table_y + table_height/2, "Chabudai\n(Low Table)", 
               ha='center', va='center', fontsize=8, color='white')
    
    def _add_tatami_info(self, ax, tatami_width, tatami_length, room_size, japanese):
        """Helper method to add information about tatami proportions."""
        # Calculate room dimensions
        if room_size == 8:
            room_width = tatami_length + tatami_width
            room_height = tatami_width * 3
        elif room_size == 6:
            room_width = tatami_length
            room_height = tatami_width * 3
        elif room_size == 4:
            room_width = tatami_length + tatami_width/2
            room_height = tatami_width * 2 + tatami_width/2
        else:
            room_width = tatami_length + tatami_width/2
            room_height = tatami_width * 2 + tatami_width/2
        
        # Show tatami dimensions
        # Standard tatami is roughly 1:2 ratio
        ax.text(room_width * 1.1, room_height * 0.1, 
               f"Standard Tatami Size:\n{japanese['tatami_size']['width']}m × {japanese['tatami_size']['length']}m\n(Ratio 1:2)", 
               ha='left', va='top', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Show common room sizes
        room_info = "Traditional Room Sizes:\n"
        for room in japanese['room_sizes']:
            room_info += f"• {room['name']}: {room['size']} tatami mats\n"
        
        ax.text(room_width * 1.1, room_height * 0.5, room_info, 
               ha='left', va='top', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Information about proportions
        proportion_info = (
            "Japanese Proportional System\n\n"
            "Traditional Japanese architecture uses the tatami mat as its fundamental unit of measure.\n"
            "The tatami's 1:2 proportion creates a flexible modular system:\n\n"
            "• Rooms are sized by the number of tatami mats (4.5, 6, 8, etc.)\n"
            "• Building dimensions follow the ken (approx. 1.82m) grid system\n"
            "• Ceiling height is typically 3 ken (about 5.46m)\n"
            "• Tokonoma (alcove) dimensions follow specific proportions\n"
            "• Shoji and fusuma (sliding doors) use standardized proportions\n\n"
            "This system creates harmony and standardization while allowing for\n"
            "flexibility in design."
        )
        
        ax.text(room_width/2, room_height * 1.1, proportion_info, 
               ha='center', va='bottom', fontsize=9,
               bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round'))


class PerspectiveTransformation:
    """
    A class for performing and visualizing perspective transformations on images and shapes.
    """
    
    def __init__(self):
        """Initialize the perspective transformation tool."""
        pass
    
    def transform_rectangle(self, width, height, source_points, ax=None):
        """
        Transform a rectangle using perspective transformation.
        
        Args:
            width: Width of the original rectangle
            height: Height of the original rectangle
            source_points: Four corner points to transform to (top-left, top-right, bottom-right, bottom-left)
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib axis with the transformed rectangle
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Original rectangle corners
        original_points = np.array([
            [0, 0],           # Top-left
            [width, 0],       # Top-right
            [width, height],  # Bottom-right
            [0, height]       # Bottom-left
        ], dtype=np.float32)
        
        # Ensure source_points is a numpy array
        source_points = np.array(source_points, dtype=np.float32)
        
        # Calculate transformation matrix
        matrix = cv2.getPerspectiveTransform(original_points, source_points)
        
        # Define a grid of points to transform
        grid_size = 10
        x_step = width / grid_size
        y_step = height / grid_size
        
        grid_points = []
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                grid_points.append([i * x_step, j * y_step])
        
        grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply transformation
        transformed_grid = cv2.perspectiveTransform(grid_points, matrix)
        transformed_grid = transformed_grid.reshape(-1, 2)
        
        # Plot original rectangle
        rect = Rectangle((0, 0), width, height, fill=False, edgecolor='blue', linewidth=2, label='Original')
        ax.add_patch(rect)
        
        # Plot grid on original rectangle
        for i in range(grid_size + 1):
            ax.plot([0, width], [i * y_step, i * y_step], 'b-', alpha=0.3)
            ax.plot([i * x_step, i * x_step], [0, height], 'b-', alpha=0.3)
        
        # Plot transformed points
        for i in range(len(transformed_grid)):
            x, y = transformed_grid[i]
            ax.plot(x, y, 'r.', markersize=3)
        
        # Draw the transformed grid
        for i in range(grid_size + 1):
            # Horizontal lines
            x_points = transformed_grid[i*(grid_size+1):(i+1)*(grid_size+1), 0]
            y_points = transformed_grid[i*(grid_size+1):(i+1)*(grid_size+1), 1]
            ax.plot(x_points, y_points, 'r-', alpha=0.5)
            
            # Vertical lines
            for j in range(grid_size + 1):
                x_points = [transformed_grid[j + i*(grid_size+1), 0] for i in range(grid_size + 1)]
                y_points = [transformed_grid[j + i*(grid_size+1), 1] for i in range(grid_size + 1)]
                ax.plot(x_points, y_points, 'r-', alpha=0.5)
        
        # Plot transformed rectangle outline
        transformed_corners = np.array([
            transformed_grid[0],                    # Top-left
            transformed_grid[grid_size],            # Top-right
            transformed_grid[-grid_size-1],         # Bottom-right
            transformed_grid[-(grid_size+1)*(grid_size+1)+grid_size]  # Bottom-left
        ])
        
        ax.fill(transformed_corners[:, 0], transformed_corners[:, 1], 
               facecolor='red', alpha=0.2, edgecolor='red', linewidth=2, label='Transformed')
        
        # Set plot properties
        all_x = np.concatenate([original_points[:, 0], source_points[:, 0]])
        all_y = np.concatenate([original_points[:, 1], source_points[:, 1]])
        
        margin = max(width, height) * 0.2
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Perspective Transformation of a Rectangle')
        ax.legend()
        
        return ax
    
    def demonstrate_anamorphosis(self, ax=None):
        """
        Demonstrate anamorphic projection (extreme perspective).
        
        Args:
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib axis with the anamorphic projection
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a simple grid as the original image (like a checkerboard)
        grid_size = 8
        checker_size = 1.0
        
        # Draw original grid
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    rect = Rectangle((i*checker_size, j*checker_size), 
                                   checker_size, checker_size, 
                                   fill=True, edgecolor='black', facecolor='black')
                    ax.add_patch(rect)
        
        # Add border
        border = Rectangle((0, 0), grid_size*checker_size, grid_size*checker_size, 
                         fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(border)
        
        # Define projection points for anamorphic view
        # This creates an extreme perspective as if viewed from a low angle
        width = grid_size * checker_size
        height = grid_size * checker_size
        
        # Anamorphic perspective points
        anamorphic_points = np.array([
            [width*0.2, height*0.7],         # Top-left
            [width*1.8, height*0.7],         # Top-right
            [width*2.5, height*2.0],         # Bottom-right
            [-width*0.5, height*2.0]         # Bottom-left
        ], dtype=np.float32)
        
        # Original rectangle corners
        original_points = np.array([
            [0, 0],           # Top-left
            [width, 0],       # Top-right
            [width, height],  # Bottom-right
            [0, height]       # Bottom-left
        ], dtype=np.float32)
        
        # Calculate transformation matrix
        matrix = cv2.getPerspectiveTransform(original_points, anamorphic_points)
        
        # Transform all vertices of the checkerboard
        vertices = []
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                vertices.append([i*checker_size, j*checker_size])
        
        vertices = np.array(vertices, dtype=np.float32).reshape(-1, 1, 2)
        transformed_vertices = cv2.perspectiveTransform(vertices, matrix).reshape(-1, 2)
        
        # Draw anamorphic projection
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    # Get the four corners of this checkerboard square
                    idx1 = j * (grid_size + 1) + i
                    idx2 = j * (grid_size + 1) + (i + 1)
                    idx3 = (j + 1) * (grid_size + 1) + (i + 1)
                    idx4 = (j + 1) * (grid_size + 1) + i
                    
                    # Draw the transformed square
                    polygon = Polygon([
                        transformed_vertices[idx1],
                        transformed_vertices[idx2],
                        transformed_vertices[idx3],
                        transformed_vertices[idx4]
                    ], closed=True, fill=True, edgecolor='black', facecolor='red', alpha=0.5)
                    
                    ax.add_patch(polygon)
        
        # Draw transformation lines between original and anamorphic corners
        for i in range(4):
            ax.plot([original_points[i, 0], anamorphic_points[i, 0]], 
                   [original_points[i, 1], anamorphic_points[i, 1]], 
                   'g--', alpha=0.5)
        
        # Add viewer position (where anamorphic image looks correct)
        viewer_x = width * 0.5
        viewer_y = -height * 0.5
        viewer_z = height * 0.8  # Represents height above the ground
        
        ax.plot(viewer_x, viewer_y, 'go', markersize=10, label='Viewer Position')
        
        # Draw viewing lines from viewer to key points
        for point in anamorphic_points:
            ax.plot([viewer_x, point[0]], [viewer_y, point[1]], 'g-', alpha=0.3)
        
        # Add information about anamorphosis
        ax.text(width * 1.5, height * 0.2, 
               "Anamorphic Projection\n\n"
               "Anamorphosis is a distorted projection that appears normal\n"
               "when viewed from a specific vantage point.\n\n"
               "Famous examples include Holbein's 'The Ambassadors' (1533)\n"
               "and modern street art that appears 3D from a particular angle.",
               ha='center', va='top', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Set plot properties
        margin = max(width, height)
        ax.set_xlim(-width*0.6, width*2.6)
        ax.set_ylim(-height*0.6, height*2.1)
        
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Anamorphic Projection Demonstration')
        ax.legend()
        
        return ax
    
    def demonstrate_foreshortening(self, ax=None):
        """
        Demonstrate foreshortening (variation in size due to perspective).
        
        Args:
            ax: Matplotlib axis to draw on
            
        Returns:
            Matplotlib axis with the foreshortening demonstration
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a row of identical squares receding into the distance
        num_squares = 7
        square_size = 1.0
        
        # Define vanishing point for one-point perspective
        vp_x = 5.0
        vp_y = 4.0
        
        # Draw squares with foreshortening
        for i in range(num_squares):
            # Calculate distance factor for foreshortening
            distance = i + 1
            scale_factor = 1 / distance
            
            # Calculate position based on perspective
            # As objects get further, they get closer to the vanishing point
            factor = 1 - scale_factor
            x = vp_x - (vp_x - i * square_size) * scale_factor - square_size * scale_factor / 2
            y = vp_y - (vp_y - 2) * scale_factor - square_size * scale_factor / 2
            
            width = square_size * scale_factor
            height = square_size * scale_factor
            
            square = Rectangle((x, y), width, height, 
                             fill=True, edgecolor='black', facecolor='blue', alpha=0.7-i*0.1)
            ax.add_patch(square)
            
            # Add measurement lines to show relative sizes
            ax.annotate("", xy=(x, y-0.2), xytext=(x+width, y-0.2),
                       arrowprops=dict(arrowstyle="<->", color='red'))
            ax.text(x+width/2, y-0.3, f"{width:.2f}", ha='center', va='top', fontsize=8)
            
            # Label each square
            ax.text(x+width/2, y+height/2, f"{i+1}", ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw vanishing point
        ax.plot(vp_x, vp_y, 'ro', markersize=8, label='Vanishing Point')
        
        # Draw perspective lines to vanishing point
        for i in [0, num_squares-1]:
            distance = i + 1
            scale_factor = 1 / distance
            
            x = vp_x - (vp_x - i * square_size) * scale_factor - square_size * scale_factor / 2
            y = vp_y - (vp_y - 2) * scale_factor - square_size * scale_factor / 2
            width = square_size * scale_factor
            height = square_size * scale_factor
            
            # Lines from corners to vanishing point
            ax.plot([x, vp_x], [y, vp_y], 'r--', alpha=0.3)
            ax.plot([x+width, vp_x], [y, vp_y], 'r--', alpha=0.3)
            ax.plot([x, vp_x], [y+height, vp_y], 'r--', alpha=0.3)
            ax.plot([x+width, vp_x], [y+height, vp_y], 'r--', alpha=0.3)
        
        # Add explanation of foreshortening
        explanation = (
            "Foreshortening\n\n"
            "The apparent size of identical objects decreases with distance.\n"
            "This follows the mathematical relationship:\n\n"
            "Apparent Size = Actual Size / Distance\n\n"
            "In artistic perspective, this creates the illusion of depth and distance."
        )
        
        ax.text(0, 6, explanation, fontsize=10, ha='left', va='top',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Set plot properties
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 8)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Foreshortening Demonstration')
        ax.legend()
        
        return ax


def demonstrate_perspective_systems():
    """Demonstrate different perspective systems."""
    # Create the perspective system
    perspective = LinearPerspective(width=12, height=8)
    
    # Create figure with multiple perspective types
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # One-point perspective
    perspective.set_one_point_perspective()
    perspective.draw_perspective_grid(ax=axes[0])
    perspective.draw_cube_in_perspective(2, 1, 2, 2, 2, ax=axes[0])
    
    # Two-point perspective
    perspective.set_two_point_perspective()
    perspective.draw_perspective_grid(ax=axes[1])
    perspective.draw_cube_in_perspective(6, 1, 2, 2, 2, ax=axes[1])
    
    # Three-point perspective
    perspective.set_three_point_perspective(vp3_y=12)
    perspective.draw_perspective_grid(ax=axes[2])
    perspective.draw_cube_in_perspective(6, 1, 2, 2, 2, ax=axes[2])
    
    plt.tight_layout()
    plt.suptitle('Comparison of Perspective Systems', fontsize=16, y=1.05)
    
    return fig

def demonstrate_proportional_systems():
    """Demonstrate different proportional systems."""
    # Create the proportion system
    proportions = ProportionSystem()
    
    # Create figure with multiple proportion types
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Golden ratio grid
    proportions.draw_proportional_grid(10, 10, grid_type='golden', ax=axes[0, 0])
    
    # Rule of thirds grid
    proportions.draw_proportional_grid(10, 10, grid_type='thirds', ax=axes[0, 1])
    
    # Dynamic symmetry grid
    proportions.draw_proportional_grid(10, 10, grid_type='dynamic', ax=axes[1, 0])
    
    # Fibonacci grid
    proportions.draw_proportional_grid(10, 10, grid_type='fibonacci', ax=axes[1, 1])
    
    plt.tight_layout()
    plt.suptitle('Comparison of Proportional Systems', fontsize=16, y=0.98)
    
    return fig

def demonstrate_human_proportions():
    """Demonstrate human proportion systems."""
    # Create the human proportions system
    human_proportions = HumanProportions()
    
    # Create figure with comparison of proportion systems
    fig = human_proportions.compare_proportion_systems(height=180)  # 180 cm height
    
    return fig

def demonstrate_architectural_proportions():
    """Demonstrate architectural proportion systems."""
    # Create the architectural proportions system
    arch_proportions = ArchitecturalProportions()
    
    # Create figure with multiple demonstrations
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Classical orders
    ax1 = fig.add_subplot(gs[0, 0])
    arch_proportions.draw_classical_orders(height=10, ax=ax1)
    
    # Renaissance proportions
    ax2 = fig.add_subplot(gs[0, 1])
    arch_proportions.draw_renaissance_proportions(width=8, ax=ax2)
    
    # Modernist (Modulor) proportions
    ax3 = fig.add_subplot(gs[1, 0])
    arch_proportions.draw_modernist_proportions(height=2.26, ax=ax3)
    
    # Japanese proportions
    ax4 = fig.add_subplot(gs[1, 1])
    arch_proportions.draw_japanese_proportions(width=9, ax=ax4)
    
    plt.tight_layout()
    plt.suptitle('Architectural Proportion Systems', fontsize=16, y=0.98)
    
    return fig

def demonstrate_perspective_transformations():
    """Demonstrate perspective transformations."""
    # Create the perspective transformation system
    perspective_transform = PerspectiveTransformation()
    
    # Create figure with multiple demonstrations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Simple perspective transformation
    width, height = 4, 3
    target_points = np.array([
        [1, 1],               # Top-left
        [width+2, 0],         # Top-right
        [width+1, height+1],  # Bottom-right
        [0, height+2]         # Bottom-left
    ])
    perspective_transform.transform_rectangle(width, height, target_points, ax=axes[0])
    
    # Anamorphic projection
    perspective_transform.demonstrate_anamorphosis(ax=axes[1])
    
    # Foreshortening
    perspective_transform.demonstrate_foreshortening(ax=axes[2])
    
    plt.tight_layout()
    plt.suptitle('Perspective Transformations and Effects', fontsize=16, y=1.05)
    
    return fig

def main():
    """Main function to demonstrate all systems."""
    print("Demonstrating Perspective and Proportion Systems")
    print("-" * 50)
    
    # Perspective systems demonstration
    print("1. Linear Perspective Systems...")
    perspective_fig = demonstrate_perspective_systems()
    
    # Proportion systems demonstration
    print("2. Proportional Grid Systems...")
    proportion_fig = demonstrate_proportional_systems()
    
    # Human proportions demonstration
    print("3. Human Proportion Systems...")
    human_fig = demonstrate_human_proportions()
    
    # Architectural proportions demonstration
    print("4. Architectural Proportion Systems...")
    arch_fig = demonstrate_architectural_proportions()
    
    # Perspective transformations demonstration
    print("5. Perspective Transformations...")
    transform_fig = demonstrate_perspective_transformations()
    
    # Display all figures
    plt.show()


if __name__ == "__main__":
    main()