"""Module for simulating a lap around a racing track given vehicle and track data."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class LapSimulator:
    def __init__(self, vehicle, racing_line):
        self.vehicle = vehicle
        self.racing_line = racing_line
        self.n_points = len(racing_line)
        
        # Calculate track geometry
        self.calculate_geometry()
        
        # Initialize velocity array
        self.v_corner = np.zeros(self.n_points)
        self.v_accel = np.zeros(self.n_points)
        self.v_brake = np.zeros(self.n_points)
        self.v_final = np.zeros(self.n_points)
        
    def calculate_geometry(self):
        """Calculate curvature and distances along the racing line"""
        x = self.racing_line[:, 0]
        y = self.racing_line[:, 1]
        
        # Calculate distances between points
        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        self.distances = np.sqrt(dx**2 + dy**2)
        
        # Calculate curvature using finite differences
        # First derivatives
        dx_ds = np.gradient(x)
        dy_ds = np.gradient(y)
        
        # Second derivatives
        d2x_ds2 = np.gradient(dx_ds)
        d2y_ds2 = np.gradient(dy_ds)
        
        # Curvature formula: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
        denominator = (dx_ds**2 + dy_ds**2)**(3/2)
        
        # No dividing by 0
        denominator = np.maximum(denominator, 1e-10)
        self.curvature = numerator / denominator
        
        self.radius = np.where(self.curvature > 1e-6, 1.0 / self.curvature, 1e6)
        
    def calculate_cornering_speed_limit(self):
        """Calculate maximum speed based on lateral grip at each point"""
        g = 9.81
        
        for i in range(self.n_points):
            # Start with base grip
            weight = self.vehicle.total_mass * g
            
            # Iteratively solve for speed considering downforce
            v = 50.0  # (m/s) We can also change this initial guess
            
            for iteration in range(5):  # A few iterations should converge
                # Calculate downforce at this speed
                downforce = self.vehicle.downforce(v)
                
                # Total normal force (weight + downforce)
                normal_force = weight + abs(downforce)
                
                # Available lateral acceleration (considering load sensitivity)
                mu = self.vehicle.tire_mu(normal_force / 4)  # / wheel
                
                # Mu must be positive
                if mu <= 0:
                    mu = 1.0  # Minimum safe friction
                
                max_lat_accel = mu * g
                
                # Speed from lateral acceleration: v = sqrt(a * r)
                if self.radius[i] < 1e5:  # Not a straight
                    # NOTE: We can adjust some of these calculations and values as needed as well.
                    v_new = np.sqrt(max_lat_accel * self.radius[i])
                else:
                    v_new = 120.0  # High speed for straights
                
                # Convergence
                if abs(v_new - v) < 0.1:
                    break
                v = v_new
            
            self.v_corner[i] = v
    
    def forward_pass(self):
        """Simulate acceleration from start, limited by engine power"""
        self.v_accel[0] = 0.0  # Rest
        
        for i in range(1, self.n_points):
            v_current = self.v_accel[i-1]
            distance = self.distances[i-1]
            
            # Calculate acceleration force (simplified - using max power)
            if v_current < 1.0:
                v_current = 1.0  
            
            # Power-limited acceleration: F = p/v
            wheel_force = self.vehicle.max_power / v_current
            drag = self.vehicle.drag_force(v_current)
            rolling = self.vehicle.rolling_resistance * self.vehicle.total_mass * 9.81
            
            accel = (wheel_force - drag - rolling) / self.vehicle.total_mass
            
            # Limit acceleration by traction
            max_accel = self.vehicle.mu_base * 9.81
            accel = min(accel, max_accel)
            
            # Ensure positive acceleration
            accel = max(accel, 0.1)
            
            # Use kinematic equation: v_f^2 = v_i^2 + 2*a*d
            v_squared = v_current**2 + 2 * accel * distance
            v_new = np.sqrt(max(v_squared, 0))
            
            # Can't exceed cornering limit
            self.v_accel[i] = min(v_new, self.v_corner[i])
    
    def backward_pass(self):
        """Simulate braking from end, ensuring we can make every corner"""
        # Start from the end with whatever speed we achieved in forward pass
        self.v_brake[-1] = self.v_accel[-1]
        
        for i in range(self.n_points - 2, -1, -1):
            v_ahead = self.v_brake[i+1]
            distance = self.distances[i]
            
            # Calculate maximum braking deceleration
            decel = self.vehicle.max_brake_decel(v_ahead)
            
            # Use kinematic equation going backward: v_i^2 = v_f^2 + 2*a*d
            v_squared = v_ahead**2 + 2 * decel * distance
            v_new = np.sqrt(v_squared)
            
            # Take the minimum of: what we could brake from, cornering limit, accel limit
            self.v_brake[i] = min(v_new, self.v_corner[i], self.v_accel[i])
    
    def combine_limits(self):
        """Final velocity is minimum of all three limits"""
        self.v_final = np.minimum(
            np.minimum(self.v_corner, self.v_accel),
            self.v_brake
        )
    
    def calculate_lap_time(self):
        """Convert velocities to lap time"""
        lap_time = 0.0
        times = np.zeros(self.n_points)
        
        for i in range(self.n_points):
            distance = self.distances[i]
            # Average velocity over this segment
            v_avg = (self.v_final[i] + self.v_final[(i+1) % self.n_points]) / 2.0
            v_avg = max(v_avg, 0.1)  # DIVIDE BY 0 BAD
            
            segment_time = distance / v_avg
            lap_time += segment_time
            times[i] = lap_time
        
        return lap_time, times
    
    def simulate(self):
        """Run the full lap simulation"""
        logger.info("Calculating cornering speed limits...")
        self.calculate_cornering_speed_limit()
        
        logger.info("Running forward pass (acceleration)...")
        self.forward_pass()
        
        logger.info("Running backward pass (braking)...")
        self.backward_pass()
        
        logger.info("Combining speed limits...")
        self.combine_limits()
        
        logger.info("Calculating lap time...")
        lap_time, times = self.calculate_lap_time()
        
        return lap_time, times