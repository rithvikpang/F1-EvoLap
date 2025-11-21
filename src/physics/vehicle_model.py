import math
import numpy as np

class VehicleModel:

    def __init__(self):
        
        # mass of car in kg
        self.mass = 795,
        self.fuel_mass = 7,
        self.total_mass = self.mass + self.fuel_mass

        # weight distribution
        self.weight_dist_front = 0.45
        self.weight_dist_rear = 0.55

        # chassis in meters
        self.wheelbase = 3.6
        self.track_width = 1.6
        self.cg_height = 0.33
        self.yaw_inertia = 800 # kg*m^2

        # aerodynamics
        self.air_density = 1.225 # kg/m^3
        self.cd = 0.85 # drag coefficient
        self.cl = 3.5 # lift coefficient (negative for downforce)
        self.frontal_area = 1.5 # m^2
        self.aero_balance = 0.45 # percentage of downforce on front axle
        self.drag_multiplier = 1.0
        self.downforce_multiplier = 1.0

        # ENGINE (ICE)
        self.max_power = 820000 # Watts
        self.engine_inertia = 0.2 # kg*m^2
        self.rpm = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000])
        self.torque = np.array([200, 300, 400, 500, 600, 700, 750, 780, 800, 790, 760, 700, 600])
        self.shift_rpm = 12100

        # GEARBOX
        self.gear_ratios = [3.20, 2.40, 1.90, 1.60, 1.40, 1.25, 1.15, 1.05]
        self.final_drive_ratio = 3.6
        self.drivetrain_efficiency = 0.85
        self.wheel_radius = 0.33 # meters

        # Hybrid System (MGU-K)
        self.ers_max_deploy = 120000 # Watts
        self.ers_max_recover = 240000 # Watts
        self.ers_energy_capacity = 4e6 # Joules
        self.ers_soc = self.ers_capacity  # start full
        self.ers_deploy_efficiency = 0.85

        # Tires
        self.mu_base = 2.4 # base friction coefficient
        self.mu_load_sensitivity = 0.0008 # load sensitivity
        self.rolling_resistance = 0.015 # rolling resistance coefficient
        self.optimal_slip_angle = 6 # degrees

        # Brakes
        self.front_brake_force = 12_000 # Newtons
        self.rear_brake_force = 10_000 # Newtons
        self.brake_bias = 0.54 # percentage of braking force on front axle
        self.brake_fade_coefficient = 0.01 # fade coefficient
        self.brake_cooling_efficiency = 0.8

    def get_engine_torque(self, rpm):
        return np.interp(rpm, self.rpm, self.torque)
    
    def wheel_torque(self, rpm, gear):
        engine_tourque = self.get_engine_torque(rpm)
        gear_ratio = self.gear_ratios[gear]
        total_ratio = gear_ratio * self.final_drive_ratio
        return engine_tourque * total_ratio * self.drivetrain_efficiency
    
    def accel_force(self, rpm, gear, v): 
        torque = self.wheel_torque(rpm, gear)
        wheel_force = torque / self.wheel_radius
        return wheel_force - self.drag_force(v) - self.rolling_resistance * self.total_mass * 9.81

    def drag_force(self, v):
        return 0.5 * self.air_density * v*v * self.CD * self.frontal_area * self.drag_mult

    def downforce(self, v):
        return 0.5 * self.air_density * v*v * self.CL * self.frontal_area * self.downforce_mult

    def tire_friction(self, normal_load):
        return self.mu_base + self.mu_load_sensitivity * (normal_load - (self.total_mass * 9.81) / 4)

    def max_brake_force(self, v):
        aero_drag = self.drag_force(v)
        total_brake = (self.front_brake_force + self.rear_brake_force) + aero_drag
        return total_brake / self.total_mass


