"""Module defining the vehicle model and its physical properties."""
import math
import numpy as np

class VehicleModel:

    def __init__(self, params):
        
        # mass of car in kg (SHOULD NOT BE EVOLVED)
        self.mass = 795 # min 798??
        self.fuel_mass = params.get("fuel_mass", 7)
        self.air_density = 1.225
        self.wheel_radius = 0.33
        self.g = 9.81
        self.total_mass = self.mass + self.fuel_mass

        # weight distribution
        self.weight_dist_front = 0.45
        self.weight_dist_rear = 0.55

        # aerodynamics (ALL EVOLVABLE)
        self.CD = params.get("CD", 0.85) # drag coefficient (0.7-1.1)
        self.CL = params.get("CL", 3.5) # lift coefficient (negative for downforce) (-1 to -2?)
        self.frontal_area = params.get("frontal_area", 1.5) # m^2
        self.aero_balance = params.get("aero_balance", 0.45) # percentage of downforce on front axle
        self.drag_mult = params.get("drag_mult", 1.0)
        self.downforce_mult = params.get("downforce_mult", 1.0)

        # ENGINE (ICE)
        self.max_power = 820000 # Watts
        self.engine_inertia = params.get("engine_inertia", 0.25) # kg*m^2
        self.rpm_curve = params.get("rpm_curve",
            np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000])
        )
        self.torque_curve = params.get("torque_curve",
            np.array([200, 300, 400, 500, 600, 700, 750, 780, 800, 790, 760, 700, 600])
        )
        self.shift_rpm = params.get("shift_rpm", 12100)

        # GEARBOX
        self.gear_ratios = params.get("gear_ratios", np.array([
            3.20, 2.40, 1.90, 1.60, 1.40, 1.25, 1.15, 1.05
        ]))
        self.final_drive = params.get("final_drive", 3.6)
        self.drivetrain_efficiency = params.get("drivetrain_efficiency", 0.92)

        # Hybrid System (MGU-K)
        self.ers_max_deploy = params.get("ers_max_deploy", 120_000)  # 120 kW
        self.ers_max_regen = params.get("ers_max_regen", 240_000)    # regen limit
        self.ers_capacity = params.get("ers_capacity", 4e6)          # 4 MJ
        self.ers_soc = params.get("ers_soc", self.ers_capacity)      # MJ
        self.ers_efficiency = params.get("ers_efficiency", 0.85)

        # MGU-H (energy harvested from turbo)
        self.mguh_max_power = params.get("mguh_max_power", 80_000)   # 80 kW typical
        self.mguh_to_battery_eff = params.get("mguh_to_battery_eff", 0.80)
        self.mguh_to_k_eff = params.get("mguh_to_k_eff", 0.75)

        # Tires
        self.mu_base = 2.4 # base friction coefficient
        self.mu_load_sensitivity = 0.0008 # load sensitivity
        self.rolling_resistance = 0.015 # rolling resistance coefficient
        self.optimal_slip_angle = 6 # degrees

        # Brakes
        self.front_brake_force = params.get("front_brake_force", 12_000)
        self.rear_brake_force = params.get("rear_brake_force", 10_000)
        self.brake_bias = params.get("brake_bias", 0.56)
        self.brake_fade_coeff = params.get("brake_fade_coeff", 0.01)
        self.brake_cooling_efficiency = params.get("brake_cooling_efficiency", 0.8)

        # Suspension (EVOLVABLE)
        self.front_ride_height = params.get("front_ride_height", 0.03)
        self.rear_ride_height = params.get("rear_ride_height", 0.05)

        self.front_spring_rate = params.get("front_spring_rate", 95000)  # N/m
        self.rear_spring_rate = params.get("rear_spring_rate", 105000)

        self.antiroll_front = params.get("arb_front", 60_000)            # Nm/rad
        self.antiroll_rear = params.get("arb_rear", 55_000)

        self.heave_stiffness = params.get("heave_stiffness", 180_000)
        self.roll_stiffness = params.get("roll_stiffness", 145_000)

        # Chassis dimensions
        self.wheelbase = params.get("wheelbase", 3.6)
        self.track_width = params.get("track_width", 1.6)
        self.cg_height = params.get("cg_height", 0.33)

    def get_engine_torque(self, rpm):
        return np.interp(rpm, self.rpm_curve, self.torque_curve)
    
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

    def tire_mu(self, normal_load):
        return self.mu_base + self.mu_load_sensitivity * (normal_load - (self.total_mass * self.g) / 4)

    def max_brake_force(self, v):
        aero_drag = self.drag_force(v)
        total_brake = (self.front_brake_force + self.rear_brake_force) + aero_drag
        return total_brake / self.total_mass

    def mguk_power_available(self):
        """Return deployable MGU-K power depending on SOC."""
        if self.ers_soc <= 0.01 * self.ers_capacity:
            return 0
        return self.ers_max_deploy
    
    def mguk_energy_use(self, dt):
        """Energy consumed during deployment."""
        used = self.ers_max_deploy * dt * self.ers_efficiency
        self.ers_soc = max(0, self.ers_soc - used)

    def mguh_harvest(self, turbo_speed, dt):
        """
        MGU-H harvests energy from the turbo.
        Turbo speed influences harvest power.
        """
        harvest_power = min(self.mguh_max_power, turbo_speed * 0.002)
        harvested = harvest_power * dt * self.mguh_to_battery_eff
        self.ers_soc = min(self.ers_capacity, self.ers_soc + harvested)

    def max_corner_speed(self, curvature, v_guess=50):
        """
        v_max = sqrt(mu * (mass*g + downforce) / curvature)
        """
        if curvature < 1e-8:
            return 500.0

        df = self.downforce(v_guess)
        normal_force = (self.total_mass * self.g + df)
        mu = self.tire_mu(normal_force / 4)

        return math.sqrt(mu * normal_force / curvature)
    
    def max_brake_decel(self, v):
        aero_drag = self.drag_force(v)
        mechanical_brake = self.front_brake_force + self.rear_brake_force
        return (mechanical_brake + aero_drag) / self.total_mass
