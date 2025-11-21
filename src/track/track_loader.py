import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

class Track:
	def __init__(self, file_path):
		self.data = pd.read_csv('./data/silverstone.csv')
		self.parse_track_data()
		self.smooth_track()
		self.compute_curvature()

	def parse_track_data(self):
		self.x_coord = self.data['x_m'].values
		self.y_coord = self.data['y_m'].values
		self.w_right = self.data['w_tr_right_m'].values
		self.w_left = self.data['w_tr_left_m'].values

		self.centerline = np.column_stack((self.x_coord, self.y_coord))

		dx = np.gradient(self.x_coord)
		dy = np.gradient(self.y_coord)
		tangent_mag = np.sqrt(dx**2 + dy**2)
		tx = dx / tangent_mag
		ty = dy / tangent_mag

		nx_left = -ty
		ny_left = tx
		nx_right = ty
		ny_right = -tx

		self.x_left = self.x_coord + nx_left * self.w_left
		self.y_left = self.y_coord + ny_left * self.w_left
		self.x_right = self.x_coord + nx_right * self.w_right
		self.y_right = self.y_coord + ny_right * self.w_right

		self.track_width = self.w_left + self.w_right

	def smooth_track(self):
		# Spline smoothing of centerline and boundaries
		t = np.arange(len(self.x_coord))
		self.spline_center_x = CubicSpline(t, self.x_coord)
		self.spline_center_y = CubicSpline(t, self.y_coord)
		self.spline_left_x = CubicSpline(t, self.x_left)
		self.spline_left_y = CubicSpline(t, self.y_left)
		self.spline_right_x = CubicSpline(t, self.x_right)
		self.spline_right_y = CubicSpline(t, self.y_right)
		self.t_smooth = np.linspace(0, len(t)-1, len(t)*10)  # 10x more points

		self.x_center = self.spline_center_x(self.t_smooth)
		self.y_center = self.spline_center_y(self.t_smooth)
		self.x_left = self.spline_left_x(self.t_smooth)
		self.y_left = self.spline_left_y(self.t_smooth)
		self.x_right = self.spline_right_x(self.t_smooth)
		self.y_right = self.spline_right_y(self.t_smooth)
		self.centerline = np.column_stack((self.x_center, self.y_center))
	
	def compute_curvature(self):
        # Curvature k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
		dx = np.gradient(self.x_center)
		dy = np.gradient(self.y_center)
		ddx = np.gradient(dx)
		ddy = np.gradient(dy)
		self.curvature = (dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
		
	def get_centerline(self):
		return self.centerline
	
	def get_boundaries(self):
		# return (np.column_stack((self.x_left, self.y_left)),
		# 		np.column_stack((self.x_right, self.y_right)))
		return (self.x_left, self.y_left, self.x_right, self.y_right)

	def get_track_width(self):
		return self.track_width
	
	def get_curvature(self):
		return self.curvature