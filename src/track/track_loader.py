import pandas as pd
import numpy as np

class Track:
	def __init__(self, file_path):
		self.data = pd.read_csv('data/silverstone.csv')
		self.parse_track_data()


	def parse_track_data(self):
		self.x_coord = self.data['x_m'].values
		self.y_coord = self.data['y_m'].values
		self.w_right = self.data['w_tr_right_m'].values
		self.w_left = self.data['w_tr_left_m'].values

		self.centerline = np.column_stack((self.x_coord, self.y_coord))

		dx = np.gradient(self.x_coord)
		dy = np.gradient(self.y_coord)

		tangent_mag = (dx**2 + dy**2)
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

	def get_centerline(self):
		return self.centerline
	
	def get_boundaries(self):
		# return (np.column_stack((self.x_left, self.y_left)),
		# 		np.column_stack((self.x_right, self.y_right)))
		return (self.x_left, self.y_left, self.x_right, self.y_right)
	
	def get_track_width(self):
		return self.track_width