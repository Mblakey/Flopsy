import sys
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def display_usage():
	eprint("draw_boundary.py <options> <file> <output>")
	eprint("<options>")
	eprint("-v         - verbose statements on")
	eprint("-c <color> - choose flourescense color (green,pink...etc)")
	eprint("-j <int>   - set pixel jump limit (default 100)")
	eprint("-o <int>   - raise line from boundary (default 25)")
	sys.exit(1)


def process_command_line(arguements, arguement_count):
	global filename
	global outfile
	global color 
	global pixel_jump_limit
	global line_offset

	filename = None
	outfile = None
	color = "green"
	pixel_jump_limit = 100
	line_offset = 25

	global flag_verbose

	flag_verbose = False

	if(arguement_count < 2):
		eprint("Error: no enough args")
		display_usage()

	j = 0
	for i in range(1,len(arguements)):
		arg = arguements[i]
		if(arg[0] == '-'):
			if(arg[1] == 'v'):
				flag_verbose = True
			elif(arg[1] == 'c'):
				if i == len(arguements)-1:
					eprint("Error: must specify a color for flourescense")
					display_usage()
				else:
					i += 1
					color = arguements[i]
					i += 1
			elif(arg[1] == 'j'):
				if i == len(arguements)-1:
					eprint("Error: must specify a number for pixel jump")
					display_usage()
				else:
					i += 1
					pixel_jump_limit = int(arguements[i])
					i += 1

			elif(arg[1] == 'o'):
				if i == len(arguements)-1:
					eprint("Error: must specify a number for line raise")
					display_usage()
				else:
					i += 1
					line_offset = int(arguements[i])
					i += 1

			else:
				eprint(f"Error: unrecognised option - '{arg}'")
				display_usage()

		else:
			if j == 0:
				filename = arg
				j += 1
			elif j == 1:
				outfile = arg
				j += 1
		
	if filename == None or outfile == None:
		eprint("Error: file path are not set correctly")
		display_usage()


	return True


# this will return the y position, col index is x position
def first_set_bit(col):
	for idx,bit in enumerate(col):
		if bit == True:
			return idx
	return None

def last_set_bit(col):
  iter = None
  for idx,bit in enumerate(col):
  	if bit == True:
  		iter = idx;
  return iter

def boundary_image(filename):

	img = cv2.imread(filename)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	if(flag_verbose):
		eprint(f"targeting {color} flourescense")

	# floursence color!

	if color == "green":
		thresh = cv2.inRange(img, (90, 90 , 0), (100, 255,255))   # GREEN
	elif color == "pink":
		thresh = cv2.inRange(img, (70, 0 , 70), (255, 100,255)) # PINK
	else:
		eprint(f"Error: color {color} is not yet supported")
		display_usage()

	mask = (thresh > 0)
	sharp_mask = np.zeros(mask.shape) # draw the boundary to
	num_cols = mask.shape[1]

	if(flag_verbose):
		eprint(f"sharpening mask - jump limit = {pixel_jump_limit}")


	# logic here is that if not found, continue horizontally
	prev_y = None
	not_written = []
	for col_index in range(0,mask.shape[1]):
		col = mask[:, col_index]
		y = last_set_bit(col)
		x = col_index

		## Fine tuning! 
		if y != None:
			y += -line_offset
			if prev_y != None and abs(y - prev_y) > pixel_jump_limit:
				y = prev_y;
			else:
				prev_y = y;

			sharp_mask[y,x] = 1
			if len(not_written) > 0:
				for term in not_written:
					sharp_mask[y,term] = 1
				not_written.clear()

		elif prev_y != None:
			sharp_mask[prev_y,x] = 1
			prev_y = prev_y
		else:
			not_written.append(x)


	if(flag_verbose):
		eprint("interpolating spline on mask")

	 # Find the column indices of non-zero pixels in each row
	non_zero_indices = np.where(sharp_mask > 0)

	# Create a set of points (x, y) for the non-zero pixels
	points = np.column_stack((non_zero_indices[1], non_zero_indices[0]))

	# Sort the points based on x-coordinate
	points = points[np.argsort(points[:, 0])]

	# Fit a spline through the points
	spline = UnivariateSpline(points[:, 0], points[:, 1], k=3, s=2)

	#spline = InterpolatedUnivariateSpline(points[:, 0], points[:, 1]) # seems to be much faster?? sometimes?

	# Generate a set of x values to evaluate the spline
	x_values = np.linspace(points[:, 0].min(), points[:, 0].max(), 1000)

	# Evaluate the spline to get corresponding y values
	y_values = spline(x_values)

	if(flag_verbose):
		eprint("smoothing coordinates")

	# Round the coordinates and draw the smooth line on the image
	smooth_line_coordinates = np.column_stack((np.round(x_values).astype(int), np.round(y_values).astype(int)))

	img_copy = img.copy()
	cv2.polylines(img_copy, [smooth_line_coordinates], isClosed=False, color=255, thickness=10)
	cv2.imwrite(outfile, img_copy)


	if(flag_verbose):
		eprint(f"output saved to {outfile}")

	return True

if __name__ == "__main__":
    process_command_line(sys.argv, len(sys.argv));
    boundary_image(filename)


    
