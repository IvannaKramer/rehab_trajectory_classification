import csv
import numpy as np
from argparse import ArgumentParser
from os import getcwd
from os import walk
from os import makedirs
from os.path import expanduser
from os.path import join
import pandas as pd
import random
import errno

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from shutil import copy
from shutil import move


DATA_NAMES = ['Joint_Positions', 'JointOrientation']
BIN_CLASSES = ['Healthy', 'Pathology']
DIR_NAME = 'Rch_Sd2Sd_Bck'
JOINT_NUMBER = 50
PATHOLOGY = []
HEALTHY = []


def fill_classes():
	for i in range(1,10):
		cl = 'H0' + str(i)
		HEALTHY.append(cl)

	HEALTHY.append('H10')

	for i in range(1,10):
		cl = 'P0' + str(i)
		PATHOLOGY.append(cl)



def _getArgs():
    parser = ArgumentParser(description='Convert joint trajectories ')
    parser.add_argument('-i', '--input_dir', metavar='DIR',\
    					default='../sample_data/toronto/data_new',\
    					help='Directory with trajectory data to be converted')
    parser.add_argument('-o', '--output_dir', metavar='DIR', \
    					default='../training_data/toronto', \
    					help='Directory where converted data are converted')
    return parser.parse_args()



def convert_trajectories(npzs_only, normalized, dir_pattern='Raw',
						 store_as_rgb=True):
	path = _getArgs().input_dir
	training_fodler = _getArgs().output_dir
	plotted_once = False
	broken_files= []
	training_files = []

	for (dirpath, dirnames, filenames) in walk(path):
		print(dirpath)
		if True: #dirpath.find(DIR_NAME) > -1:
			for f in filenames:
				if f.startswith(DATA_NAMES[0]) and f.endswith('.csv'):
					path_to_pos= join(dirpath, f)
				
					try:
						pos_df = pd.read_csv(path_to_pos,
									 index_col=False)

						pos_trajectories = []

						for ind in range(pos_df.shape[0]):
							positions = pos_df.iloc[ind]
							pos_trajectories.append(positions)

						pos_trajectories_np = np.array(pos_trajectories) 
						print(pos_trajectories_np.shape)													
						axis1_length = int(pos_trajectories_np.shape[0] / JOINT_NUMBER)

						pos_trajectories_np =  np.expand_dims(pos_trajectories_np, axis=1)
						print(pos_trajectories_np.shape)
						diff = JOINT_NUMBER * axis1_length
						ind_list = []
						for ind in range( diff, pos_trajectories_np.shape[0]):
							ind_list.append(ind)
						
						pos_trajectories_np = np.delete(pos_trajectories_np, ind_list, axis=0)
						pos_trajectories_np = pos_trajectories_np.reshape((axis1_length, JOINT_NUMBER, 3))
						pos_trajectories_np.transpose((1, 0, 2))
						print(pos_trajectories_np.shape)
						
					#	cl = get_class_from_path(dirpath)
						cl = dirpath.split('/')[-2] + '_' + dirpath.split('/')[-1]
						print(cl)
						f = cl + '_' + f.replace('csv', 'png')
						image_filename = join(training_fodler, f)
						print('image_filename=', image_filename)
					#	print(pos_trajectories_np)
						pos_trajectories_np -= np.min(pos_trajectories_np)
						pos_trajectories_np /= np.max(pos_trajectories_np)
						matplotlib.image.imsave(image_filename, pos_trajectories_np,\
													  			vmin=0, vmax=255)
						training_files.append(image_filename)

					except pd.errors.ParserError:
						print('Broken file=', join(dirpath, f_pos))
						broken_files.append(join(dirpath, f_pos))
					except IsADirectoryError:
						print('No orientation file found')
						broken_files.append(join(dirpath))
									
	print(broken_files)
	return training_files

def extract_joints(positions):
	joints = []

	for ind in BODY_2_ID.values():
		joint = []
		joint.append(positions[ind])
		joint.append(positions[ind+1])
		joint.append(positions[ind+2])
		joints.append(joint)
	joints_np = np.array(joints)
	#print(joints_np.shape)

	return joints_np 



def get_class_from_path(path):
	file_name = path.split('/')[-1]
	cl = file_name.split('_')[0]

	if cl in HEALTHY:
		return 'Healthy'
	elif cl in PATHOLOGY:
		return 'Pathology'
	else:
		return 'NoClass'



def get_class_from_class(path):
	if path.find('Expert') != -1:
		return 'Healthy'
	else:
		return 'Pathology'




def kinect_positions_to_xyz_(positions):
	x,y,z = [], [] , []
	print(positions.shape)
	for ind in BODY_2_ID.values():
		x.append(positions[ind])
		y.append(positions[ind+1])
		z.append(positions[ind+2])
	print(len(x))
	return x,y,z

#list of length 1000 wirh (25,3) array
def extract_body_part_lists(positions):
	print(positions)
	joint_ids = []
	for i in range(0, 25):
		joint_ids.append(i)
	x = []
	y = []
	z = []
	for mot in positions:
		_x = []
		_y = []
		_z = []
		for pos in mot:
			print('Pos')
			print(pos)
			_x.append(pos[0])
			_y.append(pos[1])
			_z.append(pos[2])
	
		x.append(_x)
		y.append(_y)
		z.append(_z)

	x_np = np.array(x)
	y_np = np.array(y)
	z_np = np.array(z)
	return x_np, y_np, z_np
			


def kinect_pos_to_xyz_np(joints):
	x,y,z = [], [] , []
	for pos in joints:
		x.append(pos[0])
		y.append(pos[1])
		z.append(pos[2])

	return x,y,z



def create_train_test_dirs(train_files, test_files ):
	train_dir = join(_getArgs().output_dir, 'train')
	test_dir = join(_getArgs().output_dir, 'test')

	for cl in BIN_CLASSES:	
		try:
			i_train_dir = join(train_dir, cl)
			i_test_dir = join(test_dir, cl)
			makedirs(i_train_dir)
			makedirs(i_test_dir)
			copy_files(i_train_dir, train_files, cl)
			copy_files(i_test_dir, test_files, cl)
		except OSError as e:
		    if e.errno != errno.EEXIST:
		        raise

	#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")


def copy_files(dest, training_files, cl):
	for file_src in training_files:
		_cl = get_class_from_path(file_src)
		
		if _cl.endswith(cl):

			file_name = file_src.split('/')[-1]
			print(file_src)
			print(cl)
			file_dest = join(dest, file_name)
			print('dest=', file_dest)
			copy(file_src, file_dest)
			    


def divide_test_train(training_files, testing_rate=0.2):
	assert len(training_files)>0
	train_files = []
	test_files = []
	test_number = int(len(training_files) *\
							 testing_rate)
	test_files = random.sample(training_files, \
							test_number)
	train_files = [im for im in training_files\
						if im not in set(test_files)]
	create_csv_from_list('training.csv',train_files)
	create_csv_from_list('test.csv', test_files)

	return train_files, test_files



def get_images():
	images = []
	img_folder = _getArgs().output_dir
	for (dirpath, dirnames, filenames) in walk(img_folder):
			for f in filenames:
				if f.endswith('.png'):
					images.append(join(dirpath, f))
	return images
					



def create_csv_from_list(name, files):
	with open(name, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(files)


def get_images():
	images = []
	img_folder = _getArgs().output_dir
	for (dirpath, dirnames, filenames) in walk(img_folder):
			for f in filenames:
				if f.endswith('.png'):
					images.append(join(dirpath, f))
	return images



if __name__ == '__main__':
    args = _getArgs()
    print(args.input_dir)
    fill_classes()
    training_files = convert_trajectories(False, False)
    #training_files = get_images()
    train, test = divide_test_train(training_files)
   # print(test)
   # print(train)
    create_train_test_dirs(train, test)

# 58.62
# acc/val_acc1': tensor(86.2069