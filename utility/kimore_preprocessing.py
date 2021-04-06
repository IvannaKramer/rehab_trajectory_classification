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

#ID=index in the csv line
#KINECT skelet = 25 joints
BODY_2_ID = {   'Spine_Base':0, 
				'Spine_Mid' :4,
				'Neck':8,
				'Head':12,       
				'Shoulder_Left':16,
				'Elbow_Left':20,
				'Wrist_Left':24,
				'Hand_Left':28,
				'Shoulder_Right':32,
				'Elbow_Right':36,
				'Wrist_Right':40,
				'Hand_Right':44,
				'Hip_Left':48,
				'Knee_Left':52,
				'Ankle_Left':56,
				'Foot_Left':60,    
				'Hip_Right':64,
				'Knee_Right':68,
				'Ankle_Right':72,
				'Foot_Right':76,   
				'Spine_Shoulder':80,
				'Tip_Left':84,     
				'Thumb_Left':88,   
				'Tip_Right':92,    
				'Thumb_Right':96
			}

DATA_NAMES = ['JointPosition', 'JointOrientation']
KIMORE_CLASSES = ['Parkinson', 'Stroke', 'BackPain', 'Expert', 'NotExpert']
TARGET_CLASSES = ['Parkinson', 'Stroke', 'BackPain', 'Healthy']
#TARGET_CLASSES = ['Healthy', 'Pathology']
MOVEMENT_CLASSES = ['Es1', 'Es2', 'Es3', 'Es4', 'Es5']


def _getArgs():
    parser = ArgumentParser(description='Convert joint trajectories ')
    parser.add_argument('-i', '--input_dir', metavar='DIR',\
    					default='../sample_data/kimore/',\
    					help='Directory with trajectory data to be converted')
    parser.add_argument('-o', '--output_dir', metavar='DIR', \
    					default='../training_data/kimore', \
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
		if dirpath.endswith(dir_pattern):
			f_pos, f_or  = '', ''
			for f in filenames:
				if f.startswith(DATA_NAMES[0]) and f.endswith('.csv'):
					f_pos = f
				elif f.startswith(DATA_NAMES[1]) and f.endswith('.csv'):
					f_or = f
			

			path_to_pos= join(dirpath, f_pos)
			path_to_or= join(dirpath, f_or)

			try:
				pos_df = pd.read_csv(path_to_pos,
							 index_col=False)
				or_df = pd.read_csv(path_to_or,
							 index_col=False)
				pos_trajectories = []
				or_trajectories  = []

				for ind in range(pos_df.shape[0]):
					positions = pos_df.iloc[ind]
					#print(positions.shape)	 
					joints = extract_joints(positions)
					pos_trajectories.append(joints)						
				
				for ind in range(or_df.shape[0]):
					orientations = or_df.iloc[ind]
					#print(positions.shape)	 
					joints = extract_joints(orientations)
					#joints = np.array(orientations)
					or_trajectories.append(joints)						
								
				
				_orientations =  np.array(or_trajectories)
				_orientations = np.swapaxes(_orientations, 0,1)
				print('_orientations.shape')
				print(_orientations.shape)
				
				_trajectories = np.array(pos_trajectories)
				_trajectories = np.swapaxes(_trajectories, 0, 1)
				print('_trajectories.shape')
				print(_trajectories.shape)

				if _trajectories.shape[1] != _orientations.shape[1]:
					ind_list = [] #_trajectories.shape[1] - _orientations.shape[1]
					if _trajectories.shape[1] < _orientations.shape[1]:
						for ind in range( _trajectories.shape[1],
														 _orientations.shape[1]):
							ind_list.append(ind)
						_orientations = np.delete(_orientations, ind_list, axis=1)
					
						print('new _orinetations.shape')
						print(_orientations.shape)
					else:
						for ind in range( _orientations.shape[1],
														 _trajectories.shape[1]):
							ind_list.append(ind)
						_trajectories = np.delete(_trajectories, ind_list, axis=1)
					
						print('new _trajectories.shape')
						print(_trajectories.shape)


				_trajectories = np.concatenate((_trajectories, _orientations),
																		 axis=0)
				print('_trajectories.shape')
				print(_trajectories.shape)
				
				if normalized:
					_trajectories -= np.min(_trajectories)
					_trajectories /= np.max(_trajectories)

				if plotted_once:
					print(trajectories)
					plt.imshow(_trajectories, interpolation='nearest',\
													 vmin=0, vmax=255)
					plt.show()		
					plotted_once = False
			
				_class = get_class_from_path(dirpath)#get_class_from_path(dirpath)
				npz_f = f_pos.replace('csv', 'npz')

				if not npzs_only:
					print('**********************************')
					f_pos = f_pos.replace('csv', 'png')						
					
					f_pos = _class + '_' + f_pos
					image_filename = join(training_fodler, f_pos)
					print('image_filename=', image_filename)
					matplotlib.image.imsave(image_filename, _trajectories,\
													  vmin=0, vmax=255)
				
				npz_f = _class + '_' + npz_f
				npz_filename = join(training_fodler, npz_f)
				training_files.append(npz_filename)

				npz =  np.savez(npz_filename, name1=_trajectories)

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
	_class = ''
	for p in KIMORE_CLASSES:
		if path.find(p) != -1:
			if path.find('Expert') != -1:
				_class = 'Healthy'
			else:
				_class =  p
	if len(_class) < 2:
		_class = 'NoClass'
	return _class


def get_target_from_class(path):
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
	#x_np = x_np.reshape(x_np.shape[1], x_np.shape[0])
	print(x_np.shape)
	y_np = np.array(y)
	#y_np = y_np.reshape(x_np.shape[1], x_np.shape[0])
	z_np = np.array(z)
	#z_np = z_np.reshape(x_np.shape[1], x_np.shape[0])
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

	for cl in TARGET_CLASSES:	
		try:
			
			i_train_dir = join(train_dir + '_img/', cl)
			i_test_dir = join(test_dir + '_img/', cl)
			makedirs(i_train_dir)
			makedirs(i_test_dir)
			copy_files(i_train_dir, train_files, cl)
			copy_files(i_test_dir, test_files, cl)
			
			n_train_dir = join(train_dir + '_npz/', cl)
			n_test_dir = join(test_dir + '_npz/', cl)
			print('creating test_dir=', n_test_dir)
			print('creating n_train_dir=', n_train_dir)
			makedirs(n_train_dir)
			makedirs(n_test_dir)
			copy_files(n_train_dir, train_files, cl, imgs=False)
			copy_files(n_test_dir, test_files, cl, imgs=False)

		except OSError as e:
		    if e.errno != errno.EEXIST:
		        raise

	#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")


def copy_files(dest, training_files, cl, imgs=True):
    for file_src in training_files:
	    if file_src.find(cl) > -1:
		    file_name = file_src.split('/')[-1]
		    file_dest = join(dest, file_name)

		    if imgs:
			    im_name = file_name.replace('npz', 'png')
			    im_src = file_src.replace('npz', 'png')
			    print('dest=', dest)
			    #im_dest = join(dest, cl)
			    im_dest = join(dest, im_name)
	#		    print('im_src=', im_src)
	#		    print('im_dist=', im_dest)
			    copy(im_src, im_dest)
		    else:
		#	    print('npz_src=',file_src)
	#		    print('npz_dest=',file_dest)
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



if __name__ == '__main__':
    args = _getArgs()
    print(args.input_dir)
    training_files = convert_trajectories(False, True)
    train, test = divide_test_train(training_files)
    create_train_test_dirs(train, test)
