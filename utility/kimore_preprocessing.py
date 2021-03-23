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
				'Thumb_Right':98
			}

DATA_NAMES = ['JointPosition', 'JointOrinetation']
CLASSES = ['Parkinson', 'Stroke', 'BackPain', 'Expert', 'NotExpert']
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
			for f in filenames:
				if f.startswith(DATA_NAMES[0]) and f.endswith('.csv'):
					print(dirpath)
					print(f)
					path_to_file = join(dirpath, f)
					try:
						seq_df = pd.read_csv(path_to_file,
									 index_col=False)
						trajectories = []
						for ind in range(seq_df.shape[0]):
							positions = seq_df.iloc[ind]
							#print(positions.shape)	 
							joints = extract_positions(positions)
							trajectories.append(joints)						
						_trajectories = np.array(trajectories)
						_trajectories = np.swapaxes(_trajectories, 0,1)
						print(_trajectories.shape)
						if normalized:
							_trajectories -= np.min(_trajectories)
							_trajectories /= np.max(_trajectories)	
						#_trajectories -= np.min(_trajectories)
						#_trajectories /= np.max(_trajectories)	
						if plotted_once:
							print(trajectories)
							plt.imshow(_trajectories, interpolation='nearest',\
															 vmin=0, vmax=255)
							plt.show()		
							plotted_once = False
					
						_class = get_class_from_path(dirpath)
						npz_f = f.replace('csv', 'npz')

						if not npzs_only:
							f = f.replace('csv', 'png')						
							f = _class + '_' + f
							image_filename = join(training_fodler, f)
							matplotlib.image.imsave(image_filename, _trajectories,\
															  vmin=0, vmax=255)
						
						npz_f = _class + '_' + npz_f
						npz_filename = join(training_fodler, npz_f)
						training_files.append(npz_filename)

						npz =  np.savez(npz_filename, name1=_trajectories)
						#npz =  np.savez('mat.npz', name1=arr1)


					except pd.errors.ParserError:
						print('Broken file=', join(dirpath, f))
						broken_files.append(join(dirpath, f))

							
	print(broken_files)
	return training_files

def extract_positions(positions):
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
	for p in CLASSES:
		if path.find(p) != -1:
			_class =  p
	if len(_class) < 2:
		_class = 'NoClass'
	return _class



def kinect_positions_to_xyz_(positions):
	x,y,z = [], [] , []
	for ind in BODY_2_ID.values():
		x.append(positions[ind])
		y.append(positions[ind+1])
		z.append(positions[ind+2])
	print(len(x))
	return x,y,z



def create_train_test_dirs(train_files, test_files, npz_only):
	train_dir = join(_getArgs().output_dir, 'train')
	test_dir = join(_getArgs().output_dir, 'test')

	for cl in CLASSES:	
		try:
			if not npz_only:
				_train_dir = join(train_dir + '/img/', cl)
				_test_dir = join(test_dir + '/img/', cl)
				makedirs(_train_dir)
				makedirs(_test_dir)
				copy_files(_train_dir, train_files, cl)
				copy_files(_test_dir, test_files, cl)
			
			n_train_dir = join(train_dir + '/npz/', cl)
			n_test_dir = join(test_dir + '/npz/', cl)
			makedirs(n_train_dir)
			makedirs(n_test_dir)
			copy_files(n_train_dir, train_files, cl, imgs=False)
			copy_files(n_test_dir, test_files, cl, imgs=False)

		except OSError as e:
		    if e.errno != errno.EEXIST:
		        raise

	#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")


def copy_files(dest, training_files, cl, imgs=True):
    print('dest=', dest)
    for npz_src in training_files:
	    if npz_src.find(cl) > -1:
		    npz_name = npz_src.split('/')[-1]

		    if imgs:
		        im_name = npz_name.replace('npz', 'png')
		        im_src = npz_src.replace('npz', 'png')
		        print('dest=', dest)
		        #im_dest = join(dest, cl)

		        im_dest = join(dest, im_name)
		        print('im_src=', im_src)
		        print('im_dist=', im_dest)
		        copy(im_src, im_dest)
		    else:
		    	npz_dest = join(dest, npz_name)
		    	copy(npz_src, npz_dest)



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
					



def visualize_skeleton(x,y,z):
	fig=go.Figure(go.Scatter3d(x=x,
                               y=y,
                               z=z, 
                               mode='lines', 
                               line_width=2, 
                               line_color='blue'))
	fig.update_layout(width=600, height=600)
	fig.show()

def create_csv_from_list(name, files):
	with open(name, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(files)


if __name__ == '__main__':
    args = _getArgs()
    print(args.input_dir)
    training_files = convert_trajectories(True, True)
    train, test = divide_test_train(training_files)
    create_train_test_dirs(train, test, npz_only=False)
