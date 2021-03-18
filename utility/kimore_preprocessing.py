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

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from shutil import copyfile
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



def convert_trajectories(dir_pattern='Raw',
						 store_as_rgb=True):
	path = _getArgs().input_dir
	training_fodler = _getArgs().output_dir
	plotted_once = False
	broken_files= []
	images = []

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
						#_trajectories -= np.min(_trajectories)
						#_trajectories /= np.max(_trajectories)	
						if plotted_once:
							print(trajectories)
							plt.imshow(_trajectories, interpolation='nearest',\
															 vmin=0, vmax=255)
							plt.show()		
							plotted_once = False
					

						f = f.replace('csv', 'png')
						_class = get_class_from_path(dirpath)
						f = _class + '_' + f
						image_filename = join(training_fodler, f)
						images.append(image_filename)

						matplotlib.image.imsave(image_filename, _trajectories,\
															  vmin=0, vmax=255)
					except pd.errors.ParserError:
						print('Broken file=', join(dirpath, f))
						broken_files.append(join(dirpath, f))

							
	print(broken_files)
	return images

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



def create_train_test_dirs(train_imgs, test_imgs):
	image_names = []
	train_dir = join(_getArgs.output_dir, 'train')
	test_dir = join(_getArgs.output_dir, 'test')

	for cl in CLASSES:
		train_dir = join(images_dir, cl)
		test_dir = join(images_dir, cl)
		try:
		    makedirs(train_dir)
		    makedirs(test_dir)
		except OSError as e:
		    if e.errno != errno.EEXIST:
		        raise
	copy_img(train_dir, train_imgs)
	copy_img(test_dir, test_imgs)
		#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")



def copy_img(dest_dir, imgs):
	for im_src in imgs:
		cl = get_class_from_path(im_src)
		im_name = im_src.split('/')[-1]
		im_dest = join(dest_dir, cl)
		im_dest = join(im_dest, im_name)
		print('Coping from=', im_src, ' to=', im_dest)
		dest = shutil.copy(im_src, im_dest)



def divide_test_train(images, testing_rate=0.2):
	assert len(images)>0
	train_images = []
	test_images = []
	test_number = int(len(images) *\
							 testing_rate)
	test_images = random.sample(images, \
							test_number)
	train_images = [im for im in images\
						if im not in set(test_number)]

	print(test_images)
	print(train_images)

	return train_images, test_images



def get_images():
	images = []
	img_folder = _getArgs().output_dir
	for (dirpath, dirnames, filenames) in walk(img_folder):
			for f in filenames:
				if f.endswith('.png'):
					img_folder.append(join(dirpath, f))
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



if __name__ == '__main__':
    args = _getArgs()
    print(args.input_dir)
    images = get_images() #convert_trajectories()
    train, test = divide_test_train(images)
    create_train_test_dirs(train, test)
