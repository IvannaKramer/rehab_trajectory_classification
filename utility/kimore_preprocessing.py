import csv
import numpy as np
from argparse import ArgumentParser
from os import getcwd
from os import walk
from os.path import expanduser
from os.path import join
import pandas as pd

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from shutil import copyfile

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
CLASSES = ['Parkinson', 'Stroke', 'Healthy', 'Backpain', 'Expert']
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



def convert_trajectories(store_in_same_dir=True,
						 dir_pattern='Raw',
						 store_as_rgb=True):
	path = _getArgs().input_dir
	training_fodler = _getArgs().output_dir
	plotted_once = False

	for (dirpath, dirnames, filenames) in walk(path):
		if dirpath.endswith(dir_pattern):
			for f in filenames:
				if f.startswith(DATA_NAMES[0]) and f.endswith('.csv'):
					print(dirpath)
					print(f)
					path_to_file = join(dirpath, f)
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
					_class = get_pathology_from_path(dirpath)
					f = _class + '_' + f
					image_filename = join(training_fodler, f)#

					matplotlib.image.imsave(image_filename, _trajectories,\
														  vmin=0, vmax=255)
						
	
	if not store_in_same_dir:
		path = _getArgs().output_dir

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

def get_pathology_from_path(path):
	_class = ''
	for p in CLASSES:

		if path.find(p) != -1:
			_class =  p
	if len(_class) < 2:
		_class = 'Healthy'
	return _class




def kinect_positions_to_xyz_(positions):
	x,y,z = [], [] , []
	for ind in BODY_2_ID.values():
		x.append(positions[ind])
		y.append(positions[ind+1])
		z.append(positions[ind+2])
	print(len(x))
	return x,y,z



def create_train_test_dirs(path):
	image_names = []
	full_path_images = []

	for (dirpath, dirnames, filenames) in walk(path):
		if dirpath.endswith(dir_pattern):
			for f in filenames:
				if f.endswith('.png'):#
					image_names.append(f)
					full_path_images.append()
					print(f)
						
	try:
	    copyfile(source, target)
	except IOError as e:
	    print("Unable to copy file. %s" % e)
	    exit(1)
	except:
	    print("Unexpected error:", sys.exc_info())
	    exit(1) 



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
    convert_trajectories()
