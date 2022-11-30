import numpy as np
import pandas as pd
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from kimore_preprocessing import extract_joints
from kimore_preprocessing import kinect_pos_to_xyz_np
from kimore_preprocessing import extract_body_part_lists
import plotly.express as px
from plotly.graph_objs import *

#the start and end point for each line

Kinect_Links = [(0,1), (0,12), (0,16), (1,20), (2,3), (2,20),(4,5),(4,20),(5,6),(6,22),(7,21),\
			(8,20),(8,9),(9,10),(10,11), (11, 23), (12,13), (13,14), (14,15),(16,17),(17,18),\
			(18,19), (23,24), (21,22)]

k4vLinks = [(0,1),(1,2),(1,3),(1,7),(3,4),(4,5),(5,6),(7,8), (8,9), (9, 10)]


def readKimoreTrajectory(path, f=50):
	pos_df = pd.read_csv(path,
							 index_col=False)
	print('pos_df.shape=', pos_df.shape)
	mot_sequence = []
	position_seq = []

	for ind in range(pos_df.shape[0]):
					positions = pos_df.iloc[ind]
					joints = extract_joints(positions)
					position_seq.append(joints)
					x, y, z  = kinect_pos_to_xyz_np(joints)
					pos = [x,y,z]
					#if ind % f == 0:
					mot_sequence.append(pos)
					#plot_joints(x,z,y)
	x,y,z = extract_body_part_lists(position_seq)
	#plot_signals(x)
	#plot_joints1(mot_sequence)
	animate_joints(mot_sequence)



def readTorontoTrajectory(path, f=10):
	pos_df = pd.read_csv(path,
							 index_col=False)
	mot_sequence = []
	position_seq = []
	for ind in range(pos_df.shape[0]):
					positions = pos_df.iloc[ind]
					position_seq.append(positions)
					
					if ind > 0 and ind % 12 == 0: 
						x, y, z  = extract_toronto(position_seq)
						#plot_joints(x,z,y, links=k4vLinks)
						position_seq = []
						mot_sequence.append([x, y, z])

					if ind > 100:

						break
					#
					#pos = [x,y,z]
					#if ind % f == 0:
					#	mot_sequence.append(pos)
					#plot_joints(x,z,y)
	#x, y, z  = extract_toronto(position_seq)
	#plot_joints(x,z,y, links=k4vLinks)
	#print(position_seq)	
	#x,y,z = extract_body_part_lists(position_seq)
	#plot_signals(x)
	animate_joints(mot_sequence, links=k4vLinks)
	#plot_joints1(mot_sequence, links=k4vLinks)


def extract_toronto(positions):
	x,y,z = [], [] , []
	for pos in positions:
		x.append(pos[0])
		y.append(pos[1])
		z.append(pos[2])
	return x,y,z




def plot_signals(mot_signal):
	print(mot_signal)
	fig = px.line(mot_signal)
	ind_values = [str(x) for x in range(0, 25)]
	fig.update_layout(title=dict(
                                text= "Converted signal of Y-coordinate for each body part", 
                                x=0.5,
                                xanchor='center',
                                yanchor= 'top',
                                font=dict(
                                        family="Arial",
                                        size=26,
                                        color='#000000'
                                    )),
                   xaxis_title='Y coordinate [m]',
                   yaxis_title='Body part', )
	fig.update_xaxes(title=dict(                      
                                font = dict(
                                    family= 'Arial',
                                    size= 30,
                                    color= '#000000'
                                            ),
                                ),
                    	tickfont = dict(
                                  family = 'Arial',
                                  size = 26,
                                  color = 'black'
                                  )
                    )

	fig.update_yaxes(title=dict(                      
                                font = dict(
                                    family= 'Arial',
                                    size= 30,
                                    color= '#000000'
                                            ),
                                ),
    				ticktext = ind_values,
                    tickfont = dict(
                                  family = 'Arial',
                                  size = 26,
                                  color = 'black'
                                   )
                    )
	fig.show()

	



def plot_joints1(mot_sequence,  links=Kinect_Links):
	print('Len of mot_sequence', len(mot_sequence))
	traces = []
	opac = 1
	for motion in mot_sequence:

		x_joints = motion[0]
		y_joints = motion[2]
		z_joints = motion[1]
		
		if opac < 0.95:
			opac +=0.06

		trace1 = go.Scatter3d(
	    	x=x_joints,
	    	y=y_joints,
		    z=z_joints,
		    mode='markers',
		    name='markers', 
		    marker=dict(
	        	size=8,
	        	color=z_joints,         # set color to an array/list of desired values
	        	colorscale='Viridis',   # choose a colorscale
	        	opacity=opac
	    	)
		)

		x_lines = list()
		y_lines = list()
		z_lines = list()

		#create the coordinate list for the lines
		for p in links:
		    for i in range(2):
		        x_lines.append(x_joints[p[i]])
		        y_lines.append(y_joints[p[i]])
		        z_lines.append(z_joints[p[i]])
		    x_lines.append(None)
		    y_lines.append(None)
		    z_lines.append(None)

		trace2 = go.Scatter3d(
		    x=x_lines,
		    y=y_lines,
		    z=z_lines,
		    mode='lines',
		    name='lines'
		)

		traces.append(trace1)
	#	traces.append(trace2)

	fig = go.Figure(data=traces)

	fig.show()



def animate_joints(mot_sequence,  links=Kinect_Links):
	print('animate_joints')
	print('Len of mot_sequence', len(mot_sequence))
	traces = []
	opac = 1
	for motion in mot_sequence:

		x_joints = motion[0]
		y_joints = motion[2]
		z_joints = motion[1]
		
	#	if opac < 0.95:
    #		opac +=0.06

		trace1 = go.Scatter3d(
	    	x=x_joints,
	    	y=y_joints,
		    z=z_joints,
		    mode='markers',
		    name='markers', 
		    marker=dict(
	        	size=8,
	        	color=z_joints,         # set color to an array/list of desired values
	        	colorscale='Viridis',   # choose a colorscale
	        	opacity=opac
	    	)
		)

		x_lines = list()
		y_lines = list()
		z_lines = list()

		#create the coordinate list for the lines
		for p in links:
		    for i in range(2):
		        x_lines.append(x_joints[p[i]])
		        y_lines.append(y_joints[p[i]])
		        z_lines.append(z_joints[p[i]])
		    x_lines.append(None)
		    y_lines.append(None)
		    z_lines.append(None)

		trace2 = go.Scatter3d(
		    x=x_lines,
		    y=y_lines,
		    z=z_lines,
		    mode='lines',
		    name='lines'
		)

	#	traces.append(trace1)
		traces.append(trace2)
	frames = [go.Frame(data=trace) for trace in traces]
	fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[],
                    				  mode="markers")])


	fig.update(frames=frames)
	
	fig.update_layout(updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, dict(frame=dict(redraw=True,fromcurrent=True, mode='immediate'))])])])
	


	fig.show()


def plot_joints(x_joints, y_joints, z_joints, links=Kinect_Links):

	#x_joints = [-0.669591,-0.659245,-0.645584,-0.615228,-0.796576,-0.842263,-0.7251,-0.645984,-0.520901,-0.47787,-0.385088,-0.318132,-0.723807,-0.880991,-0.785368,-0.729103,-0.598271,-0.379349,-0.374082,-0.337952,-0.649592,-0.624523,-0.680766,-0.258188,-0.341698]
	#y_joints = [2.5413,2.51537,2.47648,2.4291,2.43307,2.3813,2.25367,2.15301,2.55371,2.5427,2.43929,2.37574,2.48484,2.34228,2.36412,2.36571,2.5328,2.41106,2.44368,2.35031,2.48851,2.12284,2.112,2.37661,2.32655]
	#z_joints = [-0.734748,-0.4378,-0.145044,-0.0567484,-0.211972,-0.385253,-0.451375,-0.462517,-0.213515,-0.442252,-0.561056,-0.594653,-0.72505,-0.93354,-1.18509,-1.19241,-0.725666,-0.904529,-1.17007,-1.20029,-0.217688,-0.472149,-0.451634,-0.639623,-0.558815]
	print(x_joints)
	trace1 = go.Scatter3d(
	    x=x_joints,
	    y=y_joints,
	    z=z_joints,
	    mode='markers',
	    name='markers', 
	    marker=dict(
        	size=8,
        	color=z_joints,         # set color to an array/list of desired values
        	colorscale='Viridis',   # choose a colorscale
        	opacity=0.8
    	)
	)

	x_lines = list()
	y_lines = list()
	z_lines = list()

	#create the coordinate list for the lines
	for p in links:
	    for i in range(2):
	        x_lines.append(x_joints[p[i]])
	        y_lines.append(y_joints[p[i]])
	        z_lines.append(z_joints[p[i]])
	    x_lines.append(None)
	    y_lines.append(None)
	    z_lines.append(None)

	trace2 = go.Scatter3d(
	    x=x_lines,
	    y=y_lines,
	    z=z_lines,
	    mode='lines',
	    name='lines'
	)

	fig = go.Figure(data=[trace1, trace2])
	fig.show()

if __name__=='__main__':
	print('Ploting')
	#readKimoreTrajectory('../sample_data/kimore/GPP/Stroke/S_ID1/Es1/Raw/JointPosition060616_122123.csv')
	readTorontoTrajectory('../sample_data/toronto/H01_Joint_Positions.csv')