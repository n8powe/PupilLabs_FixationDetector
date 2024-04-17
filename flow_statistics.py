import numpy as np
import pandas as pd
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
import os

class Flow_Statistics():

    def __init__(self, flow_path, subject_folder_path, imu_path="", gaze_folder_name = "000"):
        
        
        self.subject_folder_path = subject_folder_path
        self.gaze_folder_path = subject_folder_path + '/' + gaze_folder_name

        self.flow_path = flow_path
        self.IMU_path = imu_path
        self.flow_folder = self.subject_folder_path+"/"+self.flow_path

        #with open(self.subject_folder_path+"/"+flow_path, 'rb') as handle:
            #b = pkl.load(handle)
        #    self.flow = pkl.load(handle)#self.subject_folder_path+"/"+flow_path)

        ##Maybe save the flow data out from the other script, and then read it in here instead. 

    def calculate_marginal_flow_statistics(self):
        marginal_flow_stats = {}

        ## Decompose flow into velocity magnitude and direction

        magnitude, angle = cv2.cartToPolar(self.flow[...,0], self.flow[...,1])


        ## Average the velocity magnitude


        ## Average the velocity direction

        return marginal_flow_stats

    def calculate_flow_statistics_across_visual_field(self):
        flow_folder = self.flow_folder
        flow_file_names = os.listdir(flow_folder)
        flow_stats = {}
        example_frame = np.load(self.flow_folder+'/'+flow_file_names[0])
        
        empty_velocity_mat = np.zeros([example_frame.shape[0], example_frame.shape[1]])
        empty_angle_mat = np.zeros([example_frame.shape[0], example_frame.shape[1]])

        num_flow_frames = len(flow_file_names)

        ## Decompose flow into velocity magnitude and direction
        

        for f in flow_file_names:
            flow = np.load(self.flow_folder+'/'+f)
            print (flow.shape)
            magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

            empty_velocity_mat = empty_velocity_mat + magnitude
            empty_angle_mat = empty_angle_mat + angle

        ## Average the velocity magnitude

        flow_stats["avg_velocity"] = empty_velocity_mat/num_flow_frames

        ## Average the velocity direction
        ## I think this part is incorrect? Might need to figure out a better way of determining the distribution of speed directions.
        flow_stats["avg_direction"] = empty_angle_mat/num_flow_frames


        with open('flow_stats.pickle', 'wb') as handle: 
            pkl.dump(flow_stats, handle, protocol=pkl.HIGHEST_PROTOCOL)


        return flow_stats
    
    def calculate_differential_motion_parallax(self):
        parallax = 0

        ## Create filter to pass over the magnitude and direction representations of the flow

        return parallax
    
    def combine_information_with_IMU_data(self):
        combined_data = 0
        imu_data = pd.read_csv(self.IMU_path)
        ## Convert timestamps to unix time for imu data. Is in a weird form during recording.
        return combined_data

subject_folder = "Subject7"
gaze_folder = "001"
flow_path = "Flow"
stats1 = Flow_Statistics(flow_path=flow_path, subject_folder_path=subject_folder, gaze_folder_name=gaze_folder)

avgs = stats1.calculate_flow_statistics_across_visual_field()

plt.imshow(avgs["avg_velocity"])
plt.show()
