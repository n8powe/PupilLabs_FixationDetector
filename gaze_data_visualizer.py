
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pathlib
import json
import sys
import math
import time
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageEnhance
import PIL.Image

import PIL


class Gaze_Visualizer():

    def __init__ (self, subject_folder_path, gaze_folder_name = "000", source="Core"):

        self.subject_folder_path = subject_folder_path
        self.gaze_folder_path = subject_folder_path + '/' + gaze_folder_name
        self.eye_track_source = source

    def read_gaze_data(self, export_number="000", world_video_path=0):
            '''This function finds all of the paths and reads in all of the data for the gaze.
            It will assume that the useable export is from path 000 unless otherwise specified.'''
            print ("Reading gaze data from ", self.subject_folder_path)
            gaze_data_path = self.gaze_folder_path

            print ("Eye tracking source is: ", self.tracking_source)

            if self.tracking_source=="Core":
                if os.path.exists(gaze_data_path+'/exports') and os.path.isdir(gaze_data_path+'/exports'):
                    print ("Gaze data exists and was exported correctly.")
                else:
                    print (gaze_data_path+'/exports')
                    raise ValueError("Gaze data was not exported. Export from pupil player.")

                self.gaze_positions_data = pd.read_csv(self.gaze_folder_path + '/exports/'+export_number+'/gaze_positions.csv')
                self.gaze_positions_data_path = self.gaze_folder_path + '/exports/'+export_number+'/gaze_positions.csv'
                self.pupil_positions_data = pd.read_csv(self.gaze_folder_path + '/exports/'+export_number+'/pupil_positions.csv')
                self.pupil_positions_data_path = self.gaze_folder_path + '/exports/'+export_number+'/pupil_positions.csv'


                self.world_timestamps = pd.read_csv(self.gaze_folder_path + '/exports/'+export_number+'/world_timestamps.csv')
                

                if world_video_path == 0:
                    self.world_video = cv2.VideoCapture(self.gaze_folder_path + '/world.mp4')
                    self.world_video_path = self.gaze_folder_path + '/world.mp4'
                else:
                    self.world_video = cv2.VideoCapture(world_video_path)
                    self.world_video_path = world_video_path

                self.pupil_depth_video_path = self.gaze_folder_path + '/exports/'+export_number+'/depth.mp4'
                self.pupil_depth_video = cv2.VideoCapture(self.pupil_depth_video_path)

                with pathlib.Path(self.gaze_folder_path+"/").joinpath("info.player.json").open() as file:
                    meta_info = json.load(file)

                start_timestamp_unix = meta_info["start_time_system_s"]
                start_timestamp_pupil = meta_info["start_time_synced_s"]
                start_timestamp_diff = start_timestamp_unix - start_timestamp_pupil
                
                self.gaze_positions_data["pupil_timestamp_unix"] = self.gaze_positions_data ["gaze_timestamp"] + start_timestamp_diff
                self.gaze_positions_data["gaze_timestamp"] = self.gaze_positions_data["gaze_timestamp"] - self.gaze_positions_data["gaze_timestamp"][0]

                self.gaze_positions_data["Frames"] = self.gaze_positions_data["world_index"]
                self.gaze_positions_data_Frame_avg = self.gaze_positions_data.groupby("world_index").mean()

                self.pupil_positions_data["pupil_timestamp_unix"] = (self.pupil_positions_data ["pupil_timestamp"] + start_timestamp_diff)*1000
                
                self.pupil_positions_data["Frames"] = self.pupil_positions_data["world_index"]
                self.pupil_positions_data_world_Frame = self.pupil_positions_data.groupby("world_index").mean()

                #self.pupil_positions_data = self.pupil_positions_data.sort_values(by=["pupil_timestamp"])

                self.pupil_frame_one_unix_timestamp = self.gaze_positions_data["pupil_timestamp_unix"][0]*1000

                self.number_world_video_frames = int(self.world_video.get(cv2.CAP_PROP_FRAME_COUNT))

                self.world_timestamps_unix = 1000*(self.world_timestamps["# timestamps [seconds]"] + start_timestamp_diff)
                self.world_timestamps["# timestamps [seconds]"] = self.world_timestamps["# timestamps [seconds]"] - self.world_timestamps["# timestamps [seconds]"][0]

            elif self.tracking_source=="Neon":

                export_folder_path = self.gaze_folder_path + '/raw-data-export/'
                sub_folders = [name for name in os.listdir(export_folder_path) if os.path.isdir(os.path.join(export_folder_path, name))]


                self.gaze_positions_data = pd.read_csv(export_folder_path+sub_folders[0]+'/gaze.csv')
                self.gaze_positions_data_path = export_folder_path+sub_folders[0]+'/gaze.csv'
                '''Need a file with pupil/eye states. Isn't included in my test dataset.'''
                #self.pupil_positions_data = pd.read_csv(export_folder_path+sub_folders[0]+'/eye_states.csv')
                #self.pupil_positions_data_path = export_folder_path+sub_folders[0]+'/eye_states.csv'


                video_files = glob.glob(os.path.join(export_folder_path+sub_folders[0], '*.mp4'))

                self.world_video = cv2.VideoCapture(video_files[0])
                self.world_video_path = video_files[0]

                self.world_timestamps = pd.read_csv(export_folder_path+sub_folders[0]+'/world_timestamps.csv')

            else:
                 print("Eye tracking source given is unknown.")

            print ("Gaze Data imported correctly.")

            return self
    
    def visualize_gaze(self, plot_gaze=True, show_video=True):

        return self