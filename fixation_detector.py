
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
import pickle as pkl
#matplotlib.use("Agg") # This should make it so this will work on mac. If it messes something up on PC, comment it out. 


class Fixation_Detector():

    def __init__ (self, subject_folder_path, gaze_folder_name = "000"):

        self.subject_folder_path = subject_folder_path
        self.gaze_folder_path = subject_folder_path + '/' + gaze_folder_name

    def read_gaze_data(self, export_number="000", world_video_path=0):
        '''This function finds all of the paths and reads in all of the data for the gaze.
        It will assume that the useable export is from path 000 unless otherwise specified.'''
        print ("Reading gaze data.")
        gaze_data_path = self.gaze_folder_path

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

        print ("Gaze Data imported correctly.")

        return self
    
    def find_fixation_updated(self, maxVel=60, minVel=15, maxAcc=10, pupil_file_path="pupil_positions.csv", eye_id=0, method="3d c++"):
            
            def Dot     (arr, width=1): return np.einsum( 'ij,ij->i', arr[:-width,:], arr[width:,:] )
            def Angle   (arr, width=1): return np.insert( np.arccos( Dot(arr, width) ),     0, np.zeros(width), axis=0 )
            def Diff    (arr, order=1): return np.insert( np.diff(arr, n=order),            0, np.zeros(order), axis=0 )
            def Rad2Deg (data):         return np.multiply(data, 360/(np.pi*2) )

            class Local:
                def __init__(self, arr, width): 
                    self.arr    = arr
                    self.width  = width
                    self.local  = [np.roll(self.arr, i) for i in range(-self.width, self.width+1)]
                    self.mean   = np.nanmean(    self.local, axis=0 )
                    self.median = np.nanmedian(  self.local, axis=0 )
                    self.std    = np.nanstd(     self.local, axis=0 )
                    self.min    = np.nanmin(     self.local, axis=0 )
                    self.max    = np.nanmax(     self.local, axis=0 )

                def Local(arr, width):    return [np.roll(arr, i) for i in range(-width, width+1)]

                def Mean(arr, width):     return np.nanmean(    Local.Local(arr, width), axis=0 )
                def Median(arr, width):   return np.nanmedian(  Local.Local(arr, width), axis=0 )
                def Std(arr, width):      return np.nanstd(     Local.Local(arr, width), axis=0 )
                def Min(arr, width):      return np.nanmin(     Local.Local(arr, width), axis=0 )
                def Max(arr, width):      return np.nanmax(     Local.Local(arr, width), axis=0 )
            
            def Load(path, eye_id = 0, method="pye3d 0.3.0 post-hoc"):
                data    = pd.read_csv(path)
                data["pupil_timestamp"] = data["pupil_timestamp"] - data["pupil_timestamp"][0]
                
                index   = (data.eye_id == eye_id) * (data.method == method)
                #print (index)
                raw = {}
                raw['time']   = np.array(list(data.pupil_timestamp[index])) #- np.array(data.pupil_timestamp[index][0])
                filt_sz = 2
                # Uncomment the next line if you want to smooth the vector components
                #raw['vec']    = np.column_stack([Local.Mean(data.circle_3d_normal_x[index], width=filt_sz), Local.Mean(data.circle_3d_normal_y[index], width=filt_sz), Local.Mean(data.circle_3d_normal_z[index], width=filt_sz)])
                raw['vec']    = np.column_stack([data.circle_3d_normal_x[index], data.circle_3d_normal_y[index], data.circle_3d_normal_z[index]])
                
                raw['frame'] = np.array(list(data.world_index[index]))

                return raw
            
            def find_eye_velocities_in_pixel_space(self, eye_id, method):
                '''This function finds the velocity of the eye in eye video space, normalized between 0-1'''
                index   = (self.pupil_positions_data.eye_id == eye_id) * (self.pupil_positions_data.method == method)
                x_pos_norm = Local.Mean(self.pupil_positions_data.norm_pos_x[index], width=2)#*400/2
                y_pos_norm = 1-Local.Mean(self.pupil_positions_data.norm_pos_y[index], width=2)#*400/2) # We recorded at 400x400 pixels. Subtracting from 1 gives the correct Y position. 

                dt     = Diff(self.pupil_positions_data.pupil_timestamp[index])
                t = self.pupil_positions_data.pupil_timestamp[index] - self.pupil_positions_data.pupil_timestamp[0]

                dx = (((Diff(x_pos_norm)/39/dt)))**2
                dy = (((Diff(y_pos_norm)/39/dt)))**2

                velocity_magnitude = Local.Mean(np.sqrt(dx + dy), width=2)
                acceleration = Diff(velocity_magnitude)/dt

                condition = velocity_magnitude<1000 # Filter out large values

                self.pupil_velocity_eye_vid = velocity_magnitude[condition]

                self.pupil_acceleration_eye_vid = np.abs(acceleration[condition])

                self.dt_pupil_data = t[condition]#dt[condition]

                self.pupil_frames = self.pupil_positions_data.world_index[index]
                self.pupil_frames = self.pupil_frames[condition]

                self.pupil_x_pos_norm = x_pos_norm[condition]
                self.pupil_y_pos_norm = y_pos_norm[condition]

                #print (velocity_magnitude)

                t_forplotting = t[condition]

                fig, ax = plt.subplots(3)
                ax[0].plot(t_forplotting, x_pos_norm[condition], 'r')
                ax[0].plot(t_forplotting, y_pos_norm[condition], 'b')
                ax[1].plot(t_forplotting, self.pupil_velocity_eye_vid, 'r')
                ax[2].plot(t_forplotting, self.pupil_acceleration_eye_vid, 'b')
                ax[2].set_ylim(0,2000)

                #plt.show()
                plt.close()
                
                

                return self

            if pupil_file_path=="pupil_positions.csv":
                data = Load(self.pupil_positions_data_path, eye_id=0, method=method)
            else:
                data = Load(pupil_file_path, eye_id=0, method=method)

            find_eye_velocities_in_pixel_space(self, eye_id, method)

            fix = {}
            
            dt     = Diff(data['time'])
            ang    = Angle( data['vec'] ) * (180/np.pi)
            

            fix['vel']    = ang / dt
            condition = fix['vel']<500
            velocity = fix['vel'][condition]          ######## Change filter size or function if desired. I set it to 10, but that is kind of long. 
            velocity = Local.Mean(velocity, width=2)  # small filter?
            
            acceleration    = np.gradient(velocity, axis=0) / dt[condition]
            

            self.gaze_velocity = velocity
            self.gaze_acceleration = acceleration
            self.gaze_time = data['time'][condition] 

            high    = velocity > maxVel
            low     = velocity > minVel
            acc     = acceleration < maxAcceleration

            good=[]
            last = np.roll(high, 1)
            for i in range(len(high)):
                h = high[i]
                l = low[i]

                if last[i]: good += [l]
                else:       good += [h]
            
            fix['isSaccade'] = good

            good = np.abs(np.asarray(good)-1)

            vel = velocity < maxVel

            #good = vel*acc*1


            #good = good*(acc*1)
            self.fixation_bool = good*maxVel

            #plt.close()
            #plt.plot(data['time'][condition], velocity, 'r')
            #plt.plot(data['time'][condition], high * 10, 'b')
            #plt.plot(data['time'][condition], self.fixation_bool, 'g')
            #plt.xlim([1600, 1620])
            #plt.ylim([-1, 100])
            #plt.show()

            dGood   = Diff(np.array(good).astype(np.int8))

            start   = np.where(dGood==1)[0].tolist()
            end     = np.where(dGood==-1)[0].tolist()

            if len(start) > len(end):
                start = start[0:len(start)-1]
            elif len(start) < len(end):
                end = end[0:len(end)-1]
            
            fix['index']        = np.column_stack([start,end]).tolist()
            fix['framespan']    = np.diff(fix['index'],axis=1)[:,0].tolist()
            fix['timespan']     = data['time'][end] - data['time'][start]

            frames = data["frame"][condition]
            
            alignWithWorldVideo = pd.DataFrame()
            alignWithWorldVideo["Frame"] = frames
            alignWithWorldVideo["FrameSaved"] = frames
            alignWithWorldVideo["Good"] = good*1
            alignWithWorldVideo = alignWithWorldVideo.groupby("Frame").mean()
            good = np.ceil(alignWithWorldVideo["Good"])
            self.fixation_frame_world = np.ceil(alignWithWorldVideo["FrameSaved"])

            

            self.world_frame_fixation_bool = np.asarray(np.int32(good))

            fix_index = np.asarray(fix['index']  )
            out_data = {'StartFrame': fix_index[:,0], 
                        'EndFrame': fix_index[:,1], 
                        'FrameSpan': fix['framespan'],
                        'TimeSpan': fix['timespan']}
            
            out_df = pd.DataFrame(out_data)

            out_df.to_csv(self.subject_folder_path+"/FixationData.csv", index=False)

            print ("Done finding fixations from pupil csv data. Saved to subject folder.")
            self.fixations = out_df

            return self
    
    def create_fixation_tracking_video(self, track_fixations=False, tracking_window=10):
        '''This function creates a video that tracks the initial position of a fixation in camera space to cut down on jitter. The pixel coordinates are
        then saved, and so is the video.'''

        print ("Starting fixation tracking.")

        def add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=True, color="r", fix=""):

            gaze_data = self.gaze_positions_data

            X = np.median(gaze_data.norm_pos_x[gaze_data.world_index == frame_number])
            Y = (1-np.median(gaze_data.norm_pos_y[gaze_data.world_index == frame_number]))

            timestamp = self.world_timestamps["# timestamps [seconds]"][frame_number]#gaze_data.gaze_timestamp[gaze_data.world_index == frame_number]
            
            if X == np.nan or Y == np.nan:
                X = 1
                Y = 1

            if len(gaze_data.norm_pos_x[gaze_data.world_index == frame_number])==0:
                X = 0
                Y = 0

            #X = np.mean(gaze_data_frame.norm_pos_x)
            #Y = np.mean(gaze_data_frame.norm_pos_y)

            width = frame.shape[0]
            height = frame.shape[1]

            #print (frame_number, X*height, Y*width)

            center_coordinates = (int(X*height), int(Y*width))  # Change these coordinates as needed
            radius = 10
            if color=="g":
                color = (0, 0, 255)  # Green color in BGR
            elif color=="r":
                color = (0, 255, 0)

            thickness = 2

            if plot_gaze:
                frame_with_circle = cv2.circle(frame, center_coordinates, radius, color, thickness)
            else:
                frame_with_circle = frame

            cv2.putText(frame, fix, (5, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            #print (int(X*height), int(Y*width))
            return (int(X*height), int(Y*width), frame_with_circle, timestamp)
        
        def create_graph_segment(self, frame_number, timestamp, frame_height, frame_width, num_frames, fps, time_frame=1.0):
            
            time = self.gaze_time
            frames = np.linspace(0, num_frames, len(time))
            velocity = self.gaze_velocity
            acceleration = self.gaze_acceleration

            #total_frames = self.world

            #frame_number = frame_number*(fps)/30

            #print (frame_number, timestamp, velocity[frame_number], acceleration[frame_number])

            timesync = fps*time_frame

            fig, ax = plt.subplots(3)

            ax[2].plot(time, self.gaze_velocity, 'r')
            #ax[2].plot(time, self.gaze_acceleration, 'b')
            ax[2].plot(time, self.fixation_bool, 'g')
            
            if not math.isnan(timestamp):
                ax[2].vlines(timestamp, 0, 1000, color='gray', alpha=0.5, linewidth=20)
                ax[2].vlines(timestamp, 0, 1000, color='k', linewidth=2)
                ax[2].set_xlim(timestamp-time_frame,timestamp+time_frame)
            else:
                print ("NaN time")
            ax[2].set_ylim(0,200)
            ax[2].set_ylabel("Pupil Angular Velocity")

            ax[0].plot(self.pupil_frames, self.pupil_x_pos_norm)
            ax[0].plot(self.pupil_frames, self.pupil_y_pos_norm)
            ax[0].vlines(frame_number, 0, 1000, color='gray', alpha=0.5, linewidth=20)
            ax[0].vlines(frame_number, 0, 1000, color='k', linewidth=2)
            ax[0].set_xlim(frame_number-timesync, frame_number+timesync)
            ax[0].set_ylim(0.3,0.65)
            ax[0].set_ylabel("Pupil Norm. Position Eye Cam.")


            ax[1].plot(self.pupil_positions_data_world_Frame["circle_3d_normal_x"]-np.mean(self.pupil_positions_data_world_Frame["circle_3d_normal_x"]), "r")
            ax[1].plot(self.pupil_positions_data_world_Frame["circle_3d_normal_y"]-np.mean(self.pupil_positions_data_world_Frame["circle_3d_normal_y"]), "g")
            ax[1].plot(self.pupil_positions_data_world_Frame["circle_3d_normal_z"]-np.mean(self.pupil_positions_data_world_Frame["circle_3d_normal_z"]), "b")
            ax[1].vlines(frame_number, -1000, 1000, color='gray', alpha=0.5, linewidth=20)
            ax[1].vlines(frame_number, -1000, 1000, color='k', linewidth=2)
            ax[1].set_xlim(frame_number-timesync, frame_number+timesync)
            ax[1].set_ylim(-0.08,0.08)
            ax[1].set_ylabel("Eye Vector Norm. Position")

            #plt.legend(["Velocity", "Acceleration", "Fixation (Yes/No)"])
            #plt.xlim([624000,624010])

            #middle_point = int(data['time'][int(len(data['time'])/2) ])
            #upper_point = int(middle_point + 10)

            #axs[1].set_xlim([middle_point, upper_point])

            #gaze_vec = np.asarray(data['vec'])
            #axs[0].plot(data['time'], gaze_vec[:,0],'r')
            #axs[0].plot(data['time'], gaze_vec[:,1],'b')
            #axs[0].plot(data['time'], gaze_vec[:,2],'g')

            #axs[0].set_xlim([middle_point, upper_point])
            #axs[0].set_xlim(timestamp-0.25,timestamp+0.25)

            #plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(8, 7.2)
            
            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # Define the size of the graph image
            graph_height, graph_width, _ = graph_img.shape

            # Create a blank canvas to place the graph image
            blank_canvas = np.zeros((frame_height, graph_width, 3), dtype=np.uint8)

            # Place the graph image on the blank canvas
            blank_canvas[:graph_height, :] = graph_img

            plt.close()

            return blank_canvas

        cap = self.world_video
        fixations = self.fixations
        gaze_window = tracking_window

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Unable to open the video file.")
            exit()

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        outvideo = cv2.VideoWriter(self.subject_folder_path+'/FixationTracking_withgraph.mp4', fourcc, fps, (2080, frame_height))

        frame_number = 0 
        fixation_index = 0
        success = False

        tracked_gaze_positions = []

        number_of_fixations = len(fixations["EndFrame"])-1
        prevBool = 0
        # Process frames to track the object
        print ("Video processing starting.")
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            if frame_number >= len(self.world_frame_fixation_bool):
                break
            print (frame_number, self.world_frame_fixation_bool[frame_number])
            booleanIndexWorldFrames = np.where(self.fixation_frame_world == frame_number)[0]
            if self.world_frame_fixation_bool[booleanIndexWorldFrames] == 1: #frame_number>=fixations["StartFrame"][fixation_index] and frame_number<=fixations["EndFrame"][fixation_index]:
                
                if track_fixations:
                    if (self.world_frame_fixation_bool[booleanIndexWorldFrames] == 1) and (prevBool==0):
                        gaze_x, gaze_y, frame, timestamp = add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=False)
                        
                        if gaze_x < 0 or gaze_y < 0:
                            bbox = (0, 0, gaze_window, gaze_window)
                        else:
                            bbox = (gaze_x-gaze_window, gaze_y-gaze_window, gaze_window*2, gaze_window*2)#cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
                            
                        print ("box", bbox)
                        tracker = cv2.legacy.TrackerMOSSE_create()
                        #print (bbox, tracker, frame.shape)
                        tracker.init(frame, bbox)
                        
                    # Update the tracker
                    success, bbox = tracker.update(frame)

                    # Draw bounding box around the tracked object
                    if success:
                        (x, y, w, h) = [int(i) for i in bbox]
                        gaze_position = (x+w/2 , y+h/2)
                        gaze_x_tracked = gaze_position[0]
                        gaze_y_tracked = gaze_position[1]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        print ("Unsuccesful point tracking")

                gaze_x, gaze_y, frame, timestamp = add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=True, color="g", fix="Fixation")


            else:
                gaze_x, gaze_y, frame, timestamp = add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=True, color="r", fix="No Fixation")
                
            
            if track_fixations and success:
                gaze_position = (frame_number, gaze_x_tracked, gaze_y_tracked)
            else:
                gaze_position = (frame_number, gaze_x, gaze_y)

            

            blank_canvas = create_graph_segment(self, frame_number, self.world_timestamps["# timestamps [seconds]"][frame_number], frame_height,frame_width, total_frames, fps)

            cv2.putText(frame, str(np.round(frame_number)), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            combined_frame = np.hstack((frame, blank_canvas))
            #print (combined_frame.shape)

            # Display the frame
            cv2.imshow("Frame", combined_frame)

            outvideo.write(combined_frame)

            tracked_gaze_positions.append(gaze_position)

            frame_number = frame_number + 1
            prevBool = self.world_frame_fixation_bool[booleanIndexWorldFrames]

            if frame_number>fixations["EndFrame"][fixation_index]:
                fixation_index = fixation_index + 1

            if frame_number == total_frames:
                break
        
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        np.save(self.subject_folder_path+"/trackedGazePositions", tracked_gaze_positions)
        # Release video capture object
        outvideo.release()
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print ("Done fixation tracking in video.")
        return self

    def create_fixation_tracking_video_updated(self, track_fixations=False, tracking_window=10, start_frame=1):
        '''This function creates a video that tracks the initial position of a fixation in camera space to cut down on jitter. The pixel coordinates are
        then saved, and so is the video.'''

        print ("Starting fixation tracking.")

        def add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=True, color="r", fix=""):

            gaze_data = self.gaze_positions_data

            X = np.median(gaze_data.norm_pos_x[gaze_data.world_index == frame_number])
            Y = (1-np.median(gaze_data.norm_pos_y[gaze_data.world_index == frame_number]))

            timestamp = self.world_timestamps["# timestamps [seconds]"][frame_number]#gaze_data.gaze_timestamp[gaze_data.world_index == frame_number]
            
            if X == np.nan or Y == np.nan:
                X = 1
                Y = 1

            if len(gaze_data.norm_pos_x[gaze_data.world_index == frame_number])==0:
                X = 0
                Y = 0

            #X = np.mean(gaze_data_frame.norm_pos_x)
            #Y = np.mean(gaze_data_frame.norm_pos_y)

            width = frame.shape[0]
            height = frame.shape[1]

            #print (frame_number, X*height, Y*width)

            center_coordinates = (int(X*height), int(Y*width))  # Change these coordinates as needed
            radius = 10
            if color=="g":
                color = (0, 0, 255)  # Green color in BGR
            elif color=="r":
                color = (0, 255, 0)

            thickness = 2

            if plot_gaze:
                frame_with_circle = cv2.circle(frame, center_coordinates, radius, color, thickness)
            else:
                frame_with_circle = frame

            cv2.putText(frame, fix, (5, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            #print (int(X*height), int(Y*width))
            return (int(X*height), int(Y*width), frame_with_circle, timestamp)
        
        def create_graph_segment(self, frame_number, timestamp, frame_height, frame_width, num_frames, fps, time_frame=1.0):
            
            time = self.gaze_time
            frames = np.linspace(0, num_frames, len(time))
            velocity = self.gaze_velocity
            acceleration = self.gaze_acceleration

            #total_frames = self.world

            #frame_number = frame_number*(fps)/30

            #print (frame_number, timestamp, velocity[frame_number], acceleration[frame_number])

            timesync = fps*time_frame

            fig, ax = plt.subplots(3)

            ax[2].plot(time, self.gaze_velocity, 'r')
            #ax[2].plot(time, self.gaze_acceleration, 'b')
            ax[2].plot(time, self.fixation_bool, 'g')
            
            if not math.isnan(timestamp):
                ax[2].vlines(timestamp, 0, 1000, color='gray', alpha=0.5, linewidth=20)
                ax[2].vlines(timestamp, 0, 1000, color='k', linewidth=2)
                ax[2].set_xlim(timestamp-time_frame,timestamp+time_frame)
            else:
                print ("NaN time")
            ax[2].set_ylim(0,200)
            ax[2].set_ylabel("Pupil Angular Velocity")

            ax[0].plot(self.pupil_frames, self.pupil_x_pos_norm)
            ax[0].plot(self.pupil_frames, self.pupil_y_pos_norm)
            ax[0].vlines(frame_number, 0, 1000, color='gray', alpha=0.5, linewidth=20)
            ax[0].vlines(frame_number, 0, 1000, color='k', linewidth=2)
            ax[0].set_xlim(frame_number-timesync, frame_number+timesync)
            ax[0].set_ylim(0.3,0.65)
            ax[0].set_ylabel("Pupil Norm. Position Eye Cam.")

            ## Update this to visualize phi and theta instead -- Just to double check what it is doing. Also change the ylim on this (thesevalues) to investigate them further.
            
            #ax[1].plot(self.pupil_positions_data_world_Frame["theta"]-np.mean(self.pupil_positions_data_world_Frame["theta"]), "r")
            #ax[1].plot(self.pupil_positions_data_world_Frame["circle_3d_normal_y"]-np.mean(self.pupil_positions_data_world_Frame["circle_3d_normal_y"]), "g")
            ax[1].plot(self.pupil_positions_data_world_Frame["circle_3d_normal_x"]-np.mean(self.pupil_positions_data_world_Frame["circle_3d_normal_x"]), "r")
            ax[1].plot(self.pupil_positions_data_world_Frame["circle_3d_normal_z"]-np.mean(self.pupil_positions_data_world_Frame["circle_3d_normal_z"]), "g")
            #ax[1].plot(self.pupil_positions_data_world_Frame["ellipse_angle"], "b")
            ax[1].vlines(frame_number, -1000, 1000, color='gray', alpha=0.5, linewidth=20)
            ax[1].vlines(frame_number, -1000, 1000, color='k', linewidth=2)
            ax[1].set_xlim(frame_number-timesync, frame_number+timesync)
            ax[1].set_ylim(-0.08,0.08)
            ax[1].set_ylabel("Eye Vector Norm. Position")

            #plt.legend(["Velocity", "Acceleration", "Fixation (Yes/No)"])
            #plt.xlim([624000,624010])

            #middle_point = int(data['time'][int(len(data['time'])/2) ])
            #upper_point = int(middle_point + 10)

            #axs[1].set_xlim([middle_point, upper_point])

            #gaze_vec = np.asarray(data['vec'])
            #axs[0].plot(data['time'], gaze_vec[:,0],'r')
            #axs[0].plot(data['time'], gaze_vec[:,1],'b')
            #axs[0].plot(data['time'], gaze_vec[:,2],'g')

            #axs[0].set_xlim([middle_point, upper_point])
            #axs[0].set_xlim(timestamp-0.25,timestamp+0.25)

            #plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(8, 7.2)
            
            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # Define the size of the graph image
            graph_height, graph_width, _ = graph_img.shape

            # Create a blank canvas to place the graph image
            blank_canvas = np.zeros((frame_height, graph_width, 3), dtype=np.uint8)

            # Place the graph image on the blank canvas
            blank_canvas[:graph_height, :] = graph_img

            plt.close()

            return blank_canvas

        cap = self.world_video
        fixations = self.fixations
        gaze_window = tracking_window

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Unable to open the video file.")
            exit()

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        outvideo = cv2.VideoWriter(self.subject_folder_path+'/FixationTracking_withgraph.mp4', fourcc, fps, (2080, frame_height))

        frame_number = 0 + start_frame
        fixation_index = 0

        tracked_gaze_positions = []

        number_of_fixations = len(fixations["EndFrame"])-1
        prevBool = 0
        # Process frames to track the object
        print ("Video processing starting.")
        while cap.isOpened():
            ret, frame = cap.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = PIL.Image.fromarray(img)
            new_image = ImageEnhance.Contrast(im_pil).enhance(1.5)
            new_image = ImageEnhance.Sharpness(new_image).enhance(2.0)
            frame = np.asarray(new_image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if not ret:
                break
            
            if frame_number >= len(self.world_frame_fixation_bool):
                break
            print (frame_number, self.world_frame_fixation_bool[frame_number])
            booleanIndexWorldFrames = np.where(self.fixation_frame_world == frame_number)[0]
            if self.world_frame_fixation_bool[booleanIndexWorldFrames] == 1: #frame_number>=fixations["StartFrame"][fixation_index] and frame_number<=fixations["EndFrame"][fixation_index]:
                
                if track_fixations:
                    if (self.world_frame_fixation_bool[booleanIndexWorldFrames] == 1) and (prevBool==0):
                        gaze_x, gaze_y, frame, timestamp = add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=False)
                        
                        if gaze_x < 0 or gaze_y < 0:
                            bbox = (0, 0, gaze_window, gaze_window)
                        else:
                            bbox = (gaze_x-gaze_window, gaze_y-gaze_window, gaze_window*2, gaze_window*2)#cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
                            #print (bbox)
                        tracker = cv2.legacy.TrackerMOSSE_create()
                        #print (bbox, tracker, frame.shape)
                        tracker.init(frame, bbox)
                        
                    # Update the tracker
                    success, bbox = tracker.update(frame)

                    # Draw bounding box around the tracked object
                    if success:
                        (x, y, w, h) = [int(i) for i in bbox]
                        gaze_position = (x+w/2 , y+h/2)
                        gaze_x_tracked = gaze_position[0]
                        gaze_y_tracked = gaze_position[1]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        print ("Unsuccesful point tracking")

                gaze_x, gaze_y, frame, timestamp = add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=True, color="g", fix="Fixation")


            else:
                gaze_x, gaze_y, frame, timestamp = add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=True, color="r", fix="No Fixation")
                
            
            if track_fixations and success:
                gaze_position = (frame_number, gaze_x_tracked, gaze_y_tracked)
            else:
                gaze_position = (frame_number, gaze_x, gaze_y)
            

            blank_canvas = create_graph_segment(self, frame_number, self.world_timestamps["# timestamps [seconds]"][frame_number], frame_height,frame_width, total_frames, fps)

            cv2.putText(frame, str(np.round(frame_number)), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            combined_frame = np.hstack((frame, blank_canvas))
            print (combined_frame.shape)

            # Display the frame
            cv2.imshow("Frame", combined_frame)

            outvideo.write(combined_frame)

            tracked_gaze_positions.append(gaze_position)

            frame_number = frame_number + 1
            prevBool = self.world_frame_fixation_bool[booleanIndexWorldFrames]

            if frame_number>fixations["EndFrame"][fixation_index]:
                fixation_index = fixation_index + 1

            if frame_number == total_frames:
                break
        
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        np.save(self.subject_folder_path+"/trackedGazePositions", tracked_gaze_positions)
        # Release video capture object
        outvideo.release()
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print ("Done fixation tracking in video.")
        return self

    def estimate_optic_flow(self, gaze_centered=False,only_show_fixations=True, start_frame=0, visualize_as="color", 
                            use_tracked_fixations=False, output_flow=False, output_centered_video=False, overwrite_flow_folder=False, 
                            resize_frame_scale="",remove_padding=False, padding_window_removed=200, use_CUDA=True):
        self.use_tracked_fixations = use_tracked_fixations
        self.overwrite_flow_folder = overwrite_flow_folder

        def smooth_pixel_gaze(self, filter_size=5):
            filter_ = np.ones(filter_size)/filter_size
            self.gaze_positions_data['norm_pos_x'] = np.convolve(self.gaze_positions_data['norm_pos_x'], filter_, "same")
            self.gaze_positions_data['norm_pos_y'] = np.convolve(self.gaze_positions_data['norm_pos_y'], filter_, "same")
            return self

        def preprocess_frame(frame):

            def create_sky_mask_on_hsv(hsv_in):
                lower = np.array([100, 45, 100])
                upper = np.array([120, 140, 260])

                mask = cv2.inRange(hsv_in, lower, upper)
                mask = 255 - mask

                return mask

            def create_road_mask_on_hsv(hsv_in):
                lower = np.array([0, 240, 200])
                upper = np.array([40, 255, 255])
                mask1 = cv2.inRange(hsv, lower, upper)

                lower = np.array([170, 200, 100])
                upper = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv, lower, upper)

                mask = mask1 + mask2
                mask = np.clip(mask, 0, 255).astype(np.uint8)

                mask = cv2.dilate(mask, kernel=np.ones((5, 5), np.uint8), iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
                mask = 255 - mask

                return mask

            def remove_noise_bgr_dark_patches(bgr_image):
                _, mask = cv2.threshold(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY), 30, 250, cv2.THRESH_TOZERO)
                bgr_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
                return bgr_image

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #frame = cv2.bitwise_and(frame, frame, mask=create_sky_mask_on_hsv(hsv))
            #frame = cv2.bitwise_and(frame, frame, mask=create_road_mask_on_hsv(hsv))

            #frame = remove_noise_bgr_dark_patches(frame)

            return frame
        def apply_magnitude_thresholds_and_rescale(self, magnitude, lower_mag_threshold=False, upper_mag_threshold=False):

            if lower_mag_threshold:
                magnitude[magnitude < lower_mag_threshold] = 0

            if upper_mag_threshold:
                magnitude[magnitude > upper_mag_threshold] = upper_mag_threshold

            magnitude = (magnitude / upper_mag_threshold) * 255.0

            return magnitude

        def visualize_flow_as_hsv(self, magnitude, angle):
            '''
            Note that to perform well, this function really needs an upper_bound, which also acts as a normalizing term.
            
            '''

            # create hsv output for optical flow
            hsv = np.zeros([np.shape(magnitude)[0], np.shape(magnitude)[1], 3], np.uint8)

            hsv[..., 0] = angle * 180 / np.pi / 2 # angle_rads -> degs 0-360 -> degs 0-180
            hsv[..., 1] = 255
            hsv[..., 2] = magnitude
            # cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            hsv_8u = np.uint8(hsv)
            bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

            return bgr
        
        def convert_flow_to_magnitude_angle(self,flow,
                                            bgr_world_in,
                                            lower_mag_threshold = False,
                                            upper_mag_threshold = False):

            magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

            # Clips to lower-upper thresh and then rescales into range 0-255
            magnitude = apply_magnitude_thresholds_and_rescale(self, magnitude,
                                                                    lower_mag_threshold=lower_mag_threshold,
                                                                    upper_mag_threshold=upper_mag_threshold)

            # A custom filter.  Mine masks out black patches in the input video.
            # Due to compression, they can be filled with noise.
            #processed_bgr_world = preprocess_frame(bgr_world_in)

            #processed_gray_world = cv2.cvtColor(processed_bgr_world, cv2.COLOR_BGR2GRAY)

            #_, mask = cv2.threshold(processed_gray_world, 10, 255, cv2.THRESH_TOZERO)

            #magnitude = cv2.bitwise_and(magnitude, magnitude, mask=mask)
            #angle = cv2.bitwise_and(angle, angle, mask=mask)

            return magnitude, angle
        
        def visualize_flow_as_vectors(self, image_in, magnitude, angle, skippts=15, scale=.1,scale_units='width',
                                width=.003, return_image=True):

            dpi = 100
            
            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            fig, ax = plt.subplots(figsize=(image_in.shape[1]*px, image_in.shape[0]*px))

            canvas = FigureCanvas(fig)
            ax.margins(0)

            s = slice(None, None, skippts)

            xmax = np.shape(image_in)[1]
            xpoints = int(np.shape(image_in)[1])
            x = np.linspace(0, np.shape(image_in)[1], xmax)

            ymax = np.shape(image_in)[0]
            ypoints = int(np.shape(image_in)[0])
            y = np.linspace(0, np.shape(image_in)[0], ymax)

            x = x[s]
            y = y[s]
            x2d, y2d = np.meshgrid(x, y, indexing='xy')

            u = magnitude * np.cos(angle) * -1
            v = magnitude * np.sin(angle)

            u = (u[s, s] / 255) * scale
            v = (v[s, s] / 255) * scale

            plt.imshow(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
            ax.axis('off')

            plt.quiver(x2d, y2d, u, v, color='red', alpha=0.7, width=width, scale=scale,
                    scale_units='inches')

            #if return_image:
            canvas.draw()  # draw the canvas, cache the renderer
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close('all')

            return cv2.cvtColor(image_from_plot,cv2.COLOR_BGR2RGB)
        
        
        def create_gaze_centered_frame(self, frame, frame_index, padding_size, remove_padding=False, padding_window_removed=200):
            '''Probably need to figure out how to do fixation detection and point tracking on the image over the duration of the fixation.'''
            height = np.shape(frame)[0]
            width = np.shape(frame)[1]
            
            #if os.path.isfile(self.subject_folder_path + "/trackedGazePositions.npy"):
            #    gaze_data = np.load(self.subject_folder_path + "/trackedGazePositions.npy")
            #    median_x = gaze_data[0,frame_number]
            #    median_y = gaze_data[1,frame_number]
            #else:
            gaze_data = self.gaze_positions_data

            #gaze_data = gaze_data[gaze_data.confidence>0.90]
            if self.use_tracked_fixations:
                #print (self.tracked_fixation_data)
                #tracked_fixation_data_row = np.where(self.tracked_fixation_data[0,:]==frame_index)
                tracked_fixation_data = self.tracked_fixation_data
                #print (tracked_fixation_data)
                median_x = tracked_fixation_data.X[frame_index]/width
                median_y = tracked_fixation_data.Y[frame_index]/height
            else:
                median_x = np.nanmedian(gaze_data.norm_pos_x[gaze_data.world_index == frame_index])
                median_y = (1-np.nanmedian(gaze_data.norm_pos_y[gaze_data.world_index == frame_index]))


            
            #booleanIndexWorldFrames = np.where(self.fixation_frame_world == frame_index)[0]
            #fixation = self.world_frame_fixation_bool[booleanIndexWorldFrames]


            if not (frame_index in gaze_data.world_index) and not remove_padding:
                return np.zeros((height*padding_size, width*padding_size, 3), np.uint8)
            elif not (frame_index in gaze_data.world_index) and remove_padding:
                return np.zeros((padding_window_removed*2, padding_window_removed*2, 3), np.uint8)
            print (median_x, median_y)

            if median_x < 0 :
                median_x = np.NAN
            elif median_y < 0:
                median_y = np.NAN
            elif median_x > 1:
                median_x = np.NAN
            elif median_y > 1:
                median_y = np.NAN

            if (np.isnan(median_x) or np.isnan(median_y)) and not remove_padding:
                return np.zeros((height*padding_size, width*padding_size, 3), np.uint8)
            elif (np.isnan(median_x) or np.isnan(median_y)) and remove_padding:
                return np.zeros((padding_window_removed*2, padding_window_removed*2, 3), np.uint8)

            new_image = np.zeros((height*padding_size,width*padding_size,3), np.uint8)

            new_image_2 = np.zeros((height*padding_size,width*padding_size), np.uint8)

            print (new_image.shape)

            center_x = (width * padding_size)/2.0
            center_y = (height * padding_size)/2.0
            medianpix_x = int(median_x * width)
            medianpix_y = int(median_y * height)

            x1 = int(center_x - medianpix_x) #- 44 ### WHYYYYY is it not matched exactly -- Need to test whether this changes based on camera. (Realsense Vs Pupil Scene)
            #x2 = int(center_x + width - medianpix_x)
            x2 = int(center_x + (width - medianpix_x)) #- 44 
            #y1 = int(center_y + medianpix_y)
            y1 = int(center_y - medianpix_y) #- 5
            #y2 = int(center_y + height - medianpix_y)
            y2 = int(center_y + (height - medianpix_y)) #- 5 

            #frame = cv2.circle(frame, (medianpix_x, medianpix_y), 20, (0, 255, 0), 2)

            #print (frame_index, height, width, medianpix_x, medianpix_y)
            #print (x1, x2, y1, y2)
            ones_frame2 = np.ones([height, width])

            print (new_image[ y1:y2,x1:x2,:].shape, new_image_2[ y1:y2,x1:x2].shape, frame.shape, ones_frame2.shape)

            new_image[ y1:y2,x1:x2,:] = frame
            new_image_2[ y1:y2,x1:x2] = ones_frame2
            #print (new_image.shape)
            #new_image = cv2.line(new_image,(0, int(new_image.shape[0]/2)), (int(new_image.shape[1]),int(new_image.shape[0]/2)), (250,0,0),2)
            #new_image = cv2.line(new_image,(int(new_image.shape[1]/2), 0), (int(new_image.shape[1]/2),int(new_image.shape[0])), (250,0,0),2)
            # remove padding
            if remove_padding:
                center_size = padding_window_removed
                #height = new_image.shape[0]/2
                #width = new_image.shape[1]/2
                new_image = new_image[ height-center_size:height+center_size, width-center_size:width+center_size,:]
                ones_frame = new_image_2[ height-center_size:height+center_size, width-center_size:width+center_size]
                self.total_frames_at_pixel = self.total_frames_at_pixel + ones_frame
            
            return new_image
        

         # Create a CUDA-enabled VideoCapture object
        cap = cv2.VideoCapture(self.world_video_path)
        ret, prev = cap.read()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #cap.set(1, start_frame-1)

        if self.use_tracked_fixations:
            self.tracked_fixation_data = pd.DataFrame(np.load(self.subject_folder_path+"/trackedGazePositions.npy"))
            self.tracked_fixation_data = self.tracked_fixation_data.rename(columns={0: "Frame", 1: "X", 2: "Y"})


        if output_flow:
            '''This function will only work is opencv was installed with CUDA/GPU support is install on the PC. I will add functionality so that it uses a more general optic flow algorithm.'''
           
            if use_CUDA:
                # Create optical flow object
                #optical_flow = cv2.cuda_OpticalFlowDual_TVL1.create()
                #optical_flow.setNumScales(15) 
                #optical_flow.setLambda(0.0004) 
                #optical_flow.setScaleStep(0.6)
                #optical_flow.setEpsilon(0.2)
                #optical_flow.setTau(0.05)
                #optical_flow.setGamma(0.1)

                optical_flow = cv2.cuda_FarnebackOpticalFlow.create()
                print ("Using optical flow farneback ... Might be issues. If flow looks bad, use nonCUDA algorithm.")

                #optical_flow = cv2.cuda_BroxOpticalFlow.create()
                #optical_flow.setPyramidScaleFactor(2)
                #optical_flow.setSolverIterations(50)
                #optical_flow.setFlowSmoothness(1.9) # def alpha 0.197
                # self.flow_algo.setGradientConstancyImportance() # def gamma 0
                # self.flow_algo.setInnerIterations() # def 5
                # self.flow_algo.setOuterIterations() # def 150
                optical_flow.setNumLevels(30) # def 0
                optical_flow.setPyrScale(0.5) # def 0
                optical_flow.setPolySigma(10.2)
                optical_flow.setWinSize(9)
                    
                #optical_flow = cv2.cuda_DensePyrLKOpticalFlow.create()
                #optical_flow.setMaxLevel(6)
                #optical_flow.setWinSize((41, 41))
            else:
                optical_flow = cv2.optflow.createOptFlow_DeepFlow()



        if gaze_centered:
            padding_size = 2
            prev = np.zeros([prev.shape[0]*padding_size, prev.shape[1]*padding_size, 3])
            prev_original = prev

            #params = {'perfPreset':cv2.cuda.NvidiaOpticalFlow_1_0_NV_OF_PERF_LEVEL_SLOW}
            #optical_flow = cv2.cuda.NvidiaOpticalFlow_1_0_create(prev.shape[1],prev.shape[0], **params)
        else:
            padding_size = 1
            prev = np.zeros([prev.shape[0], prev.shape[1], 3])
            
            #params = {'perfPreset':cv2.cuda.NvidiaOpticalFlow_1_0_NV_OF_PERF_LEVEL_SLOW}
            #optical_flow = cv2.cuda.NvidiaOpticalFlow_1_0_create(prev.shape[1],prev.shape[0], **params)

        
        if output_flow:
            #prev_gray = cv2.cuda.cvtColor(np.float32(prev), cv2.COLOR_BGR2GRAY)
            self.flow_out_folder = self.subject_folder_path+"/Flow/"

            if os.path.isdir(self.subject_folder_path+"/Flow/") and self.overwrite_flow_folder:
                os.rmdir(self.subject_folder_path+"/Flow/")

            if not os.path.isdir(self.subject_folder_path+"/Flow/"):
                os.mkdir(self.subject_folder_path+"/Flow/")

            if resize_frame_scale != "":
                prev = cv2.resize(prev, (int(prev.shape[0]/resize_frame_scale), int(prev.shape[1]/resize_frame_scale)))

            if remove_padding:
                prev = cv2.resize(prev, (int(padding_window_removed*2), int(padding_window_removed*2)))


            prev_original = prev
            if not gaze_centered:
                out = cv2.VideoWriter(self.subject_folder_path+'/OpticFlow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (prev.shape[1], prev.shape[0]))
            else:
                if self.use_tracked_fixations:
                    if resize_frame_scale != "":
                        out = cv2.VideoWriter(self.subject_folder_path+'/GazeCenteredFlowTrackedFixations.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(prev_original.shape[1]/resize_frame_scale), int(prev_original.shape[0]/resize_frame_scale)))
                    else:
                        out = cv2.VideoWriter(self.subject_folder_path+'/GazeCenteredFlowTrackedFixations.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(prev_original.shape[1]), int(prev_original.shape[0])))
                else:
                    out = cv2.VideoWriter(self.subject_folder_path+'/GazeCenteredFlow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (prev.shape[1], prev.shape[0]))

        if output_centered_video and gaze_centered:
            out_centered = cv2.VideoWriter(self.subject_folder_path+'/GazeCenteredVideoRGB.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (prev.shape[1], prev.shape[0]))


        frame_number = 0 
        flow_dict = {}

        first_fixation_frame = False
        
        print ("Remove Padding :", remove_padding)

        if remove_padding:
            self.total_frames_at_pixel = np.zeros([padding_window_removed*2, padding_window_removed*2])
        else:
            self.total_frames_at_pixel = np.zeros([prev.shape[0]*padding_size, prev.shape[1]*padding_size])
        

        #self = smooth_pixel_gaze(self, filter_size=5)
        while True:
            # Read the next frame
            
            ret, frame = cap.read()

            if not ret:
                break
            
            if gaze_centered:
                
                booleanIndexWorldFrames = np.where(np.int32(self.fixation_frame_world) == frame_number)[0]
                print (frame_number, booleanIndexWorldFrames, self.world_frame_fixation_bool[booleanIndexWorldFrames])

                if self.world_frame_fixation_bool[booleanIndexWorldFrames] == 1 and self.world_frame_fixation_bool[booleanIndexWorldFrames-1] == 0:
                    first_fixation_frame = True
                    prev = np.zeros([prev_original.shape[0], prev_original.shape[1], 3])
                    if resize_frame_scale != "":
                        prev = cv2.resize(prev, (int(prev.shape[0]/resize_frame_scale), int(prev.shape[1]/resize_frame_scale)))

                if only_show_fixations:
                    if self.world_frame_fixation_bool[booleanIndexWorldFrames] == 1:
                        print (frame.shape)
                        frame = create_gaze_centered_frame(self, frame, frame_number, padding_size, remove_padding=remove_padding, padding_window_removed=padding_window_removed)
                        print ("Zero: ", frame.shape)
                    else:
                        frame = np.zeros([prev.shape[0]*padding_size, prev.shape[1]*padding_size, 3], dtype=np.uint8)
                        print ("One: ", frame.shape)
                        if remove_padding:
                            frame = np.zeros([padding_window_removed*2, padding_window_removed*2, 3], dtype=np.uint8)
                            print ("Two: ", frame.shape)
                else:
                    frame = create_gaze_centered_frame(self, frame, frame_number, padding_size, remove_padding=remove_padding, padding_window_removed=padding_window_removed)
                    print ("Three: ", frame.shape)
                
            if output_centered_video and gaze_centered:
                #cv2.putText(frame, str(np.round(frame_number)), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.imshow("Centered Video", frame)
                out_centered.write(frame)
            #print (prev_gray.shape, gray.shape)
                
            #print (prev.shape, frame.shape)

            if resize_frame_scale != "":
                prev = cv2.resize(prev, (int(prev_original.shape[1]/resize_frame_scale), int(prev_original.shape[0]/resize_frame_scale)))
                frame = cv2.resize(frame, (int(prev_original.shape[1]/resize_frame_scale), int(prev_original.shape[0]/resize_frame_scale)))

            if remove_padding:
                #prev = cv2.resize(prev, (int(padding_window_removed), int(padding_window_removed)))
                #frame = cv2.resize(frame, (int(padding_window_removed), int(padding_window_removed)))
                print ()
                
            if output_flow:
                print (prev.shape, frame.shape)

                if use_CUDA:
                    frame1_gpu = cv2.cuda_GpuMat()
                    frame1_gpu.upload(np.float32(prev))

                    frame2_gpu = cv2.cuda_GpuMat()
                    frame2_gpu.upload(np.float32(frame))
                    prev_gray = cv2.cuda.cvtColor(frame1_gpu, cv2.COLOR_BGR2GRAY)
                    gray = cv2.cuda.cvtColor(frame2_gpu, cv2.COLOR_BGR2GRAY)
                    flow_gpu = optical_flow.calc(prev_gray, gray, None)
                    # Download flow matrix from GPU
                    flow = flow_gpu.download()
                else:
                    prev_gray = cv2.cvtColor(np.float32(prev), cv2.COLOR_BGR2GRAY)
                    gray = cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2GRAY)
                    flow = optical_flow.calc(prev_gray, gray, None)
                # Calculate optical flow
                # Calculate optical flow
                


                # Visualize optical flow

                flow_frame = np.array(flow) # Resolution is H x W x 2

                if self.world_frame_fixation_bool[booleanIndexWorldFrames] == 1 and self.world_frame_fixation_bool[booleanIndexWorldFrames+1] == 0:
                    print ("Last Frame of fixation")
                    flow_frame = np.zeros([flow_frame.shape[0], flow_frame.shape[1], 2])


                if output_flow:
                    if gaze_centered:
                        
                        if only_show_fixations and self.world_frame_fixation_bool[booleanIndexWorldFrames] == 1: 
                            
                            
                            if first_fixation_frame:
                                print ("First Fixation Frame. Not saving flow.")
                            else:
                                np.save(self.flow_out_folder+str(frame_number)+".npy", flow_frame)
                        #else:
                        #    continue

                        #else:

                            #np.save(self.flow_out_folder+str(frame_number)+".npy", flow_frame)
                        #with open(self.subject_folder_path+'/retinal_flow.pickle', 'wb') as handle:
                        #    pkl.dump(flow_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
                    else:
                        np.save(self.flow_out_folder+str(frame_number)+".npy", flow_frame)
                        #with open(self.subject_folder_path+'/optic_flow.pickle', 'wb') as handle:
                #    pkl.dump(flow_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


                magnitude, angle = convert_flow_to_magnitude_angle(self, flow=flow_frame, bgr_world_in=frame,
                                                                        lower_mag_threshold=0,
                                                                        upper_mag_threshold=50)
                if visualize_as=="color":
                    flow_vis = visualize_flow_as_hsv(self, magnitude, angle)#(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow)
                    cv2.imshow('Optical Flow', flow_vis)
                elif visualize_as == "vectors":
                    flow_vis = visualize_flow_as_vectors(self, frame, magnitude, angle, skippts=15, scale=.5,scale_units='width',
                                            width=.0075, return_image=True)
                    cv2.imshow('Optical Flow', flow_vis)
                else:
                    print ("Flow visualization type not defined.")
                

                #cv2.line(flow_vis,(0, int(flow_vis.shape[0]/2)), (int(flow_vis.shape[1]),int(flow_vis.shape[0]/2)), (250,0,0),2)
                #cv2.line(flow_vis,(int(flow_vis.shape[1]/2), 0), (int(flow_vis.shape[1]/2),int(flow_vis.shape[0])), (250,0,0),2)

                # Display the frame with optical flow
                
                #cv2.putText(frame, str(np.round(frame_number)), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                # Update the previous frame and previous gray frame
                if self.world_frame_fixation_bool[booleanIndexWorldFrames+1] == 0:
                    #print ([prev_original.shape[0], prev_original.shape[1], 3])
                    prev = np.zeros([prev_original.shape[0], prev_original.shape[1], 3])
                    if resize_frame_scale != "":
                        prev = cv2.resize(prev, (int(prev.shape[0]/resize_frame_scale), int(prev.shape[1]/resize_frame_scale)))
                    if remove_padding:
                        prev = np.zeros([padding_window_removed*2, padding_window_removed*2, 3])
                else:
                    prev = frame#.copy()
                #flow_vis = cv2.cvtColor(flow_vis,cv2.COLOR_RGB2BGR)
                #print (flow_vis.shape)
                out.write(flow_vis)
            
            frame_number = frame_number + 1

            first_fixation_frame = False

            print (frame_number, " out of ", total_frames)
            if frame_number >= total_frames-100:
                break
            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        
        if remove_padding:
            np.save(self.subject_folder_path + "/total_frames_at_pixel",self.total_frames_at_pixel)

        if output_flow:
            out.release()
        if output_centered_video:
            out_centered.release()

        cap.release()
        cv2.destroyAllWindows()  
        

        return self

subject_folder = "NP42324" # Add subject folder name here
gaze_folder = "000" # Within subject folder should be a gaze data folder --- something like 000 or 001
                # It will then search that for an export folder, and take export 000 from it and read in all the data
                # Find fixations will find all of the fixations and save the data to the subject folder.
                # Create fixation tracking video will create a video that shows fixations and plots the graph of the gaze velocities/accelerations next to it. 

detector = Fixation_Detector(subject_folder_path=subject_folder, gaze_folder_name=gaze_folder)

detector.read_gaze_data(export_number="000")

maxVelocity = 45  # Just change this and the next value to adjust how fixations are detected. They are angular velocity and acceleration of the eye. 
maxAcceleration = 20 # Not used

detector.find_fixation_updated(eye_id=0,maxVel=maxVelocity, minVel=10, maxAcc=maxAcceleration, method="3d c++")

#detector.create_fixation_tracking_video_updated(track_fixations=True, tracking_window=45)

# This is an optic flow estimation function, BUT it can also be used to output the retina centered video. It currently does the entire video, not breaking it up into fixations. Though this can be done easily. 
detector.estimate_optic_flow(gaze_centered=True, only_show_fixations=True, use_tracked_fixations=True,
                              output_flow=True, output_centered_video=True, visualize_as="vectors",
                                overwrite_flow_folder=True, remove_padding=True, padding_window_removed=250, use_CUDA=False)

# Add functionality so that this outputs the head centered flow too (and doesn't overwrite the retina centered flow data)

