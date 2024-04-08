
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
    
    def find_fixation_updated(self, maxVel=60, minVel=15, maxAcc=10, pupil_file_path="pupil_positions.csv", eye_id=0, method="pye3d 0.3.0 post-hoc"):
            
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
                print (index)
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

            print (good.shape)

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
                            #print (bbox)
                        tracker = cv2.legacy.TrackerKCF_create()#legacy.TrackerMOSSE_create()
                        #print (bbox, tracker, frame.shape)
                        tracker.init(frame, bbox)
                        
                    # Update the tracker
                    success, bbox = tracker.update(frame)

                    # Draw bounding box around the tracked object
                    if success:
                        (x, y, w, h) = [int(i) for i in bbox]
                        gaze_position = (x+w/2 , y+h/2)
                        gaze_x = gaze_position[0]
                        gaze_y = gaze_position[1]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        print ("Unsuccesful point tracking")

                gaze_x, gaze_y, frame, timestamp = add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=True, color="g", fix="Fixation")


            else:
                gaze_x, gaze_y, frame, timestamp = add_gaze_to_detection_video(self, frame, frame_number, plot_gaze=True, color="r", fix="No Fixation")
                
            
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


subject_folder = "Subject7" # Add subject folder name here
gaze_folder = "001" # Within subject folder should be a gaze data folder --- something like 000 or 001
                # It will then search that for an export folder, and take export 000 from it and read in all the data
                # Find fixations will find all of the fixations and save the data to the subject folder.
                # Create fixation tracking video will create a video that shows fixations and plots the graph of the gaze velocities/accelerations next to it. 

detector = Fixation_Detector(subject_folder_path=subject_folder, gaze_folder_name=gaze_folder)

detector.read_gaze_data(export_number="000")

maxVelocity = 65  # Just change this and the next value to adjust how fixations are detected. They are angular velocity and acceleration of the eye. 
maxAcceleration = 20 # Not used

detector.find_fixation_updated(eye_id=0,maxVel=maxVelocity, minVel=10, maxAcc=maxAcceleration, method="3d c++")

detector.create_fixation_tracking_video(track_fixations=False, tracking_window=10)
