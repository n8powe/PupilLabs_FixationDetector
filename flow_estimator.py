
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



class Flow_Estimator():

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

detector = Flow_Estimator(subject_folder_path=subject_folder, gaze_folder_name=gaze_folder)

detector.read_gaze_data(export_number="000")
detector.estimate_optic_flow(gaze_centered=True, only_show_fixations=True, use_tracked_fixations=True,
                              output_flow=True, output_centered_video=True, visualize_as="vectors",
                                overwrite_flow_folder=True, remove_padding=True, padding_window_removed=250, use_CUDA=False)