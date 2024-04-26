import numpy as np
import pandas as pd
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
import os
import seaborn as sns

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

        ##Maybe save the flow data out from the other script, and then read it in here instead. -- That is what I did now. 

    def calculate_flow_statistics_across_visual_field(self, center_window=100, start_frame=0, end_frame="", showFlow=False):
        flow_folder = self.flow_folder
        self.window_size = center_window
        flow_file_names = os.listdir(flow_folder)

        if end_frame == "":
            flow_file_names = flow_file_names
        else:
            flow_file_names = flow_file_names[start_frame:end_frame]

        flow_stats = {}
        example_frame = np.load(self.flow_folder+'/'+flow_file_names[0])
        
        empty_velocity_mat = np.zeros([example_frame.shape[0], example_frame.shape[1]])
        empty_angle_mat = []
        empty_angle_avg = np.zeros([example_frame.shape[0], example_frame.shape[1]])

        bottom_middle_speeds = []
        top_middle_speeds = []
        left_middle_speeds = []
        right_middle_speeds = []

        pixel_numbers = np.load(self.subject_folder_path+'/total_frames_at_pixel.npy')

        num_flow_frames = len(flow_file_names)

        flow_stats["avg_direction"] = []

        W = int(example_frame.shape[1]/2)
        H = int(example_frame.shape[0]/2)

        ## Decompose flow into velocity magnitude and direction
        
        frame_number = 1
        for f in flow_file_names:
            flow = np.load(self.flow_folder+'/'+f)

            #flow = np.flip(flow,0)
            magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)

            #magnitude = np.flip(magnitude, 0)
            #angle = np.flip(angle, 0)

            empty_angle_avg = empty_angle_avg + angle

            angle = angle[(H-center_window):(H+center_window), (W-center_window):(W+center_window)]

            empty_velocity_mat = empty_velocity_mat + magnitude
            empty_angle_mat.append( np.ravel(angle) )

            magnitude = magnitude[(H-center_window):(H+center_window), (W-center_window):(W+center_window)]
            mag_H = int(magnitude.shape[0]/2)
            mag_W = int(magnitude.shape[1]/2)
            bottom_middle_speeds.append(magnitude[0,center_window])
            top_middle_speeds.append(magnitude[mag_H,(center_window*2)-1])
            
            left_middle_speeds.append(magnitude[center_window,0])

            right_middle_speeds.append(magnitude[(center_window*2)-1,mag_W])
            

            print ("Frame number ", frame_number, " out of ", num_flow_frames)

            frame_number = frame_number + 1


            #if frame_number == 500:
            #    break

        ## Average the velocity magnitude

        flow_stats["avg_velocity"] = empty_velocity_mat/num_flow_frames

        flow_stats["top_middle_speeds"] = top_middle_speeds
       
        flow_stats["bottom_middle_speeds"] = bottom_middle_speeds
        flow_stats["left_middle_speeds"] = left_middle_speeds
        flow_stats["right_middle_speeds"] = right_middle_speeds

        ## Average the velocity direction
        ## I think this part is incorrect? Might need to figure out a better way of determining the distribution of speed directions.
        flow_stats["avg_direction"].append( np.ravel(empty_angle_mat) )

        flow_stats["avg_angle"] = np.float32(empty_angle_avg)/np.float32(num_flow_frames)


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

    def visualize_statistics(self, stat_obj):
        
        
        window_size = self.window_size
        avgs = stat_obj#stats1.calculate_flow_statistics_across_visual_field(center_window=window_size)
        vels = avgs["avg_velocity"]
        W = int(vels.shape[1]/2)
        H = int(vels.shape[0]/2)

        plt.imshow(vels)
        plt.plot(W, H, 'red', marker='.', markersize=10)
        plt.plot(W-window_size, H-window_size, 'red', marker='.', markersize=10)
        plt.plot(W-window_size, H+window_size, 'red', marker='.', markersize=10)
        plt.plot(W+window_size, H+window_size, 'red', marker='.', markersize=10)
        plt.plot(W+window_size, H-window_size, 'red', marker='.', markersize=10)
        plt.show()

        fig, axs = plt.subplots(2,2)
        


        vels_window = vels[(H-window_size):(H+window_size), (W-window_size):(W+window_size)]

        mag_H = vels_window.shape[0]
        mag_W = vels_window.shape[1]

        axs[0,0].imshow(vels_window)
        #axs[0,0].imshow(vels)
        axs[0,0].plot(mag_H-1,window_size, "yellow", marker=".", markersize=20) # Top middle -- in graph middle right
        axs[0,0].plot(window_size,0, "green", marker=".", markersize=20) # left middle -- top middle
        axs[0,0].plot(0,window_size, "cyan", marker=".", markersize=20) # bottom middle -- left middle
        axs[0,0].plot(window_size,mag_W-1, "orange", marker=".", markersize=20) # right middle bottom middle
        #axs[0,0].set_ylim((window_size*2)-1, 0)
        axs[0,0].set_xlim(0, (window_size*2)-1)
        axs[0,0].set_title("Avg. Magnitude Across Visual Field")

        avgs["top_middle_speeds"] = np.abs(np.ravel(avgs["top_middle_speeds"]))
        avgs["bottom_middle_speeds"] = np.abs(np.ravel(avgs["bottom_middle_speeds"]))
        avgs["left_middle_speeds"] = np.abs(np.ravel(avgs["left_middle_speeds"]))
        avgs["right_middle_speeds"] = np.abs(np.ravel(avgs["right_middle_speeds"]))

        #avgs["top_middle_speeds"] = avgs["top_middle_speeds"][ avgs["top_middle_speeds"]<200]
        #avgs["bottom_middle_speeds"] = avgs["bottom_middle_speeds"][avgs["bottom_middle_speeds"]<200]
        #avgs["left_middle_speeds"] = avgs["left_middle_speeds"][ avgs["left_middle_speeds"]<200]
        #avgs["right_middle_speeds"] = avgs["right_middle_speeds"][ avgs["right_middle_speeds"]<500]

        avgs["top_middle_speeds"] = avgs["top_middle_speeds"][ avgs["top_middle_speeds"]>0]
        avgs["bottom_middle_speeds"] = avgs["bottom_middle_speeds"][avgs["bottom_middle_speeds"]>0]
        avgs["left_middle_speeds"] = avgs["left_middle_speeds"][ avgs["left_middle_speeds"]>0]
        avgs["right_middle_speeds"] = avgs["right_middle_speeds"][ avgs["right_middle_speeds"]>0]

        sns.distplot(np.ravel(avgs["top_middle_speeds"]), ax=axs[0,1], hist=False, kde=True, 
                    bins=100, color = 'orange', 
                    hist_kws={'edgecolor':'black'},
                    kde_kws={'linewidth': 2})
        
        sns.distplot(np.ravel(avgs["bottom_middle_speeds"]), ax=axs[0,1], hist=False, kde=True, 
                    bins=100, color = 'cyan', 
                    hist_kws={'edgecolor':'black'},
                    kde_kws={'linewidth': 2})
        sns.distplot(np.ravel(avgs["left_middle_speeds"]), ax=axs[0,1], hist=False, kde=True, 
                    bins=100, color = 'green', 
                    hist_kws={'edgecolor':'black'},
                    kde_kws={'linewidth': 2})
        sns.distplot(np.ravel(avgs["right_middle_speeds"]), ax=axs[0,1], hist=False, kde=True, 
                    bins=100, color = 'yellow', 
                    hist_kws={'edgecolor':'black'},
                    kde_kws={'linewidth': 2})
        axs[0,1].set_xlim(0,20)
        axs[0,1].set_title("Magnitude Distribution (Top, bottom, left, right)")
        #axs[1,1].imshow(vels)
        zeros_removed = np.ravel(np.asarray(avgs["avg_direction"]))
        zeros_removed = zeros_removed[zeros_removed>0]
        #axs[2].hist(zeros_removed, bins=60)
        sns.distplot(zeros_removed, ax=axs[1,1], hist=True, kde=True, 
                    bins=50, color = 'darkblue', 
                    hist_kws={'edgecolor':'black'},
                    kde_kws={'linewidth': 4})
        axs[1,1].set_title("Direction Distribution")

        angs = avgs["avg_angle"]*(np.pi/180)
        skip_step = 10
        #x = np.arange(0, window_size*2)
        #y = np.arange(0, window_size*2)
        
        #X, Y = np.meshgrid(x, y)
        center_angles = angs[(H-window_size):(H+window_size), (W-window_size):(W+window_size)]
        center_mags = (vels[(H-window_size):(H+window_size), (W-window_size):(W+window_size)]+1)**3
        
        '''
        w = window_size
        h = window_size
        margin = 0

        nx = int((w - 2 * margin) / skip_step)
        ny = int((h - 2 * margin) / skip_step)

        x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
        y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

        center_mags = center_mags[np.ix_(y, x)]
        center_angles = center_angles[np.ix_(y, x)]

        #kwargs = {**dict(angles="xy", scale_units="xy")}
        axs[1,0].quiver(x, y, center_mags, center_angles)

        axs[1,0].set_ylim(sorted(axs[1,0].get_ylim(), reverse=True))
        axs[1,0].set_aspect("equal")'''
        skippts=20
        scale=0.5
        width=.01
        s = slice(None, None, skippts)

        xmax = np.shape(center_mags)[1]
        xpoints = int(np.shape(center_mags)[1])
        x = np.linspace(0, np.shape(center_mags)[1], xmax)

        ymax = np.shape(center_mags)[0]
        ypoints = int(np.shape(center_mags)[0])
        y = np.linspace(0, np.shape(center_mags)[0], ymax)

        x = x[s]
        y = y[s]
        x2d, y2d = np.meshgrid(x, y, indexing='xy')

        scale2 = 1
        v = scale2 * center_mags * np.cos(center_angles) * -1
        u = scale2 * center_mags * np.sin(center_angles) * -1

        v = u[s, s] / 255 * scale
        u = v[s, s] / 255 * scale
        im1 = axs[1,0].imshow(center_angles*(180/np.pi))
        axs[1,0].quiver(x2d, y2d, u, v, color='red', alpha=0.7, width=width, scale=scale)
        axs[1,0].set_aspect("equal")
        axs[1,0].set_title("Avg. Direction across Visual field center")
        axs[1,0].set_ylim(sorted(axs[1,0].get_ylim(), reverse=True))
        fig.colorbar(im1, ax=axs[1,0])
        plt.tight_layout()

        #axs[1,0].imshow(angs[(H-window_size):(H+window_size), (W-window_size):(W+window_size)])

        ## Add two more figures that show average direction across visual field and one that shows densities of speed at different parts of vis field


        plt.show()
        return 
    
    def compare_flow(self, start_frame=0, end_frame=""):
        '''This needs to be run after the calculate flow statistics function and will give what each of the frames in the given range 
        look like comparing their retina centered flow with the average flow in that same range. -- Make sure it's the same range. '''

        return

subject_folder = "NP42324"
gaze_folder = "000"
flow_path = "Flow"
stats1 = Flow_Statistics(flow_path=flow_path, subject_folder_path=subject_folder, gaze_folder_name=gaze_folder)

stat_obj = stats1.calculate_flow_statistics_across_visual_field(center_window=200, start_frame=3000, end_frame=8000, showFlow=True)
stats1.visualize_statistics(stat_obj)