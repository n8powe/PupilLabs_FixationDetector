import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
import os
import pathlib
import glob
import scipy
import csv

class Object_Detector():

    def __init__(self, subject_folder_path, imu_path="", gaze_folder_name = "000", tracking_source="Core"):
        self.subject_folder_path = subject_folder_path
        self.gaze_folder_path = subject_folder_path + '/' + gaze_folder_name
        self.IMU_path = imu_path
        self.tracking_source = tracking_source

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

    def run_object_segmentation_on_video_MIT(self, hardware="cpu", save_labels=False):
            '''Runs an alternative method of semantic segmentation of the scene. See MIT CSAIL implementation github. 
            https://github.com/CSAILVision/semantic-segmentation-pytorch?tab=readme-ov-file '''

            def find_gaze_object(self, prediction, names, gaze_y, gaze_x):
                '''This will find the object gaze falls within the bounding box of. If it falls in multiple bounding boxes it will choose the one whose object center is closest.'''
                #print (prediction.shape)
                if gaze_y >= prediction.shape[1] or gaze_x >= prediction.shape[0]:
                    gaze_x = 0 
                    gaze_y = 0

                if gaze_y < 0 or gaze_x < 0:
                    gaze_x = 0
                    gaze_y = 0
                    
                frame_gaze_class_index_at_gaze = prediction[gaze_x, gaze_y]+1

                print (frame_gaze_class_index_at_gaze)

                object_label = names[frame_gaze_class_index_at_gaze]

                if object_label==0:
                    object_label="No Label"

                return object_label

            def add_gaze_to_detection_video(self, frame, frame_number):

                gaze_data = self.gaze_positions_data

                if self.tracking_source=="Core":

                    X = np.median(gaze_data.norm_pos_x[gaze_data.world_index == frame_number])
                    Y = (1-np.median(gaze_data.norm_pos_y[gaze_data.world_index == frame_number]))

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
                    color = (0, 0, 255)  # Green color in BGR
                    thickness = 2
                    frame_with_circle = cv2.circle(frame, center_coordinates, radius, color, thickness)

                    X = int(X*height)
                    Y = int(Y*width)
                    frame = frame

                elif self.tracking_source=="Neon":
                    gaze_index = np.argmin(np.abs(gaze_data['timestamp [ns]'] - self.world_timestamps['timestamp [ns]'][frame_number]))
                    X = gaze_data["gaze x [px]"][gaze_index]
                    Y = gaze_data["gaze y [px]"][gaze_index]

                    if X == np.nan or Y == np.nan:
                        X = 1
                        Y = 1

                    #X = np.mean(gaze_data_frame.norm_pos_x)
                    #Y = np.mean(gaze_data_frame.norm_pos_y)

                    width = frame.shape[0]
                    height = frame.shape[1]

                    #print (frame_number, X*height, Y*width)

                    center_coordinates = (int(X), int(Y))  # Change these coordinates as needed
                    radius = 10
                    color = (0, 0, 255)  # Green color in BGR
                    thickness = 2
                    frame_with_circle = cv2.circle(frame, center_coordinates, radius, color, thickness)
                else:
                    print ("Unknown tracking source. (Adding gaze to detection video function.)")

                return (int(X), int(Y), frame)

            colors = scipy.io.loadmat('Segmentation/color150.mat')['colors']
            names = {}
            with open('Segmentation/object150_info.csv') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    names[int(row[0])] = row[5].split(";")[0]

            def visualize_result(img, pred, index=None):
                # filter prediction class if requested
                if index is not None:
                    pred = pred.copy()
                    pred[pred != index] = -1
                    print(f'{names[index+1]}:')
                    
                # colorize prediction
                pred_color = colorEncode(pred, colors).astype(np.uint8)

                # aggregate images and save
                im_vis = np.concatenate((img, pred_color), axis=1)
                #print (PIL.Image.fromarray(im_vis))

                return im_vis

            net_encoder = ModelBuilder.build_encoder(
                arch='resnet50dilated',
                fc_dim=2048,
                weights='Segmentation/encoder_epoch_20.pth')
            net_decoder = ModelBuilder.build_decoder(
                arch='ppm_deepsup',
                fc_dim=2048,
                num_class=150,
                weights='Segmentation/decoder_epoch_20.pth',
                use_softmax=True)

            crit = torch.nn.NLLLoss(ignore_index=-1)
            segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
            segmentation_module.eval()
            if hardware=="cuda":
                segmentation_module.cuda()
            else:
                segmentation_module.cpu()

            pil_to_tensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
                    std=[0.229, 0.224, 0.225])  # across a large photo dataset.
            ])

            # Open the video file
            cap = self.world_video

            # Check if the video file opened successfully
            if not cap.isOpened():
                print("Error: Unable to open video file.")
                exit()

            # Get the frames per second (fps) of the video
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Get the frame width and height
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create VideoWriter object
            out = cv2.VideoWriter(self.subject_folder_path+'/Object_Segmentation_Video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            frame_number = 0

            labels_out = []

            # Process each frame of the video
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                pil_image = frame
                img_original = np.array(pil_image)
                img_data = pil_to_tensor(pil_image)
                if hardware=="cuda":
                    singleton_batch = {'img_data': img_data[None].cuda()}
                elif hardware=="cpu":
                    singleton_batch = {'img_data': img_data[None]}

                output_size = img_data.shape[1:]

                # Run the segmentation at the highest resolution.
                with torch.no_grad():
                    scores = segmentation_module(singleton_batch, segSize=output_size)
                    
                # Get the predicted scores for each pixel
                _, pred = torch.max(scores, dim=1)
                pred = pred.cpu()[0].numpy()

                gaze_x, gaze_y, img_original = add_gaze_to_detection_video(self, img_original, frame_number)

                gaze_obj = find_gaze_object(self, pred, names, gaze_x, gaze_y)

                #segmented_frame = visualize_result(img_original, pred)

                alpha = 0.7
                #print (img_original.shape)
                #print (pred.shape)
                seg_image_color = cv2.applyColorMap(np.uint8(pred * (255/20)), cv2.COLORMAP_JET)
                segmented_frame = cv2.addWeighted(img_original, alpha, seg_image_color, 1 - alpha, 0)


                cv2.putText(segmented_frame, gaze_obj, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                labels_out.append(gaze_obj)

                out.write(segmented_frame)

                # Display the segmented frame
                cv2.imshow('Semantic Segmentation', segmented_frame)

                frame_number = frame_number + 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_labels:
                np.save(self.subject_folder_path+"/object_labels", np.asarray(labels_out))
            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()

                # Top classes in answer
                #predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]
                #for c in predicted_classes[:15]:
                #    visualize_result(img_original, pred, c)

            return self
    

## Folder structure is assumed to be AllSubjectData/SubjectNeonData/raw-data-export/            (neon only)
## It finds the folder in raw-data-export and then reads the data from within that. 
detector = Object_Detector("Subject7", gaze_folder_name = "NP", tracking_source="Neon")

detector.read_gaze_data()

# This will run on hardware="cpu" -- but it will be REALLY slow. 
detector.run_object_segmentation_on_video_MIT(hardware="cuda", save_labels=True)