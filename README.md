# PupilLabs_FixationDetector
This repository does fixation detection on eye tracking data collected from Pupil Labs Core eye tracker.

Within subject folder should be a gaze data folder --- something like 000 or 001

It will then search that for an export folder, and take export 000 from it and read in all the data. If you are using a different export, give the function export_number= with your desired export number. 

Find fixations will find all of the fixations and save the data to the subject folder.

Create fixation tracking video will create a video that shows fixations and plots the graph of the gaze velocities and in head eye orientation next to it. 

**Use Pupil Player version 2.6 to export your data**

added an optic flow estimation algorithm. 

This flow algorithm function also does gaze centering, and will output a retina centered video if specified as well as the optic flow for that video (if also specified).



# Object Detection

Uses a method of semantic segmentation of the scene. See MIT CSAIL implementation github. 
            https://github.com/CSAILVision/semantic-segmentation-pytorch?tab=readme-ov-file

Follow their method for installing the modules that are required to run the script. 

Also, make sure that you have pyTorch installed. Follow this link to install if you don't already have it. 
https://pytorch.org/get-started/locally/

This script is computationally intensive, so having torch installed with CUDA support is really useful. 

After that, download this folder https://drive.google.com/drive/folders/18kiZnw7zPW0Lu1BkbfrdbZO2zXv0QXo8?usp=sharing and add
it to the path where the object detection script is located. 

If you want more than just the video, make sure the save labels flag is set to true in the run_object_segmentation_on_video_MIT function. 