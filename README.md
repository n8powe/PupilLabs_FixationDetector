# PupilLabs_FixationDetector
This repository does fixation detection on eye tracking data collected from Pupil Labs Core eye tracker.

Within subject folder should be a gaze data folder --- something like 000 or 001

It will then search that for an export folder, and take export 000 from it and read in all the data. If you are using a different export, give the function export_number= with your desired export number. 

Find fixations will find all of the fixations and save the data to the subject folder.

Create fixation tracking video will create a video that shows fixations and plots the graph of the gaze velocities and in head eye orientation next to it. 

**Use Pupil Player version 2.6 to export your data**

added an optic flow estimation algorithm. 

This flow algorithm function also does gaze centering, and will output a retina centered video if specified as well as the optic flow for that video (if also specified).