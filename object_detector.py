import numpy as np
import pandas as pd
import cv2


class Object_Detector():

    def __init__(self, subject_folder_path, imu_path="", gaze_folder_name = "000"):
        self.subject_folder_path = subject_folder_path
        self.gaze_folder_path = subject_folder_path + '/' + gaze_folder_name
        self.IMU_path = imu_path


        