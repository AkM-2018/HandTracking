#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv
import cv2
import shutil
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sam2.build_sam import build_sam2_video_predictor

from videoprocessor import VideoProcessor
from utils.imageselector import ImageSelector


if __name__ == '__main__':
    param_dict = {}
    param_dict['use_cpu'] = (True if '--use-cpu' in sys.argv else False)
    param_dict['manual_mode'] = (True if '--manual' in sys.argv else False)
    param_dict['check_landmark'] = (True if '--check-landmark' in sys.argv else False)
    param_dict['check_segment'] = (True if '--check-segment' in sys.argv else False)
    
    source_video_path = sys.argv[1]
    target_video_path = sys.argv[2]
    video_processor = VideoProcessor(source_video_path, target_video_path, param_dict)
    video_processor.process()

