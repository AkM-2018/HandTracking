import os

from utils.handtracker import HandTracker
from utils.videopredictor import VideoPredictor
from utils.imageselector import ImageSelector

class VideoProcessor:
    """Initialize variables

    Args:
        source_video_path: path of the source video file
        target_video_path: path of the target video file
        use_cpu: boolean on whether to use cpu or available resources(like cuda) 
    """
    def __init__(self, source_video_path: str, target_video_path: str, use_cpu: bool, manual_mode: bool):
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.use_cpu = use_cpu
        self.manual_mode = manual_mode

    """Validates source_video_path and target_video_path

    Throws:
        source file is not present at source_video_path
        target file is already present at target_video_path
    """
    def validate_input_video_file(self):
        if not os.path.isfile(self.source_video_path):
            raise Exception("Source video does not exist! Please provide a valid path")

        if os.path.isfile(self.target_video_path):
            raise Exception(f"Target file: {self.target_video_path} already present!")
    
    """Processes the input video file and saves the processed file

    """
    def process(self):
        self.validate_input_video_file()
        self.videoPredictor = VideoPredictor()
        self.videoPredictor.init_device(self.use_cpu)
        self.videoPredictor.modify_torch_settings_for_device()
        self.videoPredictor.save_jpg_dir_from_vid(self.source_video_path)

        first_image_path = './vids/00000.jpg'
        global hand_landmark, landmark_negative
        
        if self.manual_mode == True:
            image_selector = ImageSelector(first_image_path)
            hand_landmark, landmark_negative = image_selector.open_image_editor()
        else:
            self.handTracker = HandTracker(first_image_path)
            self.handTracker.find_hand_landmarks()
            hand_landmark = self.handTracker.get_wrist_landmark()
            landmark_negative = []

        print('hand_landmark', hand_landmark)
        print('landmark_negative', landmark_negative)
        
        self.videoPredictor.init_video_predictor()
        self.videoPredictor.add_label_points(hand_landmark, landmark_negative)
        self.videoPredictor.propogate_in_video(self.target_video_path)
        self.videoPredictor.cleanup()