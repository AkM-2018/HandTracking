import numpy as np
import cv2
import mediapipe as mp
from numpy.typing import NDArray

from .imageutil import ImageUtil

class HandTracker:
    """Initializes variables required for hand landmark detection.

    Args:
        image_path: path to the image frame for hand detection
        min_detection_confidence: minimum confidence for detection
        min_tracking_confidence: minimum confidence for tracking
    """
    def __init__(self, image_path: str, min_detection_confidence: int = 0.5, min_tracking_confidence: int = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.image_path = image_path
        self.image_util = ImageUtil()

    """Processes the image frame and stores hand landmark information in hand_results variable

    """
    def find_hand_landmarks(self):
        image = cv2.imread(self.image_path)
        image_height, image_width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        self.hand_results = {'left': [], 'right': []}

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = 'left' if handedness.classification[0].label == 'Left' else 'right'
                hand_landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    x_pixel = int(landmark.x * image_width)
                    y_pixel = int(landmark.y * image_height)
                    hand_landmarks_list.append(np.array([x_pixel, y_pixel]))
                self.hand_results[hand_type].append(hand_landmarks_list)

    """Shows the hand coordinates on the image
    """
    def show_image(self, hand_coors: list):
        image = cv2.imread(self.image_path)
        if image is None:
            print("Error: Unable to load image.")
            return
        image_copy = image.copy()
        
        for coord in hand_coors:
            cv2.circle(image_copy, tuple(coord), 8, (6, 161, 78), -1)  # Green for hand coordinates
        
        self.image_util.show_image(image_copy)

    """Returns the pixel values of the wrist in the image frame

    Returns:
        Pixel values of wrist in the image frame
    """
    def get_wrist_landmark(self, check_landmark: bool) -> NDArray[np.int32]:
        hand_coords = []

        if len(self.hand_results['left']) != 0:
            hand_coords.append(np.array(self.hand_results['left'][0][0]))

        if len(self.hand_results['right']) != 0:
            hand_coords.append(np.array(self.hand_results['right'][0][0]))

        if check_landmark == True:
            self.show_image(hand_coords)

        return np.array(hand_coords)