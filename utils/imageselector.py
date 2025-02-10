import cv2
import numpy as np

class ImageSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.left_clicks = []
        self.right_clicks = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:  # Ctrl + Left Click
                self.right_clicks.append((x, y))
                cv2.circle(self.image_copy, (x, y), 8, (12, 56, 199), -1)  # Red for Ctrl + Left Click
            else:
                self.left_clicks.append((x, y))
                cv2.circle(self.image_copy, (x, y), 8, (6, 161, 78), -1)  # Green for Left Click
        cv2.imshow('Image', self.image_copy)

    def open_image_editor(self):
        image = cv2.imread(self.image_path)
        if image is None:
            print('Error: Unable to load image.')
            return
        self.image_copy = image.copy()
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)
        while True:
            cv2.imshow('Image', self.image_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Clear
                self.left_clicks.clear()
                self.right_clicks.clear()
                self.image_copy = image.copy()
                cv2.imshow('Image', self.image_copy)
            elif key == ord('o'):
                print ('Left Clicks:', np.array(self.left_clicks))
                print ('Right Clicks:', np.array(self.right_clicks))
                break
            elif key == ord('q'):
                raise Exception("Error: Exiting the program...")
        cv2.destroyAllWindows()
        return np.array(self.left_clicks), np.array(self.right_clicks)