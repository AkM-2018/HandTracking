import cv2

class ImageUtil:
    def __init__(self):
        pass

    """Shows the image in a window. Press 'o' to continue, 'c' to exit the program
    """
    def show_image(self, image):
        cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("Image", image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('o'):  # OK
                break
            elif key == ord('q'):  # Return
                raise Exception("Error: Exiting the program...")
        
        cv2.destroyAllWindows()