import cv2

class ImageUtil:
    def __init__(self):
        pass

    """Shows the image in a window. Press 'o' to continue, 'c' to exit the program
    """
    def show_image(self, image):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        height, width = self.resize_with_aspect_ratio(image)
        cv2.resizeWindow('Image', height, width)
        cv2.imshow("Image", image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('o'):  # OK
                break
            elif key == ord('q'):  # Return
                raise Exception("Error: Exiting the program...")
        
        cv2.destroyAllWindows()
    
    """Returns resized dimensions
    """
    def resize_with_aspect_ratio(self, image, height=None, width=None, inter=cv2.INTER_AREA):
        dim = None
        (w, h) = image.shape[:2]
        
        if h <= 800 and w <= 800:
            return h, w

        if width is None and height is None:
            if h >= w:
                height = 800
                r = height / float(h)
                width = int(w * r)
            else:
                width = 800
                r = width / float(w)
                height = int(h * r)
        elif width is None:
            r = height / float(h)
            width = int(w * r)
        else:
            r = width / float(w)
            height = int(h * r)

        print("height:", height, "width:", width)
        return height, width