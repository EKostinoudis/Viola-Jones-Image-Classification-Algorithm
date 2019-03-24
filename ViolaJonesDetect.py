from utils import integralImage
import cv2
import math

class ViolaJonesDetect:
    def __init__(self, cascade):
        """
        Arguments:
            cascade: Cascade object
        """
        self.cascade = cascade

    def detect(self, image, scale):
        """
        Detect if there are any positive areas based on the cascade

        Arguments:
            image: numpy 2D array of a grayscale image that we want to detect
            scale: ratio for resizing the image

        Returns:
            list of tuples with the positive rectangles
                tuple(x, y, width, height)
        """
        if scale >= 1.0 and scale < 0:
            raise  ValueError("scale must be in [0, 1)")
        positiveRect = []

        height, width = image.shape
        casHeight, casWidth = self.cascade.shape
        
        while image.shape[0] >= self.cascade.shape[0] and \
              image.shape[1] >= self.cascade.shape[1]:
            # calculate integral image
            ii = integralImage(image)

            # check every posible rectangle
            for i in range(image.shape[0] - self.cascade.shape[0] + 1):
                for j in range(image.shape[1] - self.cascade.shape[1] + 1):
                    if self.cascade.classifyOff(ii, j, i) == 1:
                        # find the values based on the original image
                        x = math.floor((j / image.shape[1]) * width)
                        y = math.floor((i / image.shape[0]) * height)
                        w = math.floor((casWidth / image.shape[1]) * width)
                        h = math.floor((casHeight / image.shape[0]) * height)
                        positiveRect.append((x, y, w, h))

            # resize image
            newHeight = math.floor(image.shape[0] * scale)
            newWidth  = math.floor(image.shape[1] * scale)
            
            image = cv2.resize(image, dsize=(newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
            
        return positiveRect

    

