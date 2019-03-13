from PIL import Image
import numpy as np

def readImage(address):
    return np.array(Image.open(address).convert('L'))

def integralImage(image):
    return image.cumsum(axis=1).cumsum(axis=0)

class Rectangle:
        def __init__(self, x, y, width, height):
            """
            x, y: start potition of the rectangle
            width, height: width, height of the rectangle
            """
            self.x = x
            self.y = y
            self.width = width
            self.height = height    

        def calculateRect(self, ii):
            """
            calulates the value of the rectangle region 
            given the integral image (ii)

            arguments:
                ii: integral image (numpy array)

            returns:
                value of the sum of all points in the rectagle region
            """
            return  ii[self.y]              [self.x] +\
                    ii[self.y + self.height][self.x + self.width] -\
                    ii[self.y]              [self.x + self.width] -\
                    ii[self.y + self.height][self.x]

class Feature:
    def __init__(self, ftype, x, y, width, height):
        """ 
        ftype: is the type of feature
            1: 2 blocks, negative left | positive right
            2: 2 blocks, positive up   | negative down
            3: 3 blocks, negative left and right | positive middle
            4: 4 blocks, negative top left and bottom right | 
                        positive top right and bottom left
        x, y: start position of the feature (top left)
        width, height: total width and height of the feature

        positive: array with the positive rectangles
        negative: array with the negative rectangles
        """
        self.ftype = ftype
        self.positive = []
        self.negative = []

        if ftype == 1:
            if width % 2 == 0:
                halfWidth = width//2

                self.positive.append(Rectangle(x + halfWidth, y, halfWidth, height))
                self.negative.append(Rectangle(x, y, halfWidth, height))
            else:
                raise ValueError('Width must be multiple of 2')
        elif ftype == 2:
            if height % 2 == 0:
                halfHeight = height//2

                self.positive.append(Rectangle(x, y, width, halfHeight))
                self.negative.append(Rectangle(x, y + halfHeight, width, halfHeight))
            else:
                raise ValueError('Height must be multiple of 2')
        elif ftype == 3:
            if width % 3 == 0:
                tierceWidth = width//3

                self.positive.append(Rectangle(x + tierceWidth, y, tierceWidth, height))
                
                self.negative.append(Rectangle(x, y, tierceWidth, height))
                self.negative.append(Rectangle(x + 2*tierceWidth, y, tierceWidth, height))
            else:
                raise ValueError('Width must be multiple of 3')
        elif ftype == 4:
            if width % 2 == 0 and height % 2 == 0:
                halfWidth = width//2
                halfHeight = height//2

                self.positive.append(Rectangle(x + halfWidth, y, halfWidth, halfHeight))
                self.positive.append(Rectangle(x, y + halfHeight, halfWidth, halfHeight))
                

                self.negative.append(Rectangle(x, y, halfWidth, halfHeight))
                self.negative.append(Rectangle(x + halfWidth, y + halfHeight, halfWidth, halfHeight))
            else:
                raise ValueError('Width and height must be multiples of 23')
        else:
            raise ValueError('ftype must be an integer in the range [1,4]')

    def calculate(self, ii):
        pos = [x.calculateRect(ii) for x in self.positive]
        neg = [x.calculateRect(ii) for x in self.negative]
        return [pos, neg]
         

    