from PIL import Image
import numpy as np
import pickle

def readImage(address):
    return np.array(Image.open(address).convert('L'))

def integralImage(image):
    return image.cumsum(axis=1).cumsum(axis=0)

def saveObject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadObject(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

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

            if self.x == 0:
                if self.y != 0:
                    value = ii[self.y + self.height  - 1][self.x + self.width - 1] -\
                            ii[self.y - 1]               [self.x + self.width - 1]
                else:
                    value = ii[self.y + self.height - 1][self.x + self.width - 1]
            elif self.y == 0:
                value = ii[self.y + self.height - 1][self.x + self.width - 1] -\
                        ii[self.y + self.height - 1][self.x - 1]
            else:         
                value = ii[self.y - 1]              [self.x - 1] +\
                        ii[self.y + self.height - 1][self.x + self.width - 1] -\
                        ii[self.y - 1]              [self.x + self.width - 1] -\
                        ii[self.y + self.height - 1][self.x - 1]    

            return value

        def calculateRectOff(self, ii, xoff, yoff):
            """
            calulates the value of the rectangle region 
            given the integral image (ii) and a offset

            arguments:
                ii: integral image (numpy array)
                xoff: offset on x axis
                yoff: offset on y axis

            returns:
                value of the sum of all points in the rectagle region
            """
            x , y = self.x + xoff, self.y + yoff

            if x == 0:
                if y != 0:
                    value = ii[y + self.height  - 1][x + self.width - 1] -\
                            ii[y - 1]               [x + self.width - 1]
                else:
                    value = ii[y + self.height - 1][x + self.width - 1]
            elif y == 0:
                value = ii[y + self.height - 1][x + self.width - 1] -\
                        ii[y + self.height - 1][x - 1]
            else:         
                value = ii[y - 1]              [x - 1] +\
                        ii[y + self.height - 1][x + self.width - 1] -\
                        ii[y - 1]              [x + self.width - 1] -\
                        ii[y + self.height - 1][x - 1]    

            return value

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
                raise ValueError('Width and height must both be multiples of 2')
        else:
            raise ValueError('ftype must be an integer in the range [1,4]')

    def calculate(self, ii):
        pos = sum([x.calculateRect(ii) for x in self.positive])
        neg = sum([x.calculateRect(ii) for x in self.negative])
        return pos - neg

    def calculateOff(self, ii, xoff, yoff):
        pos = sum([x.calculateRectOff(ii, xoff, yoff) for x in self.positive])
        neg = sum([x.calculateRectOff(ii, xoff, yoff) for x in self.negative])
        return pos - neg
         

class WeakClassifier:
    def __init__(self, feature, threshold, polarity):
        """
        feature: Feature (class) object
        threshold: threshold of the classifier
        polarity: polarity of the classifier
        """
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity

    def classifie(self, ii):
        """
        Classifies an image.

        Arguments:
            ii: integral image

        Returns:
            1: for positive classification
            0: for negative classification
        """
        f = self.feature.calculate(ii)
        return 1 if self.polarity * f < self.polarity * self.threshold else 0

    def classifieOff(self, ii, xoff, yoff):
        """
        Classifies an image given an offset.

        Arguments:
            ii: integral image
            xoff: offset on x axis
            yoff: offset on y axis

        Returns:
            1: for positive classification
            0: for negative classification
        """
        f = self.feature.calculateOff(ii, xoff, yoff)
        return 1 if self.polarity * f < self.polarity * self.threshold else 0

class StrongClassifier:
    def __init__(self, alphas, features, thresholds, polarities, s = 0):
        """
        alphas: list of weights of the weak classifiers
        features: list of Feature (class) objects
        thresholds: list of thresholds of the weak classifiers
        polarities: list of polarities of the weak classifiers
        s: shift of the clasifier

        weakClassifiers: list of WeakClassifier objects
        """
        self.alphas = []
        self.weakClassifiers = []
        self.s = s

        if type(alphas) is list and type(features) is list \
            and type(polarities) is list and type(thresholds) is list:
            
            self.alphas.extend(alphas)
            
            for polarity, threshold, feature in zip(polarities, thresholds, features):
                self.weakClassifiers.append(WeakClassifier(feature, threshold, polarity))

        else:
            raise TypeError('All arguments must be type: list')

    def addWeakClassifier(self, alpha, feature, threshold, polarity):
        """
        Adds a value weak classifier to the strong

        Arguments:
            alpha: weight of the weak classifier
            feature: Feature (class) object
            threshold: threshold of the classifier
            polarity: polarity of the classifier
        """
        self.alphas.append(alpha)
        self.weakClassifiers.append(WeakClassifier(feature, threshold, polarity))

    def changeShift(self, s):
        self.s = s

    def classify(self, ii):
        """
        Classifies an image. (strong classifier)

        Arguments:
            ii: integral image

        Returns:
            1: for positive classification
            0: for negative classification
        """
        sumAlphas = sum(self.alphas)
        sumH = 0
        for index, weak in enumerate(self.weakClassifiers):
            sumH += self.alphas[index] * (weak.classifie(ii) + self.s)

        return 1 if sumH >= 0.5 * sumAlphas else 0
    
    def classifyOff(self, ii, xoff, yoff):
        """
        Classifies an image given an offset. (strong classifier)

        Arguments:
            ii: integral image
            xoff: offset on x axis
            yoff: offset on y axis

        Returns:
            1: for positive classification
            0: for negative classification
        """
        sumAlphas = sum(self.alphas)
        sumH = 0
        for index, weak in enumerate(self.weakClassifiers):
            sumH += self.alphas[index] * (weak.classifieOff(ii, xoff, yoff) + self.s)

        return 1 if sumH >= 0.5 * sumAlphas else 0
                
            
class Cascade:
    def __init__(self, strongClassifiers, shape):
        """
        strongClassifiers: list of strong classifiers
        shape: tuple with the shape of the training images (height, width)
        """
        self.strongClassifiers = strongClassifiers
        self.shape = shape


    def addStrongClassifier(self, strongClassifier):
        """
        Adds a strong classifier to the cascade

        Argument:
            strongClassifier: a StrongClassifier object
        """
        self.strongClassifiers.append(strongClassifier)

    def classify(self, ii):
        """
        Classifies an image. (cascaded classifier)

        Arguments:
            ii: integral image

        Returns:
            1: for positive classification
            0: for negative classification
        """
        for strongClassifier in self.strongClassifiers:
            if strongClassifier.classify(ii) == 0:
                return 0

        return 1

    def classifyOff(self, ii, xoff, yoff):
        """
        Classifies an image given an offset. (cascaded classifier)

        Arguments:
            ii: integral image
            xoff: offset on x axis
            yoff: offset on y axis

        Returns:
            1: for positive classification
            0: for negative classification
        """
        for strongClassifier in self.strongClassifiers:
            if strongClassifier.classifyOff(ii, xoff, yoff) == 0:
                return 0

        return 1

    def saveCascade(self, filename):
        """
        Saves the class object into a file

        Arguments:
            filename: name of the file
        """
        saveObject(self, filename)

    @staticmethod
    def loadCascade(filename):
        return loadObject(filename)
