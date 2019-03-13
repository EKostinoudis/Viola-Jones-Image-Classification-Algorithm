from utils import readImage, integralImage, Feature
import numpy as np
from os import listdir

class ViolaJonesTrain:
    def __init__(self, T, positivePath = "positive", negativePath = "negative"):
        """
        T: number of weak classifiers
        poitivePath: folder with the positive images
        negativePath: folder with the negative images

        tainingData: list with the integral image of the training data and the type of the image
                     1 for positve
                     0 for negative
        positiveNumber: number of positive images
        negativeNumber: number of negative images
        """
        self.T = T
        
        posImages = listdir(positivePath)
        negImages = listdir(negativePath)

        self.positveNumber = len(posImages)
        self.negativeNumber = len(negImages)

        self.trainingData = []

        for im in posImages:
            self.trainingData.append([integralImage(readImage(positivePath + "/" + im)), 1])

        for im in negImages:
            self.trainingData.append([integralImage(readImage(negativePath + "/" + im)), 0])

    def createFeatures(self):
        """
        Creates all posible features for the training images
        
        Returnes:
            a list of features
        """
        totalHeight, totalWidth = self.trainingData[0][0].shape
        features = []

        # Type 1 features
        for x in range(0, totalWidth):
            for y in range(0, totalHeight):
                for width in range(2, totalWidth - x + 1, 2):
                    for height in range(1, totalHeight - y + 1):
                        features.append(Feature(1, x, y, width, height))

        # Type 2 features
        for x in range(0, totalWidth):
            for y in range(0, totalHeight):
                for width in range(1, totalWidth - x + 1):
                    for height in range(2, totalHeight - y + 1, 2):
                        features.append(Feature(2, x, y, width, height))

        # Type 3 features
        for x in range(0, totalWidth):
            for y in range(0, totalHeight):
                for width in range(3, totalWidth - x + 1, 3):
                    for height in range(1, totalHeight - y + 1):
                        features.append(Feature(3, x, y, width, height))

        # Type 4 features
        for x in range(0, totalWidth):
            for y in range(0, totalHeight):
                for width in range(2, totalWidth - x + 1, 2):
                    for height in range(2, totalHeight - y + 1, 2):
                        features.append(Feature(4, x, y, width, height))

        return features


    def applyFeatures(self, features):
        """
        Apply all features to each training image

        Returns:
            2D list with the values of the features
        """

        featureValues = []
        counter = 0
        for train in self.trainingData:
            imageValues = []
            for feature in features:
                imageValues.append(feature.calculate(train[0]))

            counter +=1
            print("Calculated {} out of {}".format(counter, len(self.trainingData)))
            featureValues.append(imageValues)

        return featureValues



    def trainModel(self):
        # init weights
        weights = np.concatenate((np.full([self.positveNumber], 1.0 / (2 * self.positveNumber)),
                                  np.full([self.negativeNumber], 1.0 / (2 * self.negativeNumber))
                                  ))

        # Calculate all posible features
        features = self.createFeatures()

        # Calculate for every image the value of every feature
        featureValues = self.applyFeatures(features)

        return featureValues
