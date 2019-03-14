from utils import readImage, integralImage, Feature
import numpy as np
from os import listdir
from multiprocessing import Pool
from functools import partial


class ViolaJonesTrain:
    def __init__(self, T, threads = 4, positivePath = "positive", negativePath = "negative", 
                    maxPositiveImages = -1, maxNegativeImages = -1):
        """
        T: number of weak classifiers
        threads: number of threads to run parallel (on some computations)
        poitivePath: folder with the positive images
        negativePath: folder with the negative images
        maxPositiveImages: mamimum number of positive images (-1: all posible)
        maxNegativeImages: mamimum number of negative images (-1: all posible)

        tainingData: list with the integral image of the training data and the type of the image
                     1 for positve
                     0 for negative
        positiveNumber: number of positive images
        negativeNumber: number of negative images
        """
        self.T = T
        self.threads = threads
        
        posImages = listdir(positivePath)
        negImages = listdir(negativePath)

        if maxPositiveImages > -1:
            posImages = posImages[:maxPositiveImages]

        if maxNegativeImages > -1:
            negImages = negImages[:maxNegativeImages]

        self.positveNumber = len(posImages)
        self.negativeNumber = len(negImages)

        self.trainingData = []
        self.imageClass = []

        for im in posImages:
            self.trainingData.append(integralImage(readImage(positivePath + "/" + im)))
            self.imageClass.append(1)

        for im in negImages:
            self.trainingData.append(integralImage(readImage(negativePath + "/" + im)))
            self.imageClass.append(0)

    def createFeatures(self):
        """
        Creates all posible features for the training images
        
        Returnes:
            a list of features
        """
        totalHeight, totalWidth = self.trainingData[0].shape
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

        pool = Pool(self.threads)
        featureValues = pool.map(partial(self.calculateFeatures, features=features), self.trainingData)

        return featureValues

    @staticmethod
    def calculateFeatures(ii, features):
        imageValues = []
        for feature in features:
            imageValues.append(feature.calculate(ii))
        return imageValues


    def trainModel(self):
        # init weights
        weights = np.concatenate((np.full([self.positveNumber], 1.0 / (2 * self.positveNumber)),
                                  np.full([self.negativeNumber], 1.0 / (2 * self.negativeNumber))
                                  ))

        # Calculate all posible features
        print("Creating features.")
        features = self.createFeatures()

        # Calculate for every image the value of every feature
        print("Calculating features for all images.")
        featureValues = self.applyFeatures(features)

        # return features
        return featureValues
