from utils import readImage, integralImage, Feature, StrongClassifier
import numpy as np
from os import listdir
from multiprocessing import Pool
from functools import partial


class ViolaJonesTrain:
    def __init__(self, threads = 4, positivePath = "positive", negativePath = "negative", 
                    maxPositiveImages = -1, maxNegativeImages = -1):
        """
        threads: number of threads to run parallel (on some computations)
        poitivePath: folder with the positive images
        negativePath: folder with the negative images
        maxPositiveImages: maximum number of positive images (-1: all posible)
        maxNegativeImages: maximum number of negative images (-1: all posible)

        tainingData: list with the integral image of the training data and the type of the image
                     1 for positve
                     0 for negative
        positiveNumber: number of positive images
        negativeNumber: number of negative images
        """
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

        Arguments:
            features: list of Feature objects

        Returns:
            2D numpy array with the values of the features
        """

        pool = Pool(self.threads)
        featureValues = pool.map(partial(self.calculateFeatures, features=features), self.trainingData)

        return np.array(featureValues).T

    @staticmethod
    def calculateFeatures(ii, features):
        imageValues = []
        for feature in features:
            imageValues.append(feature.calculate(ii))
        return imageValues

    def trainWeakClassifiers(self, appliedFeatures, weights):
        """
        This method trains a weak classifier.

        Arguments:
            appliedFeatures: 2D list of the values for every feature in every imagea
            weights: list with the weights of every image

        Returns:
            weakClassifiers: list of tuples containing the error, threshold, polarity 
                and feature index of every weak classifier
        """
        posWeightSum = negWeightSum = 0
        for imClass, weight in zip(self.imageClass, weights):
            if imClass == 1:
                posWeightSum += weight
            else:
                negWeightSum += weight

        pool = Pool(self.threads)
        weakClassifiers = pool.map(
                                   partial(self.trainUnitWeakClassifier, imageClass = self.imageClass, weights = weights, 
                                           posWeightSum = posWeightSum, negWeightSum = negWeightSum), 
                                   enumerate(appliedFeatures)
                                   )

        return weakClassifiers


    @staticmethod
    def trainUnitWeakClassifier(enumeratedFeatures, imageClass, weights, posWeightSum, negWeightSum):
        """
        This method calculates the best classifier for a feature
        """
        index, feature = enumeratedFeatures

        # sorted based on feature value
        sortedValues = sorted(zip(feature, weights, imageClass), key = lambda x: x[0])

        currentPosSum = currentNegSum = currentPosWeights = currentNegWeights = 0

        minError = float('inf')
        theta = p = 0

        for featureValue, weight, imClass in sortedValues:
            if imClass == 1:
                currentPosSum += 1
                currentPosWeights += weight
            else:
                currentNegSum += 1
                currentNegWeights += weight

            error = min(posWeightSum + currentNegWeights - currentPosWeights, 
                        negWeightSum + currentPosWeights - currentNegWeights 
                       )
            
            if error < minError:
                minError = error
                theta = featureValue # threshold
                p = 1 if currentPosSum > currentNegSum else -1 # polarity

        return (minError, theta, p, index)
        
    def bestWeakClassifier(self, weakClassifiers, featureValues, weights):
        """
        This method finds the best weak classifier.

        Arguments:
            weakClassifiers: list with the values of all weak classifiers
            featureValues: 2D list of the values for every feature in every imagea
            weights: list with the weights of every image

        Returns:
            tuple with the ((threshold, polarity, index),(error, epsilon)) of the best classifier
        """
        pool = Pool(self.threads)
        weakErrors = pool.map(
                              partial(self.calculateWeakError, featureValues = featureValues,
                                      weights = weights, imageClass = self.imageClass
                                     ),
                              weakClassifiers
                             )

        # Find the index of the best weak classifier
        minWeak = min(enumerate(weakErrors), key = lambda x: x[1])

        return (weakClassifiers[minWeak[0]][1:], minWeak[1])

    @staticmethod
    def calculateWeakError(weakClassifier, featureValues, weights, imageClass):
        """
        This method calculates the error for a weak classifier
        """
        _, theta, p, indexClassifier = weakClassifier
        error = 0.0
        epsilon = []

        for index, f in enumerate(featureValues[indexClassifier]):
            h = 1 if p * f < p * theta else 0
            e = abs(h - imageClass[index])
            epsilon.append(e)
            error += weights[index] * e

        return (error, epsilon)


    def pureStrongClassifierTrain(self, T):
        """
        Creates a stong classifier of T weak classifiers using AdaBoost algorithm

        Arguments:
            T: number of weak classifiers

        Returns:
            StrongClassifier object
        """
        # Calculate all posible features
        print("Creating features.")
        allFeatures = self.createFeatures()

        # Calculate for every image the value of every feature
        print("Calculating features for all images.")
        featureValues = self.applyFeatures(allFeatures)

        #################################################################
        # AdaBoost algorithm
      
        alphas = []
        bestWeakClassifiers = []

        # init weights
        weights = np.concatenate((np.full([self.positveNumber], 1.0 / (2 * self.positveNumber)),
                                  np.full([self.negativeNumber], 1.0 / (2 * self.negativeNumber))
                                  ))

        for itter in range(T):
            print('{} out of {} weak clasifiers.'.format(itter + 1, T))

            # Normalize weights
            weights = weights / np.sum(weights)

            # Train all weak classifiers
            print("\tTraining weak classifiers.")
            weakClassifiers = self.trainWeakClassifiers(featureValues, weights)

            # ((threshold, polarity, feature index),(error, epsilon)) of the best classifier
            print("\tChoosing the best.")
            bestWeak, (error, epsilon) = self.bestWeakClassifier(weakClassifiers, featureValues, weights)

            epsilon = np.array(epsilon)

            # Make sure error is not 0
            if error == 0.0:
                error = np.finfo(np.float32).eps

            beta = error / (1.0 - error)

            # Calculate new weights
            weights = weights * (beta ** (1 - epsilon))

            # Calculate alpha
            alphas.append(np.log10(1 / beta))

            bestWeakClassifiers.append(bestWeak)

        # End of AdaBoost algorithm
        #################################################################
        print('End of AdaBoost.')

        # Construct Stong classifier object
        features = []
        thresholds = []
        polarities = []

        for weak in bestWeakClassifiers:
            threshold, polarity, index = weak
            features.append(allFeatures[index])
            polarities.append(polarity)
            thresholds.append(threshold)

        strongClassifier = StrongClassifier(alphas, features, thresholds, polarities)

        return (strongClassifier)


    def trainModel(self):
        # # Calculate all posible features
        # print("Creating features.")
        # allFeatures = self.createFeatures()

        # # Calculate for every image the value of every feature
        # print("Calculating features for all images.")
        # featureValues = self.applyFeatures(allFeatures)

        # #################################################################
        # # AdaBoost algorithm
        
        # alphas = []
        # bestWeakClassifiers = []

        # # init weights
        # weights = np.concatenate((np.full([self.positveNumber], 1.0 / (2 * self.positveNumber)),
        #                           np.full([self.negativeNumber], 1.0 / (2 * self.negativeNumber))
        #                           ))

        # for itter in range(self.T):
        #     print('{} out of {} weak clasifiers.'.format(itter + 1, self.T))

        #     # Normalize weights
        #     weights = weights / np.sum(weights)

        #     # Train all weak classifiers
        #     print("\tTraining weak classifiers.")
        #     weakClassifiers = self.trainWeakClassifiers(featureValues, weights)

        #     # ((threshold, polarity, feature index),(error, epsilon)) of the best classifier
        #     print("\tChoosing the best.")
        #     bestWeak, (error, epsilon) = self.bestWeakClassifier(weakClassifiers, featureValues, weights)

        #     epsilon = np.array(epsilon)

        #     # Make sure error is not 0
        #     if error == 0.0:
        #         error = np.finfo(np.float32).eps

        #     beta = error / (1.0 - error)

        #     # Calculate new weights
        #     weights = weights * (beta ** (1 - epsilon))

        #     # Calculate alpha
        #     alphas.append(np.log10(1 / beta))

        #     bestWeakClassifiers.append(bestWeak)

        # # End of AdaBoost algorithm
        # #################################################################
        # print('End of AdaBoost.')

        # # Construct Stong classifier object
        # features = []
        # thresholds = []
        # polarities = []

        # for weak in bestWeakClassifiers:
        #     threshold, polarity, index = weak
        #     features.append(allFeatures[index])
        #     polarities.append(polarity)
        #     thresholds.append(threshold)

        # strongClassifier = StrongClassifier(alphas, features, thresholds, polarities)

        # # tp = tn = fp = fn = 0
        # # for imClass, ii in zip(self.imageClass, self.trainingData):
        # #     retClass = strongClassifier.classify(ii)
        # #     if retClass == imClass:
        # #         if retClass == 1:
        # #             tp += 1
        # #         else:
        # #             tn += 1
        # #     elif retClass == 1:
        # #         fp += 1
        # #     else:
        # #         fn += 1
                

        # # return features
        # return (strongClassifier)
