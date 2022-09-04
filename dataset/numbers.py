from operator import index
import os
import git
import random
from PIL import Image
import numpy as np

from torch import rand

class NumbersDataloader(object):
    """
    Class to load numbers dataset
    """
    def __init__(self, path):
        self.__path = path
        self.__pathDataset = os.path.join(self.__path, "numbers")
        self.__downloadDataset()
        self.__dictNumberPaths = self.__getDictNumbersPath()

    def getDictPaths(self):
        """
        Method to obtain all paths dataset
        """
        return self.__dictNumberPaths

    def __downloadDataset(self):
        """
        Tool to download numbers dataset
        https://github.com/kensanata/numbers
        """
        if os.path.exists(self.__pathDataset):
            pass
        else:
            git.Git(self.__path).clone("https://github.com/kensanata/numbers.git")

    def __getDictNumbersPath(self):
        """
        Tool to obtain dict with the paths for each number
        """
        dictNumberPaths = {
            0 : [],
            1 : [],
            2 : [],
            3 : [],
            4 : [],
            5 : [],
            6 : [],
            7 : [],
            8 : [],
            9 : [],
        }
        folders = os.listdir(self.__pathDataset)
        for folder in folders:
            if "00" in folder:
                folderPath = os.path.join(self.__pathDataset, folder)
                pathNumbers = os.listdir(folderPath)
                for number in pathNumbers:
                    if "scan" not in number and ".jpg" not in number and "README" not in number and ".jpeg" not in number:
                        pathNumber = os.path.join(folderPath, number)
                        imagesNumber = os.listdir(pathNumber)
                        for image in imagesNumber:
                            pathImage = os.path.join(pathNumber, image)
                            dictNumberPaths[int(number)].append(pathImage)

        return dictNumberPaths

    def getImage(self, path):
        """
        Tool to load image

        return numpy array
        """
        imageNumpy = np.asarray(Image.open(path))
        if len(imageNumpy.shape) == 2:
            return imageNumpy.astype(float)
        else:
            return np.mean(imageNumpy, axis=2).astype(float)

    def __getBatchImages(self, equal, size):
        """
        Tool to get a batch of images

        equal : boolean -> if each sample has equal classes
        size : int -> size batch
        """
        batchImages1 = []
        batchImages2 = []
        batchLabels = []
        for index in range(size):
            indexClass  = random.randrange(0, 10)
            if equal:
                batchLabels.append(1.0)
                indexSample1 = random.randrange(0, len(self.__dictNumberPaths[indexClass]))
                while(True):
                    indexSample2 = random.randrange(0, len(self.__dictNumberPaths[indexClass]))
                    if indexSample2 != indexSample1:
                        break

                pathImage1 = self.__dictNumberPaths[indexClass][indexSample1]
                pathImage2 = self.__dictNumberPaths[indexClass][indexSample2]
            else:
                batchLabels.append(0.0)
                indexesOtherClasses = [i for i in range(0, 10)]
                del indexesOtherClasses[indexClass]
                indexOtherClass = indexesOtherClasses[random.randrange(0, len(indexesOtherClasses))]

                indexSample1 = random.randrange(0, len(self.__dictNumberPaths[indexClass]))
                indexSample2 = random.randrange(0, len(self.__dictNumberPaths[indexOtherClass]))

                pathImage1 = self.__dictNumberPaths[indexClass][indexSample1]
                pathImage2 = self.__dictNumberPaths[indexOtherClass][indexSample2]

            batchImages1.append(self.getImage(pathImage1))
            batchImages2.append(self.getImage(pathImage2))

        return batchImages1, batchImages2, batchLabels

    def getRandomBatchSample(self, batchSize):
        """
        Method to obtain random batch sample
        """
        batchSize = batchSize

        batchImages1Equal, batchImages2Equal, batchLabelsEqual = self.__getBatchImages(
            True,
            int(batchSize/2),
        )

        batchImages1Different, batchImages2Different, batchLabelsDifferent = self.__getBatchImages(
            False,
            int(batchSize/2),
        )

        batchImages1 = batchImages1Equal + batchImages1Different
        batchImages2 = batchImages2Equal + batchImages2Different
        batchLabels = batchLabelsEqual + batchLabelsDifferent

        return batchImages1, batchImages2, batchLabels

    def loadSupportVector(self):
        """
        Method to load support vector
        """
        support = {
            0 : self.getImage(self.__dictNumberPaths[0][0]),
            1 : self.getImage(self.__dictNumberPaths[1][0]),
            2 : self.getImage(self.__dictNumberPaths[2][0]),
            3 : self.getImage(self.__dictNumberPaths[3][0]),
            4 : self.getImage(self.__dictNumberPaths[4][0]),
            5 : self.getImage(self.__dictNumberPaths[5][0]),
            6 : self.getImage(self.__dictNumberPaths[6][0]),
            7 : self.getImage(self.__dictNumberPaths[7][0]),
            8 : self.getImage(self.__dictNumberPaths[8][0]),
            9 : self.getImage(self.__dictNumberPaths[9][0]),
        }

        return support

