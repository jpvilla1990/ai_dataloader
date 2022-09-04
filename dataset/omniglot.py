import random
import numpy as np
from torchvision.datasets import Omniglot

class OmniglotDataloader(object):
    """
    Class to load dataset omniglot

    path : String -> path where data is stored
    """
    def __init__(self, path):
        self.__path = path
        self.omniglot = Omniglot(
            path,
            download=True,
        )

    def __getLengthDataset(self):
        """
        Tool to obtain length dataset
        """
        return len(self.omniglot)

    def getRandomSample(self):
        """
        Method to get random samples

        return image1, image2, label -> 0 if equal, 1 if different
        """
        index1 = random.randrange(0 , self.__getLengthDataset())
        index2 = random.randrange(0 , self.__getLengthDataset())

        image1, label1 = self.omniglot[index1]
        image2, label2 = self.omniglot[index2]

        if label1 == label2:
            label = 1.0
        else:
            label = 0.0

        return np.asarray(image1), np.asarray(image2), label

    def __getIndexesSameClass(self, index):
        """
        Method to obtain a range of indexes sample class
        """
        downIndex = index
        upIndex = index

        lengthDataset = self.__getLengthDataset()

        currentClass = self.omniglot[index][1]

        counter = index
        while(True):
            counter -= 1
            if counter <= 0:
                break
            nextClass = self.omniglot[counter][1]
            if nextClass == currentClass:
                downIndex = counter
            else:
                break

        counter = index
        while(True):
            counter += 1
            if counter >= lengthDataset:
                break
            nextClass = self.omniglot[counter][1]
            if nextClass == currentClass:
                upIndex = counter
            else:
                break

        return downIndex, upIndex

    def __getBatchImagesByRange(self, indexDown, indexUp, size, exclude):
        """
        Method to obtain batch by receiving a range and excluding or including such range

        indexDown : int
        indexUp : int
        size : int -> size of the batch
        exclude : boolean -> if true the range is excluded, when false the images are obtained from the range
        """
        batchImages1 = []
        batchImages2 = []
        batchLabels = []
        rangeIndexes = [i for i in range(indexDown, indexUp + 1)]
        indexImage1 = indexDown
        for _ in range(size):
            if exclude:
                batchLabels.append(0.0)
                while(True):
                    indexImage2 = random.randrange(0, self.__getLengthDataset())
                    if indexImage2 not in rangeIndexes:
                        break
            else:
                batchLabels.append(1.0)
                while(True):
                    indexImage2 = random.randrange(indexDown, indexUp + 1)
                    if indexImage2 != indexImage1:
                        break

            batchImages1.append(np.asarray(self.omniglot[indexImage1][0]))
            batchImages2.append(np.asarray(self.omniglot[indexImage2][0]))

            if indexImage1 == indexUp:
                indexImage1 = indexDown

            indexImage1 += 1

        return batchImages1, batchImages2, batchLabels

    def getSample(self, index):
        """
        Method to get a sample
        """
        return np.asarray(self.omniglot[index][0]), self.omniglot[index][1]

    def getRandomBatchSample(self, batchSize):
        """
        Method to get a batch, the half of the batch contain images with the same class and the another half images with different class

        batchSize -> size of the batch

        return batchImages1, batchImage2, batchLabels
        """
        index = random.randrange(0 , self.__getLengthDataset())
        rangeIndexes = self.__getIndexesSameClass(index)

        batchImages1Equal, batchImages2Equal, batchLabelsEqual = self.__getBatchImagesByRange(
            rangeIndexes[0],
            rangeIndexes[1],
            int(batchSize/2),
            exclude=False,
        )

        batchImages1Different, batchImages2Different, batchLabelsDifferent = self.__getBatchImagesByRange(
            rangeIndexes[0],
            rangeIndexes[1],
            int(batchSize/2),
            exclude=True,
        )

        batchImages1 = batchImages1Equal + batchImages1Different
        batchImages2 = batchImages2Equal + batchImages2Different
        batchLabels = batchLabelsEqual + batchLabelsDifferent

        return batchImages1, batchImages2, batchLabels

