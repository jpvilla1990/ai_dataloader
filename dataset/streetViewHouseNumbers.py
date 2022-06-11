import os
import tarfile
from tokenize import Double
import requests
import numpy as np
from PIL import Image
import h5py
from jsonHandler.jsonHandler import JsonHandler

class SVHN(object):
    """
    Class to manage Stree View House Numbers dataset
    http://ufldl.stanford.edu/housenumbers/
    """
    def __init__(self):
        self.__nonCropDatasetUrl = {
            "train" : "http://ufldl.stanford.edu/housenumbers/train.tar.gz",
            "test" : "http://ufldl.stanford.edu/housenumbers/test.tar.gz",
        }

        self.__cropDatasetUrl = {
            "train" : "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "test" : "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
        }

        self.__targetFile = {
            "train" : "train.tar.gz",
            "test" : "test.tar.gz",
        }

        self.__imageFormat = ".png"

        groundTruthFile = "digitStruct.mat"
        groundTruthJsonFile = "groundTruth.json"
        normalizationParametersFile = "normalizationParameters.json"
        trainFolder = os.path.join("train", "train")
        testFolder = os.path.join("test", "test")
        groundTruthTrainFolder = os.path.join("train", "groundTruth")
        groundTruthTestFolder = os.path.join("test", "groundTruth")
        imagesDirectoryTrainFolder = os.path.join("train", "imagesDirectory")
        imagesDirectoryTestFolder = os.path.join("test", "imagesDirectory")

        self.__folderStructure = {
            "train" : {
                "folder" : trainFolder,
                "groundTruthFile" : os.path.join(trainFolder, groundTruthFile),
                "jsonGroundTruth" : os.path.join(groundTruthTrainFolder, groundTruthJsonFile),
                "jsonImagesIndex" : os.path.join(imagesDirectoryTrainFolder, groundTruthJsonFile),
                "jsonNormalizationParameters" : os.path.join(imagesDirectoryTrainFolder, normalizationParametersFile),
            },
            "test" : {
                "folder" : testFolder,
                "groundTruthFile" : os.path.join(testFolder, groundTruthFile),
                "jsonGroundTruth" : os.path.join(groundTruthTestFolder, groundTruthJsonFile),
                "jsonImagesIndex" : os.path.join(imagesDirectoryTestFolder, groundTruthJsonFile),
            },
        }
        self.__matKeys = {
            "digitStruct" : "digitStruct",
            "box" : "bbox",
            "boxKeys" : {
                "height" : "height",
                "left" : "left",
                "width" : "width",
                "top" : "top",
                "label" : "label",
            }
        }

    def __checkH5pyDatasetType(self, value, matFile):
        """
        Checks if the value is type reference or value and returns its value

        value : String or Reference
        matFile : h5py file
        """
        if type(value) == h5py.h5r.Reference:
            return int(np.array(matFile[value]).squeeze())
        elif type(value) == np.float64:
            return int(value)
        else:
            raise Exception("Unexpected type : " + type(value))
    
    def __unpackGroundTruth(self, path):
        """
        Private method to unpack from .mat file into csv

        path : String -> path where the dataset is stored
        """
        for key in list(self.__targetFile.keys()):
            groundTruthFile = os.path.join(
                path,
                self.__folderStructure[key]["groundTruthFile"],
            )
            matFile = h5py.File(groundTruthFile)
            matBox = matFile[self.__matKeys["digitStruct"]][self.__matKeys["box"]]

            index = 0

            jsonContent = dict()
            for box in matBox:
                referenceBox = matFile[box[0]]

                numbers = dict()

                if referenceBox[self.__matKeys["boxKeys"]["height"]].shape[0] == 1:
                    height = self.__checkH5pyDatasetType(
                        referenceBox[self.__matKeys["boxKeys"]["height"]][0][0],
                        matFile,
                    )
                    width = self.__checkH5pyDatasetType(
                        referenceBox[self.__matKeys["boxKeys"]["width"]][0][0],
                        matFile,
                    )
                    top = self.__checkH5pyDatasetType(
                        referenceBox[self.__matKeys["boxKeys"]["top"]][0][0],
                        matFile,
                    )
                    left = self.__checkH5pyDatasetType(
                        referenceBox[self.__matKeys["boxKeys"]["left"]][0][0],
                        matFile,
                    )
                    label = self.__checkH5pyDatasetType(
                        referenceBox[self.__matKeys["boxKeys"]["label"]][0][0],
                        matFile,
                    )
                    numbers.update({
                        0 : {
                            "height" : height,
                            "width" : width,
                            "top" : top,
                            "left" : left,
                            "label" : label,
                        }
                    })
                else:
                    for numberIndex in range(referenceBox[self.__matKeys["boxKeys"]["height"]].shape[0]):
                        height = self.__checkH5pyDatasetType(
                            referenceBox[self.__matKeys["boxKeys"]["height"]][numberIndex][0],
                            matFile,
                        )
                        width = self.__checkH5pyDatasetType(
                            referenceBox[self.__matKeys["boxKeys"]["width"]][numberIndex][0],
                            matFile,
                        )
                        top = self.__checkH5pyDatasetType(
                            referenceBox[self.__matKeys["boxKeys"]["top"]][numberIndex][0],
                            matFile,
                        )
                        left = self.__checkH5pyDatasetType(
                            referenceBox[self.__matKeys["boxKeys"]["left"]][numberIndex][0],
                            matFile,
                        )
                        label = self.__checkH5pyDatasetType(
                            referenceBox[self.__matKeys["boxKeys"]["label"]][numberIndex][0],
                            matFile,
                        )
                        numbers.update({
                            numberIndex : {
                                "height" : height,
                                "width" : width,
                                "top" : top,
                                "left" : left,
                                "label" : label,
                            }
                        })
                jsonContent.update({
                    index : numbers,
                })
                index += 1

                JsonHandler.saveJson(
                    os.path.join(path, self.__folderStructure[key]["jsonGroundTruth"]),
                    jsonContent,
                )

    def __indexImages(self, path):
        """
        Private method to store location of the images

        path : String -> path where the dataset is stored
        """
        for key in list(self.__targetFile.keys()):
            pathImages = os.path.join(
                path,
                self.__folderStructure[key]["folder"],
            )

            index = 0
            imagesDirectory = dict()
            while(True):
                indexImage = index + 1
                imageFileName = str(indexImage) + self.__imageFormat
                imageFile = os.path.join(pathImages, imageFileName)

                if os.path.isfile(imageFile):
                    imagesDirectory.update({
                        index : imageFile,
                    })
                else:
                    break

                index += 1

            if len(list(imagesDirectory.keys())) == 0:
                raise Exception("No " + key + " images in the path : " + pathImages)

            JsonHandler.saveJson(
                    os.path.join(path, self.__folderStructure[key]["jsonImagesIndex"]),
                    imagesDirectory,
                )

    def __normalizationParameters(self, path):
        """
        Privare method to calculate normalization parameters and store them in a json

        path : String
        """
        listImages = JsonHandler.loadJson(
            os.path.join(path, self.__folderStructure["train"]["jsonImagesIndex"])
        )

        sumPixels = 0
        sumVariation = 0
        min = 0
        max = 0
        countPixels = 0
        for key in list(listImages.keys()):
            image = np.asarray(Image.open(listImages[key]))

            numberPixels = np.prod(image.shape)
            countPixels += numberPixels

            sumPixels += image.mean() * numberPixels
            sumVariation += (image.std() ** 2) * numberPixels

            imageMin = image.min()
            imageMax = image.max()

            if imageMin < min:
                min = imageMin
            if imageMax > max:
                max = imageMax

        average = sumPixels / countPixels
        deviation = (sumVariation / countPixels) ** (1/2)

        normalizationParameters = {
            "average" : float(average),
            "deviation" : float(deviation),
            "min" : float(min),
            "max" : float(max),
        }

        JsonHandler.saveJson(
            os.path.join(path, self.__folderStructure["train"]["jsonNormalizationParameters"]),
            normalizationParameters,
        )

    def downloadDataset(self, path, crop=False):
        """
        Method to download dataset in a specific path and extract it

        path : String path where the dataset will be downloaded
        crop : boolean if the cropped dataset is desired
        """
        for key in list(self.__targetFile.keys()):
            targetFile = os.path.join(path, self.__targetFile[key])
            targetPath = os.path.join(path, key)

            if crop:
                response = requests.get( self.__cropDatasetUrl[key])
            else:
                response = requests.get( self.__nonCropDatasetUrl[key])

            open(targetFile, "wb").write(response.content)

            tarFile = tarfile.open(targetFile)
            tarFile.extractall(targetPath)
            tarFile.close()

            os.remove(tarFile)

    def prepareData(self, pathData):
        """
        Method to prepare the dataset for training
        """
        unPack = False
        indexImages = False
        normalizationParameters = False
        for key in list(self.__targetFile.keys()):
            if os.path.exists(os.path.join(pathData, self.__folderStructure[key]["jsonGroundTruth"])) is False:
                unPack = True
            if os.path.exists(os.path.join(pathData, self.__folderStructure[key]["jsonImagesIndex"])) is False:
                indexImages = True
        if os.path.exists(os.path.join(pathData, self.__folderStructure["train"]["jsonNormalizationParameters"])) is False:
            normalizationParameters = True
        if unPack:
            self.__unpackGroundTruth(pathData)
        if indexImages:
            self.__indexImages(pathData)
        if  normalizationParameters:
            self.__normalizationParameters(pathData)

