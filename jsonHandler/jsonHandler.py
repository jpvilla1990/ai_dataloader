import json
import os

class JsonHandler(object):
    """
    Class to handle json files
    """
    def saveJson(pathFile, contentDict):
        """
        Method to save dict content in a json file

        pathFile : String
        contentDict : dict
        """
        if os.path.exists(os.path.dirname(pathFile)) is False:
            os.makedirs(os.path.dirname(pathFile))

        with open(pathFile, 'w') as f:
            json.dump(contentDict, f)

    def loadJson(pathFile):
        """
        Method to load content of json file

        pathFile : String
        """
        with open(pathFile, "r") as f:
            jsonContent = json.load(f)

        return jsonContent