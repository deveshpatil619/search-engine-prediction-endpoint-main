import os
from from_root import from_root ## importing the library from_root to get the root directory of the project
from dotenv import load_dotenv ## importing the load_dotenv that loads all variables in .env file


class AwsStorage: ## class name is AwsStorage
    def __init__(self): #__init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        load_dotenv() # method that loads  environment variables from a .env file.
        self.ACCESS_KEY_ID = os.getenv["AWS_ACCESS_KEY_ID"]  # os.environ["ACCESS_KEY_ID"] assigns the value of the 
        #environment variable ACCESS_KEY_ID to the ACCESS_KEY_ID attribute of the instance.
        self.SECRET_KEY = os.getenv["AWS_SECRET_ACCESS_KEY"] ## same as above for variable AWS_SECRET_ACCESS_KEY
        self.REGION_NAME = os.getenv["AWS_REGION"] ## same as above for AWS_REGION
        self.BUCKET_NAME = os.getenv["AWS_BUCKET_NAME"] ## same as above for AWS_BUCKET_NAME
        self.KEY = "model"  # #  assigns the string "model" to the KEY attribute of the instance.
        self.ZIP_NAME = "model/artifacts.tar.gz" #  assigns the string "artifacts.tar.gz" to the ZIP_NAME attribute of the instance.
        self.ARTIFACTS_ROOT = os.path.join(from_root(), "artifacts") ## creating artifacts folder in the current directory
        self.ARTIFACTS_PATH = os.path.join(from_root(), "artifacts", "artifacts.tar.gz") # storing the artifacts.tar.gz in the artifacts folder.

    def get_aws_storage_config(self):
        return self.__dict__  ## returns a dictionary of the above instance's attribute


# Label Should Update from MongoDb
class PredictConfig: ## class name is PredictConfig that defines the input to model creation.
    def __init__(self):#__init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        self.LABEL = 101 ## total number of labels of images is 101
        self.REPOSITORY = 'pytorch/vision:v0.10.0' #set the REPOSITORY variable to the PyTorch vision library version v0.10.0.
        self.BASE_MODEL = 'resnet18' ## algorithm used is resnet-18
        self.PRETRAINED = True # pretraining is set to true
        self.IMAGE_SIZE = 256 # size of image will be 256
        self.EMBEDDINGS_LENGTH = 256 ## embeddings length is 256
        self.SEARCH_MATRIX = 'euclidean' ## SEARCH_MATRIX distance is calculated through the euclidean.
        self.NUMBER_OF_PREDICTIONS = 20  ## NUMBER_OF_PREDICTIONS is set to 20 for 20 images to be predicted
        self.STORE_PATH = os.path.join(from_root(), "artifacts") ## the predicted images will be stored into the artifacts folder in the root directory
        self.MODEL_PATHS = [(os.path.join(from_root(), "artifacts", "embeddings.ann"), "embeddings.ann"), 
                            (os.path.join(from_root(), "artifacts", "model.pth"), "model.pth")] 
        """The first value in the tuple (os.path.join(from_root(), "artifacts", "embeddings.ann")) is the path to 
        the file that contains the embeddings index. This file path is used by the program to load the embeddings index when needed.
        The second value in the tuple ("embeddings.ann") is a user-friendly name that can be used to refer to the
        embeddings index file. This name is used to display information to the user, such as when the embeddings index is loaded or when a search is performed."""
                            

    def get_pipeline_config(self):
        return self.__dict__  ## returns a dictionary of the above instance attributes.












