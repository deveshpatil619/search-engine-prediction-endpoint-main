import os
from from_root import from_root ## importing the library from_root to get the root directory of the project
from dotenv import load_dotenv ## importing the load_dotenv that loads all variables in .env file


class AwsStorage: ## class name is AwsStorage
    def __init__(self): #__init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        load_dotenv() # method that loads  environment variables from a .env file.
        self.ACCESS_KEY_ID = os.getenv["AWS_ACCESS_KEY_ID"]
        self.SECRET_KEY = os.getenv["AWS_SECRET_ACCESS_KEY"]
        self.REGION_NAME = os.getenv["AWS_REGION"]
        self.BUCKET_NAME = os.getenv["AWS_BUCKET_NAME"]
        self.KEY = "model"
        self.ZIP_NAME = "model/artifacts.tar.gz"
        self.ARTIFACTS_ROOT = os.path.join(from_root(), "artifacts")
        self.ARTIFACTS_PATH = os.path.join(from_root(), "artifacts", "artifacts.tar.gz")

    def get_aws_storage_config(self):
        return self.__dict__


# Label Should Update from MongoDb
class PredictConfig:
    def __init__(self):
        self.LABEL = 101
        self.REPOSITORY = 'pytorch/vision:v0.10.0'
        self.BASE_MODEL = 'resnet18'
        self.PRETRAINED = True
        self.IMAGE_SIZE = 256
        self.EMBEDDINGS_LENGTH = 256
        self.SEARCH_MATRIX = 'euclidean'
        self.NUMBER_OF_PREDICTIONS = 20
        self.STORE_PATH = os.path.join(from_root(), "artifacts")
        self.MODEL_PATHS = [(os.path.join(from_root(), "artifacts", "embeddings.ann"), "embeddings.ann"),
                            (os.path.join(from_root(), "artifacts", "model.pth"), "model.pth")]

    def get_pipeline_config(self):
        return self.__dict__
