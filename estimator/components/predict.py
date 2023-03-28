import os ## for the os operations 
from from_root import from_root ## importing the library from_root to get the root directory of the project
from estimator.components.custom_ann import CustomAnnoy # importing the CustomAnnoy class  taht has the ANNOY algorith customized into it.
from estimator.components.storage_helper import StorageConnection #Importing the StorageConnection class that contains the AWS connection details
from estimator.entity.config import PredictConfig #importing class name is PredictConfig that defines the input to model creation.
from estimator.components.model import NeuralNet #importing the class called NeuralNet which inherits from the nn.Module
from torchvision import transforms # The transforms module from torchvision provides a set of common image transformations that can be applied to PIL images or tensors. 
from PIL import Image # The Image module from PIL (Python Imaging Library) is a third-party library that for opening, manipulating, and saving many different image file formats.  
from torch import nn # importing the neural network creators from the pytorch library
import numpy as np # importing numpy
import torch #importing the pytorch library
import io #In the context of machine learning and deep learning, io is commonly used to work with data in memory, 
#such as loading and saving models, reading and writing data to and from byte arrays, and transforming data between different formats.


class Prediction(object): #class takes input data in the form of Python objects PyTorch tensors.
    """
    Prediction class Prepares the model endpoint which is the interface between the model and any external systems
    that will interact with it.
    """
    def __init__(self): # several important variables and methods are initialized:
        self.config = PredictConfig()  # instance of PredictConfig class that defines the input to model creation.
        self.device = "cuda" if torch.cuda.is_available() else "cpu" #  variable is set to either "cuda" (if a GPU is available) or "cpu" (if not).
        self.initial_setup() #sets up any other components of the model or dependencies needed for the model to function.

        self.ann = CustomAnnoy(self.config.EMBEDDINGS_LENGTH,
                               self.config.SEARCH_MATRIX)

        self.ann.load(self.config.MODEL_PATHS[0][0])
        self.estimator = self.load_model()
        self.estimator.eval()
        self.transforms = self.transformations()

    @staticmethod
    def initial_setup():    
        """ method to set up the project environment by creating necessary directories and retrieving any 
    required data or packages from a remote source."""    
        if not os.path.exists(os.path.join(from_root(), "artifacts")): ## checks if the artifact folder exist in root folder
            os.makedirs(os.path.join(from_root(), "artifacts")) ## creates folder artifact is it does not exist
        connection = StorageConnection() # creates an instance of StorageConnection class  Created connection with S3 bucket using boto3 api to fetch the model from Repository.
        connection.get_package_from_testing()

    def load_model(self):
        model = NeuralNet()
        model.load_state_dict(torch.load(self.config.MODEL_PATHS[1][0], map_location=self.device))
        return nn.Sequential(*list(model.children())[:-1])

    def transformations(self):
        TRANSFORM_IMG = transforms.Compose(
            [transforms.Resize(self.config.IMAGE_SIZE),
             transforms.CenterCrop(self.config.IMAGE_SIZE),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]
        )

        return TRANSFORM_IMG

    def generate_embeddings(self, image):
        image = self.estimator(image.to(self.device))
        image = image.detach().cpu().numpy()
        return image

    def generate_links(self, embedding):
        return self.ann.get_nns_by_vector(embedding, self.config.NUMBER_OF_PREDICTIONS)

    def run_predictions(self, image):
        image = Image.open(io.BytesIO(image))
        if len(image.getbands()) < 3:
            image = image.convert('RGB')
        image = torch.from_numpy(np.array(self.transforms(image)))
        image = image.reshape(1, 3, 256, 256)
        embedding = self.generate_embeddings(image)
        return self.generate_links(embedding[0])









