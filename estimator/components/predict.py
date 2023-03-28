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

        self.ann = CustomAnnoy(self.config.EMBEDDINGS_LENGTH, # instance of the CustomAnnoy class is created, which
        # is used for searching and retrieving embeddings from the model, it takes two input arguments EMBEDDINGS_LENGTH=256,
        # SEARCH_MATRIX is euclidean.
                               self.config.SEARCH_MATRIX)

        self.ann.load(self.config.MODEL_PATHS[0][0]) #method is called to load a pre-trained model from a specified path.
        self.estimator = self.load_model() #instance of the load_model method is created 
        self.estimator.eval() # method is called to put the model in evaluation mode, which may disable certain features like dropout or batch normalization.
        self.transforms = self.transformations() #method is called, which applies any necessary transformations to the input data before feeding it into the model.

    @staticmethod
    def initial_setup():    
        """ method to set up the project environment by creating necessary directories and retrieving any 
    required data or packages from a remote source."""    
        if not os.path.exists(os.path.join(from_root(), "artifacts")): ## checks if the artifact folder exist in root folder
            os.makedirs(os.path.join(from_root(), "artifacts")) ## creates folder artifact is it does not exist
        connection = StorageConnection() # creates an instance of StorageConnection class  Created connection with S3 bucket using boto3 api to fetch the model from Repository.
        connection.get_package_from_testing() # calls method get_package_from_testing that downloads the model artifacts 
        #from an S3 bucket by creating a connection with the bucket using boto3 and then downloading the ZIP file containing the model artifacts.

    def load_model(self):
        """The load_model method loads the trained PyTorch model and returns a new model that is the same as the
        trained model, but with the final output layer removed."""
        model = NeuralNet() #instance of the NeuralNet class is created which contains Replica of the neural network used while training.
        model.load_state_dict(torch.load(self.config.MODEL_PATHS[1][0], map_location=self.device)) # load_state_dict()
#method loads the trained model parameters into the new model instance. torch.load() is a function used to load a saved model
#input the file path of the saved model or checkpoint and returns a dictionary object that contains the model parameters and other information that was saved with the model.
# The map_location parameter specifies the device where the model should be loaded in CPU or GPU.
        return nn.Sequential(*list(model.children())[:-1])
        """The final output layer is removed from the loaded 
        model by creating a new Sequential model instance and copying all the layers from the original model
        except the final one, which is removed using the list slicing notation [:-1]. This is because the 
        final layer is specific to the training task, and is not necessary for making predictions with the model."""

    def transformations(self): ## This method  the transforms module provides a set of common image transformations 
        """
            Transformation Method Provides TRANSFORM_IMG object. Its pytorch's transformation class to apply on images.
            :return: TRANSFORM_IMG
            """
        TRANSFORM_IMG = transforms.Compose(  ## Composes several transforms together.
            [transforms.Resize(self.config.IMAGE_SIZE),## resize the image to shape to 256
             transforms.CenterCrop(self.config.IMAGE_SIZE), ## This transformation takes a PIL (Python Imaging Library) image as input and 
#returns a new image of size (256, 256) by cropping the input image from its center. The transformation ensures
#  that the center of the image is preserved while removing the outer edges of the image.
             transforms.ToTensor(), ##Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
#Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
             transforms.Normalize(mean=[0.485, 0.456, 0.406], #This transformation is typically applied to the image data after it has been resized or cropped, and before it is fed into the neural network.
#The mean and std arguments are lists of length 3, corresponding to the mean and standard deviation of the pixel values for the red, green, and blue 
# channels, respectively. These values are usually calculated based on the training dataset and are used to normalize the pixel values of the input images so that they have a similar scale and range.
                                  std=[0.229, 0.224, 0.225])]
        )

        return TRANSFORM_IMG ## return the transformed image parameters.

    def generate_embeddings(self, image):
        """method  that generates embeddings for an image"""
        image = self.estimator(image.to(self.device)) #image as input and applies the PyTorch model self.estimator to the image on the self.device 
        #which is probably a CPU). The output is a NumPy array containing the embeddings for the image.
        image = image.detach().cpu().numpy()  #returns the NumPy array representation of the tensor after it has been detached from the computation graph and moved to the CPU.
        return image ## image in array

    def generate_links(self, embedding):
        """The generate_links function takes an embedding vector and returns the self.config.NUMBER_OF_PREDICTIONS
        nearest neighbor links from the database, as determined by the ann index. The ann index is created using 
        the annoy library and is built on the embeddings of the images in the database. The get_nns_by_vector 
        function is used to get the nearest neighbors of the given embedding."""
        return self.ann.get_nns_by_vector(embedding, self.config.NUMBER_OF_PREDICTIONS)

    def run_predictions(self, image):
        """method that takes an image as input and returns the links of similar images."""
        image = Image.open(io.BytesIO(image)) # it opens the image using the PIL library's Image module and reads the image data from a bytes buffer.
        if len(image.getbands()) < 3: # it checks if the image has less than 3 color channels, which indicates it's a grayscale image. 
            image = image.convert('RGB') #If so, it converts it to an RGB image.
        image = torch.from_numpy(np.array(self.transforms(image)))# applies the image transformation using the transform function defined earlier.
        image = image.reshape(1, 3, 256, 256) #reshapes the transformed image to have the required shape of (batch_size, channels, height, width)
        embedding = self.generate_embeddings(image) #generates an embedding for the image using the generate_embeddings function.
        return self.generate_links(embedding[0]) #it returns the links of similar images based on the embedding using the generate_links function.









