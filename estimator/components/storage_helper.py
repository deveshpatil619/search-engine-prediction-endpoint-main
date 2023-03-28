from estimator.entity.config import AwsStorage ## importing the class AwsStorage
import tarfile #The tarfile module provides a way to read and write tar archives, which are a type of file format used to store multiple files in a single file.
from boto3 import Session #The boto3 module is a Python library for interacting with AWS services, including Amazon Simple Storage Service (S3).
import os  ## for os operations


class StorageConnection: ## class name is StorageConnection
    """
    Created connection with AWS S3 bucket using boto3 api to fetch the model from Repository.
    """
    def __init__(self): ## several important variables and methods are initialized:
        self.config = AwsStorage() # instance of class AwsStorage is created where all the secrets related to AWS are mentioned
        self.session = Session(aws_access_key_id=self.config.ACCESS_KEY_ID,
                               aws_secret_access_key=self.config.SECRET_KEY,
                               region_name=self.config.REGION_NAME)
        """creates a session object with the AWS access key ID, secret access key, and region name from the configuration. 
         """
        self.s3 = self.session.resource("s3") #It also creates an S3 resource object
        self.bucket = self.s3.Bucket(self.config.BUCKET_NAME) #sets the S3 bucket to be used for fetching the model.

    def get_package_from_testing(self):
        """This method downloads the model artifacts from an S3 bucket by creating a connection with the bucket 
        using boto3 and then downloading the ZIP file containing the model artifacts. """
        print("Fetching Artifacts From S3 Bucket .....")
        if os.path.exists(self.config.ARTIFACTS_ROOT + "embeddings.ann"): ## checks if the file embeddings.ann already exists in the directory
            os.remove(self.config.ARTIFACTS_ROOT + "embeddings.ann") ## if file is found then it is removed 

        if os.path.exists(self.config.ARTIFACTS_ROOT + "model.pth"): ## checks if the file model.pth exists already in the directory
            os.remove(self.config.ARTIFACTS_ROOT + "model.pth") ## if the file is found then it is removed

        if os.path.exists(self.config.ARTIFACTS_ROOT + "embeddings.json"): ## checks if the file embeddings.json is already in the directory
            os.remove(self.config.ARTIFACTS_ROOT + "embeddings.json") ## if the file is found then it is removed

        self.bucket.download_file(self.config.ZIP_NAME, self.config.ARTIFACTS_PATH) #The download_file method from 
        #the s3.Bucket class of the boto3 package is used to download the file. It takes two arguments: the name of
        #  the file in the S3 bucket, and the local path where the file should be downloaded.
        folder = tarfile.open(self.config.ARTIFACTS_PATH) # The tarfile.open method takes the path of the downloaded
        #file and returns a TarFile object, which can be used to extract its contents.
        folder.extractall(self.config.ARTIFACTS_ROOT) #extractall method of the TarFile object is used to extract all 
        #the contents of the zip file to the specified directory.
        folder.close() ## the file is closed
        os.remove(self.config.ARTIFACTS_PATH) #the downloaded zip file is removed using os.remove
        print("Fetching Completed !") 


if __name__ == "__main__":
    connection = StorageConnection() # creating the instance of StorageConnection class 
    connection.get_package_from_testing() ## running method get_package_from_testing from the object of that class.
 




 