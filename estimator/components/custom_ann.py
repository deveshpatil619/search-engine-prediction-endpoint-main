from annoy import AnnoyIndex #AnnoyIndex class is the core class of the Annoy library and is used to build and 
#query Annoy indexes. An Annoy index is a data structure that allows for efficient nearest neighbor searches in 
# high-dimensional spaces by partitioning the space into small hyper-rectangles (known as "nodes") and indexing the
# points based on the nodes they belong to.
from typing import Literal # The Literal class is useful for enforcing strict type constraints in your code
import json #json exposes an API familiar to users of the standard library marshal and pickle modules.


class CustomAnnoy(AnnoyIndex): # class called CustomAnnoy that inherits from the AnnoyIndex class.
    """
    Inherits AnnoyIndex: The save and load functions have been modified according to the website needs.
    """
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]):
# constructor for the CustomAnnoy class. It takes two arguments: f, which is an integer, and metric,
#  which is a Literal type that can only take on the values "angular", "euclidean", "manhattan", "hamming", or "dot".
        super().__init__(f, metric)#It calls the constructor of the AnnoyIndex class with these arguments
        self.label = []  #initializes an empty list called self.label.

    # noinspection PyMethodOverriding
    def add_item(self, i: int, vector, label: str) -> None: #This method overrides the add_item method of the AnnoyIndex class
##takes three arguments: i, which is an integer, vector, which can be any type, and label, which is a string          
        super().add_item(i, vector) ## calls the add_item method of the parent class with the first two arguments
        self.label.append(label)# appends the label argument to the self.label list.

    def get_nns_by_vector(self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...): 
        ##  method overrides the get_nns_by_vector method of the AnnoyIndex class.
        #It takes three arguments: vector, which can be any type, n which is an integer, and search_k and include_distances, which are optional arguments with default values.
        
        indexes = super().get_nns_by_vector(vector, n) #calls the get_nns_by_vector method of the parent class with these arguments,
        labels = [self.label[link] for link in indexes]  # then creates a new list called labels by iterating over
        #the indexes list and looking up the corresponding label value in the self.label list.
        return labels   #returns the labels list.

    def load(self, fn: str, prefault: bool = ...): #method overrides the load method of the AnnoyIndex class.
    #takes two arguments: fn, which is a string, and prefault, which is an optional boolean argument with a default value
        """
        Responsible for loading .ann and .json files saved by save method.
        """
        super().load(fn) ##calls the load() method of the superclass AnnoyIndex with the argument fn.
        path = fn.replace(".ann", ".json")#line creates a new string path by replacing the extension .ann in the fn argument with .json.
        self.label = json.load(open(path, "r")) #loads the JSON file at the path and assigns the result to the 
        # label instance variable of the CustomAnnoy` class.

    def save(self, fn: str, prefault: bool = ...):
        """another overides method of the CustomAnnoy class. It takes two arguments, a string fn representing the file name to 
        save and an optional boolean prefault. It saves the index to the specified file and also saves the
        corresponding labels to a JSON file.
        Responsible for Saving .ann and .json files.
        """
        super().save(fn) #calls the save() method of the superclass AnnoyIndex with the argument fn.
        path = fn.replace(".ann", ".json") #creates a new string path by replacing the extension .ann in the fn argument with .json.
        json.dump(self.label, open(path, "w")) #saves the label instance variable of the CustomAnnoy class to a JSON file at the path path.




