#Importing all necessary library
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm

# after unzipping we get a folder named "danbooru-sketch-pair-128x" 

with open("../config_documentation/config.json",r) as file:
    config_file = json.load(file)

path = config_file["Model_params"]["data_path"]

# checking folders inside "color" directory
path = path + "color"

folder_dict = dict()
for d in os.listdir(path+"/sketch"):
    folder_dict[d] = [img_name for img_name in os.listdir(path+"/sketch/"+d)]
    
sample_train = dict()
for key, val in folder_dict.items():
    index = random.sample(range(0, len(val)), 300)
    sample_train[key] = [val[i] for i in index]
    
def split_data(sample_dict,src,directory,dst):
    '''
        This function take a dictionary,directory name, source directory, destination_directory
        and copy its file from source to destination according to the data present in the dictionary. 
    '''
    for folder, image in sample_dict.items():
        source = '{}/{}/{}/'.format(src,directory, folder)
        destination = '../data/{}/{}/'.format(dst,directory)

        # we will check if the folder exists or not
        if not os.path.isdir(destination):
            os.makedirs(destination)
        for img in image:
            shutil.move(source+img,destination)
            
def split_train_data(sample_dict,src,directory,dst):
    '''
        This function take a dictionary,directory name, source directory, destination_directory
        and copy its file from source to destination according to the data present in the dictionary. 
    '''
    for _ , image in sample_dict.items():
        source = '../data/{}/{}/'.format(src,directory)
        destination = '../data/{}/{}/'.format(dst,directory)

        # we will check if the folder exists or not
        if not os.path.isdir(destination):
            os.makedirs(destination)
        for img in image:
            shutil.move(source+img,destination)
            
# moving 700 images from each folder of sketch directory to train_set
split_data(sample_train,path,"sketch","train_set")

# moving 700 images from each folder of src directory to train_set
split_data(sample_train,path,"src","train_set")

# sampling 40 images from each folder of train_set directory to test_set
sample_test = dict()
for key, val in sample_train.items():
    index = random.sample(range(0, 300), 40)
    sample_test[key] = [val[i] for i in index]
    
# moving 40 images from each folder of sketch directory to test_set
split_train_data(sample_test,"train_set","sketch","test_set")

# moving 40 images from each folder of src directory to test_set
split_train_data(sample_test,"train_set","src","test_set")
            
