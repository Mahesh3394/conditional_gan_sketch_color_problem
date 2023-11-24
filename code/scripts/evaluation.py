# importing library
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
from IPython import display
import math

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D,Dropout,Conv2DTranspose,BatchNormalization,LeakyReLU,Concatenate,Activation
from tensorflow.keras.layers import Reshape, GlobalAveragePooling2D, ReLU
from tensorflow.keras.models import load_model
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datetime import datetime
import glob

#setting GPU Configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = .25

def generator(img_shape):
    
    '''
    This function takes the shape of image as input and returns the generator model as U-net architecture.
    '''
    
    k_in = RandomNormal(stddev=0.02)
    
    # image input
    sketch_img = Input(shape=(img_shape))
    
    # encoder
    gen_en1 = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(sketch_img)
    gen_en1 = LeakyReLU(alpha=0.2)(gen_en1)
    
    gen_en2 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_en1)
    gen_en2 = BatchNormalization()(gen_en2, training=True)
    gen_en2 = LeakyReLU(alpha=0.2)(gen_en2)
    
    gen_en3 = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_en2)
    gen_en3 = BatchNormalization()(gen_en3, training=True)
    gen_en3 = LeakyReLU(alpha=0.2)(gen_en3)
    
    gen_en4 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_en3)
    gen_en4 = BatchNormalization()(gen_en4, training=True)
    gen_en4 = LeakyReLU(alpha=0.2)(gen_en4)
    
    gen_en5 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_en4)
    gen_en5 = BatchNormalization()(gen_en5, training=True)
    gen_en5 = LeakyReLU(alpha=0.2)(gen_en5)
    
    gen_en6 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_en5)
    gen_en6 = BatchNormalization()(gen_en6, training=True)
    gen_en6 = LeakyReLU(alpha=0.2)(gen_en6)
    
    
    # bottleneck
    bttl = Conv2D(512, (4,4), strides=(2,2), padding='same',activation="relu", kernel_initializer=k_in)(gen_en6)
    
    # decoder
    gen_de1 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(bttl)
    gen_de1 = BatchNormalization()(gen_de1, training=True)
    gen_de1 = Dropout(0.5)(gen_de1, training=True)
    gen_de1 = Concatenate()([gen_de1, gen_en6])
    gen_de1 = Activation('relu')(gen_de1)
    
    gen_de2 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_de1)
    gen_de2 = BatchNormalization()(gen_de2, training=True)
    gen_de2 = Dropout(0.5)(gen_de2, training=True)
    gen_de2 = Concatenate()([gen_de2, gen_en5])
    gen_de2 = Activation('relu')(gen_de2)
    
    gen_de3 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_de2)
    gen_de3 = BatchNormalization()(gen_de3, training=True)
    gen_de3 = Dropout(0.5)(gen_de3, training=True)
    gen_de3 = Concatenate()([gen_de3, gen_en4])
    gen_de3 = Activation('relu')(gen_de3)
    
    gen_de4 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_de3)
    gen_de4 = BatchNormalization()(gen_de4, training=True)
    gen_de4 = Concatenate()([gen_de4, gen_en3])
    gen_de4 = Activation('relu')(gen_de4)
    
    gen_de5 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_de4)
    gen_de5 = BatchNormalization()(gen_de5, training=True)
    gen_de5 = Concatenate()([gen_de5, gen_en2])
    gen_de5 = Activation('relu')(gen_de5)
    
    gen_de6 = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(gen_de5)
    gen_de6 = BatchNormalization()(gen_de6, training=True)
    gen_de6 = Concatenate()([gen_de6, gen_en1])
    gen_de6 = Activation('relu')(gen_de6)
    
    
    # output
    gen_col_img = Conv2DTranspose(3, (4,4), strides=(2,2), activation= 'tanh',padding='same', kernel_initializer=k_in)(gen_de6)
    
    # define model
    model = Model(sketch_img, gen_col_img)
    return model

def test_plot(g_model, sketch_img, color_img,img_name):
    '''
    Predict fake images using generator model and plot the [sketch,real,fake] image
    and further saves the figure
    
    '''
    # generating fake images
    fake_col_img = g_model.predict(sketch_img)
    
    # rescaling image [0-1]
    sketch_img = 0.5 * sketch_img + 0.5
    color_img = 0.5 * color_img + 0.5
    fake_col_img = 0.5 * fake_col_img + 0.5
    
    fig, axs = plt.subplots(sketch_img.shape[0], 3, figsize=(.8 * sketch_img.shape[0],5 * sketch_img.shape[0]))
    axs = axs.flatten()
    cnt = 0
    for sketch, real, fake in zip(sketch_img, color_img, fake_col_img):
        axs[cnt].imshow(sketch)
        axs[cnt].set_title('sketch')
        axs[cnt].axis('off')
        
        axs[cnt+1].imshow(real)
        axs[cnt+1].set_title('real_color_image')
        axs[cnt+1].axis('off')
        
        axs[cnt+2].imshow(fake)
        axs[cnt+2].set_title('fake_generated')
        axs[cnt+2].axis('off')
        cnt += 3
    fig.savefig("{}.png".format(img_name))
    plt.subplots_adjust(wspace=.1, hspace=.1)
    plt.show()
    
def print_test(color_img_path,sketch_path,total_test_img,best_generator,index):
    '''
    Take the random sketch and real image pair and generate the corresponding fake images 
    '''
    # load best generator model
    best_g_model = load_model("../model/generator/{}".format(best_generator))
    color_img = []
    sketch_img = []
    for sketch, image in zip(sketch_path[index], color_img_path[index]):
        # reading images
        sketch_img.append(np.array(Image.open(sketch).convert('RGB')))
        color_img.append(np.array(Image.open(image).convert('RGB')))

    # normalize [-1,1]
    sketch_img = np.array(sketch_img, dtype='float32')/127.5 - 1
    color_img = np.array(color_img, dtype='float32')/127.5 - 1
    
    # plotting the result
    test_plot(best_g_model,sketch_img,color_img, best_generator.split('.')[0])
    
    
with open("../config_documentation/config.json",r) as file:
    config_file = json.load(file)
    
best_generator = config_file["Model_params"]["best_generator"]
    
# loading unseen test data 
color_img_path = glob.glob('test_set/src/*.png')
sketch_path = glob.glob('test_set/sketch/*.png')

color_img_path = np.array(color_img_path)
sketch_path = np.array(sketch_path)
total_test_img = color_img_path.shape[0]

# randomly taking 20 sketch-color pair from the test set.
index = np.random.randint(0, total_test_img, 20)

#checking for generator_35
print_test(color_img_path,sketch_path,total_test_img,best_generator,index)