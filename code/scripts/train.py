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
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from datetime import datetime
import glob

#setting GPU Configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = .25


def discriminator(img_shape):
    '''
    This function takes the shape of image as input and returns the discriminator as PatchGAN.
    '''
    
    # source image 
    source_image = Input(shape=(img_shape))
    # target image
    target_image = Input(shape=(img_shape))
    
    # defining kernel_initializer
    k_in = RandomNormal(stddev=0.02)
    
    # concatenating images
    con = Concatenate()([source_image, target_image])
    
    dis = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(con) #64x64x64
    dis = LeakyReLU(alpha=0.2)(dis)   
    
    dis = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(dis) #32x32x128
    dis = BatchNormalization()(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
   

    dis = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=k_in)(dis) #16x16x256
    dis = BatchNormalization()(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
    
    
    # second last layer
    dis = Conv2D(512, (2,2), kernel_initializer=k_in)(dis) #15x15x512
    dis = BatchNormalization()(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
    
    # last layer
    out = Conv2D(1, (2,2), activation = "sigmoid", padding='valid', kernel_initializer=k_in)(dis) #14x14x1
    
    final_out = GlobalAveragePooling2D()(out)
    
    # defining model
    model = Model([source_image, target_image], final_out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model

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

# Defining loss in model.complile keras; https://stackoverflow.com/a/45963039/9079093

# loss between feature map of target and generated image
def feature_loss(true, pred):
    '''
    parameter "true" is feature map of real color image,
    parameter "pred" is feature map of generated image, 
    it returns the euclidean distance between feature map 
    of real color image and generated image.
    '''
    def f_loss(y_true, y_pred):
        return K.mean( K.sqrt( K.sum( K.square( true - pred ))))
    return f_loss


# About variation loss https://www.tensorflow.org/api_docs/python/tf/image/total_variation
def variation_loss(pred):
    '''
    parameter "pred" is generated image, 
    it returns the square root of sum of differences between neighbouring pixels
    of geberated image
    '''
    def v_loss(y_pred):
        return K.sqrt(K.sum(K.square(pred[:, 1:, :, :] - pred[:, :-1, :, :]))\
                             + K.sum(K.square(pred [:, :, 1:, :] - pred [:, :, :-1, :])))
    return v_loss

# pixel wize loss between target and generated image
def pixel_loss(true, pred):
    '''
    parameter "true" is real color image, parameter "pred" is generated image,
    For each pixel of real color image and generated image it returns the average 
    of distance between them.
    '''
    
    def p_loss(y_true, y_pred):
        return K.mean( K.abs( true - pred ) )
    return p_loss

def define_gan(g_model, d_model, image_shape):
    
    '''
    Takes generator, discriminator model and image_shape as input, and returns the GAN model as output
    '''
    
#     def gan_loss(y_true, y_pred):
#         '''
#         returns the loss function of GAN.
#         '''
#         return tf.keras.losses.binary_crossentropy(y_true, y_pred) + pixel_loss_weight * pixel_loss_out(y_true, y_pred) +\
#                variation_loss_weight * variation_loss_out(y_pred) + feature_loss_weight * feature_loss_out(y_true, y_pred)

  
    d_model.trainable = False

    #Generator
    sketch_image = Input(image_shape)
    gen_output = g_model([sketch_image])

    #Discriminator
    dis_output = d_model([sketch_image, gen_output])
    
    #Pixel Loss
    color_image = Input(image_shape)
    pixel_loss_out = pixel_loss(color_image, gen_output)
  
    #Variation Loss
    variation_loss_out = variation_loss(gen_output)
    
    #Feature Loss
    vgg1_out = vgg_1([tf.image.resize(color_image, (128,128), tf.image.ResizeMethod.BILINEAR)])
    vgg2_out = vgg_2([tf.image.resize(gen_output, (128,128), tf.image.ResizeMethod.BILINEAR)])

    feature_loss_out = feature_loss(vgg1_out,vgg2_out)
  
    #Final Model
    model = Model(inputs=[sketch_image, color_image], outputs=dis_output)

    #Single output multiple loss functions in keras : https://stackoverflow.com/a/51705573/9079093

    model.compile(loss= 'binary_crossentropy', optimizer= Adam(lr=0.0002, beta_1=0.5))
    
    return model

def generate_real_images(sketch_path, color_image_path, n_sample):
    '''
    This function will read the data from paths provided and return the 
    actual sketch-color pair along with class label as "1" which denotes real images.
    '''
    
    index = np.random.randint(0, total_img, n_sample)
    sketch_image = []
    color_image = []
  
    for sketch, img in zip(sketch_path[index], color_image_path[index]):
        sketch_image.append(np.array(Image.open(sketch).convert('RGB')))
        color_image.append(np.array(Image.open(img).convert('RGB')))
  
    # Normalizing the values to be between [-1, 1].
    sketch_image = (np.array(sketch_image, dtype='float32')/127.5 - 1)
    color_image = (np.array(color_image, dtype='float32')/127.5 - 1)
    
    y_real = np.ones((n_sample,1))

    return sketch_image, color_image, y_real

def generate_fake_images(g_model, sketch_path, n_sample , seed_sketch=None, seed_color=None):
    '''
    This function will read only sketch data, and generate fake data using generator and return
    fake sketch-color pair along with class label as "0" which denotes fake images.
    '''
    
    sketch_image = []
    fake_color_image = []
    
    if seed_sketch is not None and seed_color is not None:
        fake_color_image = g_model.predict(seed_sketch)
        return seed_color, fake_color_image
    
    else:
        index = np.random.randint(0, total_img, n_sample)

        for sketch in sketch_path[index]:
            sketch_image.append(np.array(Image.open(sketch).convert('RGB')))
        
        # Normalizing the values to be between [-1, 1].
        sketch_image = (np.array(sketch_image, dtype='float32')/127.5 - 1)

        fake_color_image = g_model.predict(sketch_image)
        y_fake = np.zeros((n_sample, 1))

        return sketch_image, fake_color_image, y_fake
        
    

def summary_save_plot(epoch, g_model, sketch_path, seed_sketch, seed_color, n_sample=8):
    '''
    This Function will plot the real color image along with generated color image, and saves the plot
    '''

    real_color_image, fake_color_image = generate_fake_images(g_model, sketch_path, n_sample, seed_sketch=seed_sketch,\
                                                               seed_color=seed_color)

    row,col = 4,4
    
    gen_imgs = np.concatenate([fake_color_image, real_color_image])
    
    # Rescaling images to [0 - 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Generated','Generated','Original','Original']
    
    fig, axs = plt.subplots(row, col, figsize=(10,10))
    cnt = 0
    for i in range(row):
        for j in range(col):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("../../output/{}.png".format(epoch))
    plt.show()
    plt.close()


def save_model(epoch,g_model):
    '''
        Saves the Generator Model
    '''
    # saving models after each epoch
    if not os.path.isdir('../../model/generator'):
        os.makedirs('../../model/generator')
        
    g_model.save('../../model/generator/generator_{}.h5'.format(epoch))
    
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()


def train(g_model, d_model, gan_model, sketch_path, color_image_path, seed_sketch, seed_color, n_epoch, n_batch, initial_epoch):
    '''
    This function train the model over randomly sampled batch of real images 
    and fake images. Both discriminator and generator is trained alternatively.
    Results are displayed and generator model is saved after each epoch.
    '''
    batch_per_epoch = int(total_img / n_batch)
    half_batch = int(n_batch / 2) # this is created for discriminator as er are calling it twice so we pass half data.
  
    for i in range(initial_epoch, n_epoch):
        start2 = datetime.now()
        generator_loss = []
        discriminator_loss = []
    
        for j in range(batch_per_epoch):

            # discriminator model needs to be called twice
            if not j%2: # train only for odd value of j, to avoid over training
                
                # Discriminator real loss 
                real_sketch, real_col_image, y_real = generate_real_images(sketch_path, color_image_path, half_batch)
                d_loss_real = d_model.train_on_batch([real_sketch, real_col_image], y_real * .9)

            if not j%3: # train only when j is multiple of 3. to avoid over training
                
                # Discriminator fake loss 
                fake_sketch, fake_col_image, y_fake = generate_fake_images(g_model, sketch_path, half_batch)
                d_loss_fake = d_model.train_on_batch([fake_sketch, fake_col_image], y_fake)
                
            total_dis_loss = d_loss_real + d_loss_fake

            #GAN loss
            real_sketch, real_col_image, y_real = generate_real_images(sketch_path, color_image_path, n_batch)
            gan_loss = gan_model.train_on_batch([real_sketch, real_col_image], y_real)
            
            # saving loss per epoch
            discriminator_loss.append(total_dis_loss)
            generator_loss.append(gan_loss)

            
            # output after 100 iter
            if not j % 100:
                print("epoch>{}, {}/{}, d_real={:.3f}, d_fake={:.3f}, gan={:.3f}".format(i+1, j+1, batch_per_epoch,\
                                                                                         d_loss_real,d_loss_fake, gan_loss))

        # writing to tensorboard
#         write_log(disc_callback, 'discriminator_loss', np.mean(discriminator_loss), i+1, (i+1)%3==0)
#         write_log(gen_callback, 'generator_loss', np.mean(generator_loss), i+1, (i+1)%3==0)

        #Summary after every epoch.
        display.clear_output(True)
        print('Time for epoch {} : {}'.format(i+1, datetime.now()-start2))
        print('epoch>{}, {}/{}, d_real={:.3f}, d_fake={:.3f}, gan={:.3f}'.format(i+1, j+1, batch_per_epoch, d_loss_real,\
                                                                       d_loss_fake, gan_loss))
#         summary_save_plot(i, g_model, sketch_path, seed_sketch,seed_color, seed_color.shape[0])
        
        # saving model
        save_model(i,g_model)
    
    # Final summary
    display.clear_output(True)
    print('epoch>{}, {}/{}, d_real={:.3f}, d_fake={:.3f}, gan={:.3f}'.format(i+1, j+1, batch_per_epoch, d_loss_real,\
                                                                             d_loss_fake, gan_loss))
    summary_save_plot(i, g_model, sketch_path, seed_sketch, seed_color, seed_color.shape[0])

vgg16 = VGG16(weights='imagenet')

# Extracting intermediate layer features keras : https://keras.io/applications/#vgg16

vgg_1 = Model(inputs=vgg16.input, outputs=ReLU()(vgg16.get_layer('block2_conv2').output))
vgg_2 = Model(inputs=vgg16.input, outputs=ReLU()(vgg16.get_layer('block2_conv2').output))

pixel_loss_weight = 100
variation_loss_weight = .0001
feature_loss_weight = .01

# checking the model
with open("../config_documentation/config.json",r) as file:
    config_file = json.load(file)

image_shape = config_file["Model_params"]["image_shape"]
n_epoch = config_file["Model_params"]["n_epoch"]
n_batch = config_file["Model_params"]["n_batch"]

g_model = generator(image_shape)

d_model = discriminator(image_shape)

color_img_path = glob.glob('../data/train_set/src/*.png')
sketch_path = glob.glob('../data/train_set/sketch/*.png')

# color_img_path = glob.glob('test_set/src/*.png')
# sketch_path = glob.glob('test_set/sketch/*.png')

color_image_path = np.array(color_img_path)
sketch_path = np.array(sketch_path)

total_img = color_image_path.shape[0]

# sampling data for the 
seed_sketch = []
seed_color = []
index = np.random.randint(0, total_img, 8)

for sketch, image in zip(sketch_path[index], color_image_path[index]):
    seed_sketch.append(np.array(Image.open(sketch).convert('RGB')))
    seed_color.append(np.array(Image.open(image).convert('RGB')))

#Normalizing the values to be between [-1, 1].
seed_sketch = (np.array(seed_sketch, dtype='float32')/127.5 - 1)
seed_color = (np.array(seed_color, dtype='float32')/127.5 - 1)

#Training
train(g_model, d_model, gan_model, sketch_path, color_image_path, seed_sketch, seed_color, n_epoch=n_epoch, n_batch=n_batch, initial_epoch=0)
