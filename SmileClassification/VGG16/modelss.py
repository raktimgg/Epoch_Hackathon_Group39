############imports#########
print("starting code")
import random 
# import matplotlib.pyplot as plt
import os ,cv2
import time
import warnings,csv
# warnings.ignore()
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152.
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam , SGD ,RMSprop
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, concatenate, UpSampling2D, BatchNormalization, Activation, Cropping2D, ZeroPadding2D 
# import warnings
import random , numpy as np
# import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint,CSVLogger
from numpy import asarray
from numpy import savetxt
import tensorflow.keras
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dropout
import csv
# if tf.test.is_gpu_available()==True:
# 	print("GPU is available")
# else:
# 	print("GPU is not available")
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img
from tensorflow.keras.applications.imagenet_utils import preprocess_input , decode_predictions
from tensorflow.keras.layers import Dense, Activation, Flatten , GlobalAveragePooling2D ,Dropout,Conv2D, MaxPooling2D ,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
print("imports done succesfully")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)
# import data_aug ,just_test
##################################

###############Dataset_making#################3


######################
image_inputA = Input(shape=(224, 224, 3))
image_inputB = Input(shape=(224, 224, 3))
inputep= Input(shape=(2,))
def make_inception(num_classes):
	model = InceptionV3(input_tensor= image_inputA,include_top=True)
	last_layer = model.get_layer('avg_pool').output
	out = Dense(1, activation='sigmoid', name='output')(last_layer)
	custommodel = Model(inputs=image_inputA,outputs= out)
# summarize the model
	return custommodel

# make_incpetion(2)

def make_Resnet_final():
	model = ResNet50(input_tensor=image_inputA, include_top=True)
	print(model.summary())
	return model 
def make_vgg_base():
	model = VGG16(input_tensor=image_inputB, include_top=True,weights='imagenet')
	print(model.summary())
	return model
def make_resnet(num_classes):
	model = ResNet50(input_tensor=image_inputA, include_top=True)

	last_layer = model.get_layer('avg_pool').output
	# out = Dense(num_classes, activation='softmax', name='output_layer')(last_layer)
	out = Dense(1, activation='sigmoid', name='output')(last_layer)
	custom_resnet_model = Model(inputs=image_inputA,outputs= out)
	# print(custom_resnet_model.summary())
	return custom_resnet_model

def mep():
	y = Dense(4, activation="relu", name ="d")(inputep)
	y = Dense(16, activation="relu" ,name ="de")(y)
	y = Dense(256, activation="relu" ,name ="dee")(y)
	y = Model(inputs=inputep, outputs=y)
	print(y.summary())
	return y
# m2()
# mep()
def modelwithep(num_classes):
	model1 = make_resnet(num_classes)
	model2= mep()
	last_layer2 = model2.get_layer('dee').output
	last_layer1 = model1.get_layer('flatten').output
	x = tensorflow.keras.layers.concatenate([last_layer1, last_layer2], axis = -1)
	x = Dense(4096, activation='relu', name='fc0')(x)
	x = Dense(4096, activation='relu', name='fc00')(x)
	x = Dense(4096, activation='relu', name='fc000')(x)
	out = Dense(num_classes, activation='softmax', name='output')(x)
	model3 = Model([image_inputA, inputep], out)
	return model3

def mvgg(num_classes):
	model = VGG16(input_tensor=image_inputB, include_top=True,weights='imagenet')
	last_layer = model.get_layer('fc2').output
	for layer in model.layers:
		layer.trainable = False
	
	out = Dense(1, activation='sigmoid', name='output',trainable = True)(last_layer)
	custom_vgg_model2 = Model(image_inputB, out)
	# custom_vgg_model2.summary()
	# for layer in custom_vgg_model2.layers[:-3]:
	# 	layer.trainable = False
	# custom_vgg_model2.summary()
	print("model done")
	return custom_vgg_model2
def modelwithseg(num_classes):
	model1 = make_resnet(num_classes)
	model2= mvgg(num_classes)
	last_layer1 = model1.get_layer('flatten').output
	last_layer2 = model2.get_layer('flattenvgg').output
	x = tensorflow.keras.layers.concatenate([last_layer1, last_layer2], axis = -1)
	x = Dense(4096, activation='relu', name='fc0')(x)
	x = Dense(4096, activation='relu', name='fc00')(x)
	x = Dense(4096, activation='relu', name='fc000')(x)
	out = Dense(num_classes, activation='softmax', name='output')(x)
	model3 = Model([image_inputA, image_inputB], out)
	return model3

def make_alex(num_classes):
	x= Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid')(image_inputB)
	x= Activation('relu')(x)
	# Max Pooling
	x= MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

	# 2nd Convolutional Layer
	x= Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid')(x)
	x= Activation('relu')(x)
	# Max Pooling
	x= MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

	# 3rd Convolutional Layer
	x= Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
	x= Activation('relu')(x)

	# 4th Convolutional Layer
	x= Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
	x= Activation('relu')(x)

	# 5th Convolutional Layer
	x= Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
	x= Activation('relu')(x)
	# Max Pooling
	x= MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

	# Passing it to a Fully Connected layer
	x= Flatten(name="flatalex")(x)
	# out = Dense(num_classes, activation='softmax', name='output')(x)
	out = Dense(1, activation='sigmoid', name='output')(x)
	model = Model(inputs=image_inputB, outputs=out)
	# 1st Fully Connected Layer
	

	# model.summary()
	return model
def modelwithseg2(num_classes):
	model1 = make_resnet(num_classes)
	model2= make_alex()
	last_layer1 = model1.get_layer('flatten').output
	last_layer2 = model2.get_layer('flatalex').output
	x = tensorflow.keras.layers.concatenate([last_layer1, last_layer2], axis = -1)
	x = Dense(4096, activation='relu', name='fc0')(x)
	x = Dense(4096, activation='relu', name='fc00')(x)
	x = Dense(4096, activation='relu', name='fc000')(x)
	out = Dense(num_classes, activation='softmax', name='output')(x)
	model3 = Model([image_inputA, image_inputB], out)
	return model3

# mvgg(3)
# make_Resnet_final()
# make_resnet(3)