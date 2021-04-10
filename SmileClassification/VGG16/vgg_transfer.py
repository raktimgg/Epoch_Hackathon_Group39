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
if tf.test.is_gpu_available()==True:
	print("GPU is available")
else:
	print("GPU is not available")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.applications.resnet50 import ResNet50
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
import modelss
num_classes = 2
def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img
def d_aug(img):
	D=[]
	a0,a1,a2,a3,a4,a5,a6=img[:,:,:],img[:,:,:],img[:,:,:],img[:,:,:],img[:,:,:],img[:,:,:],img[:,:,:]
	# ratio = 0.5
	# a1 = data_aug.vertical_shift(a1, 0.4)
	# # D.append(a1)
# 
	# a2 = data_aug.horizontal_shift(a2, 0.4) 
	# a3 = data_aug.brightness(a3, 0.6,1.5 )
	# a4 = data_aug.zoom(a4, 0.9)
	# a4 = data_aug.zoom(a4, 0.9)
	# r = np.random.randint(2)
	# if (r==0):
	a5=horizontal_flip(a5, True)
	# elif r==1:
	# a6=data_aug.vertical_flip(a5, True)
	# a6 =data_aug.channel_shift(a6, 30)
	# D=[a0, a1 , a2,a4]
	D=[a0,a5]
	return D

# model.compile(loss='categorical_crossentropy',
#                     optimizer=optimizers.Adam(),
#                     metrics=['accuracy'])
# make_Resnet_final()aaaaaaaaaaa
path = '/home/shahid/Desktop/Abhishek/data/train/'
folders = ['not','negative','positive']
# folders = ['negative','positive']
img_data=[]
labels = []
for (z,folder) in enumerate(folders,0):
	print(folder,z)
	pathn = path+folder
	os.chdir(pathn)
	filenames = os.listdir(pathn)
	for f in filenames:
		x=cv2.imread(f)
		x  = cv2.resize(x,(224,224), interpolation = cv2.INTER_CUBIC)
		if z ==1 or z==2 :
			D=d_aug(x)
		else:
			D=[x]
			# D=D[:3]

		for x in D:
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			x = x/255
			img_data.append(x)
			if z==2:
				labels.append(1)
			else:
				labels.append(z)
		
img_data = np.array(img_data)
img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)
# labels = np.expand_dims(labels, axis=0)
Y = np.array(labels) 
# Y=np.rollaxis(Y,1,0)
# Y = to_categorical(labels, num_classes)
# print(Y)
img_data_test=[]
labels_test = []
path = '/home/shahid/Desktop/Abhishek/data/test/'
for (z,folder) in enumerate(folders,0):
	print(folder,z)
	pathn = path+folder
	os.chdir(pathn)
	filenames = os.listdir(pathn)
	for f in filenames:
		x=cv2.imread(f)
		x  = cv2.resize(x,(224,224), interpolation = cv2.INTER_CUBIC)
		D = [x]
		for x in D:
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			x = x/255
			img_data_test.append(x)
			if z==2:
				labels_test.append(1)
			else:
				labels_test.append(z)
		
img_data_test = np.array(img_data_test)
img_data_test = img_data_test.astype('float32')
print (img_data_test.shape)
img_data_test=np.rollaxis(img_data_test,1,0)
print (img_data_test.shape)
img_data_test=img_data_test[0]
print (img_data_test.shape)
# Y_test = to_categorical(labels_test, num_classes)
# labels_test = np.expand_dims(labels_test, axis=0)
Y_test = np.array(labels_test) 
# Y_test=np.rollaxis(Y_test,1,0)
# print(Y_test)
X_train,y_train = shuffle(img_data,Y, random_state=2)
X_test,y_test = shuffle(img_data_test,Y_test, random_state=2)
img_data,img_data_test,labels,labels_test=[],[],[],[]

modelc = modelss.mvgg(num_classes)
# modelc = modelss.make_v==(num_classes)

modelc.summary()
path = "/home/shahid/Desktop/Abhishek/ch/modelsh/"
os.chdir(path)
print("training starts")
Model_name = 'modelvgg_t2'
path1 = path + Model_name +".h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path1,monitor='val_accuracy',save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)
# save_string = [Model_name, " Dataset = Sat ,used heavy data_aug, model - vgg16, architecture - added 3 layers , optimizer - adam , lr - 0.00001 batch_size = 8 , other points - training the last 6 layers , 5 classes"]
# savetxt(Model_name+'.txt', save_string, delimiter=" ", fmt="%s") 
csv_logger = CSVLogger(Model_name+'.csv', append=True, separator=',')
# opt = Adam(learning_rate=0.00001)
opt = Adam(learning_rate=0.0001)
modelc.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
hist = modelc.fit(X_train, y_train, batch_size=32 , epochs=25, verbose=1, validation_data=(X_test, y_test),callbacks=[ cp_callback,csv_logger])
# hist = modelc.fit([X_train,Xep_train], y_train, batch_size=16 , epochs=30, verbose=1, validation_data=([X_test,Xep_test], y_test),callbacks=[ cp_callback])
# modelc.save(Model_name+".h5")
(loss, accuracy) = modelc.evaluate(X_test, y_test, batch_size=12, verbose=1)
out = np.array(modelc.predict(X_test))
print(out)
result = (out>0.5)*1.0
print(len(result),np.sum(result))

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
