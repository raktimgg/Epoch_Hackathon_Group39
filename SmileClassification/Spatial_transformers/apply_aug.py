import cv2 as cv2
import numpy as np
# import matplotlib.pyplot as plt
import random
import math
from PIL import ImageEnhance
from scipy import ndimage
from skimage.transform import warp, AffineTransform
from skimage import data
import matplotlib.pyplot as plt
import os

from aug import random_brightness, rotate, affine_shear, affine_translate, flipv

def select_aug(ind):
	if(ind == 0):
		return random_brightness
	if(ind == 1):
		return rotate
	if(ind == 2):
		return affine_shear
	if(ind == 3):
		return affine_translate
	if(ind == 4):
		return flipv

loc = '/home/scitech/Hackathon/SmileDetection/data/train/smile'
imgs = os.listdir(loc)
l = len(imgs)
N = 3500
r = int(N/l) - 1
for i in range(r):
	print(i)
	for img_loc in imgs:
		loc1 = os.path.join(loc,img_loc)
		name = loc1[:-4]
		img = plt.imread(loc1)
		ind = np.random.randint(0,5)
		augm = select_aug(ind)
		img = augm(img)
		new_name = name+'_'+str(ind)+'.jpg'
		plt.imsave(new_name,img)
