import argparse
import cv2
import numpy as np
from torchvision import models
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys
import time
from data1 import initialize_data, data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_hflip,data_vflip
import pandas as pd
from sklearn.metrics import f1_score
from pytorch_grad_cam import CAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

if __name__ == '__main__':
    """ python gradcam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
device = 'cuda'
device = 'cpu'
use_gpu = True


class Net(nn.Module):
	def __init__(self,out_dim):
		super(Net,self).__init__()
		# CNN layers
		self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
		self.bn1 = nn.BatchNorm2d(100)
		self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
		self.bn2 = nn.BatchNorm2d(150)
		self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
		self.bn3 = nn.BatchNorm2d(250)
		self.conv_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(250*10*10, 10000) ## change dim here
		self.fc2 = nn.Linear(10000, 512)
		self.fc3 = nn.Linear(512, out_dim)

		self.localization = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=7),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True),
			nn.Conv2d(8, 10, kernel_size=5),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True)
			)

		# Regressor for the 3 * 2 affine matrix
		self.fc_loc = nn.Sequential(
			nn.Linear(10 * 21 * 21, 32), ## change dim here
			nn.ReLU(True),
			nn.Linear(32, 3 * 2)
			)

		# Initialize the weights/bias with identity transformation
		self.fc_loc[2].weight.data.zero_()
		self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


	# Spatial transformer network forward function
	def stn(self, x):
		xs = self.localization(x)
		# print(xs.shape)
		xs = xs.view(-1, 10 * 21 * 21) ## change dim here
		theta = self.fc_loc(xs)
		theta = theta.view(-1, 2, 3)
		# print(theta.shape, x.shape)
		grid = F.affine_grid(theta, x.size())
		x = F.grid_sample(x, grid)
		return x

	def forward(self, x):
		# transform the input
		x = self.stn(x)

		# Perform forward pass
		x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
		x = self.conv_drop(x)
		x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
		x = self.conv_drop(x)
		x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
		x = self.conv_drop(x)
		x = x.view(-1, 250*10*10) ## change dim here
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# return F.log_softmax(x, dim=1)
		return F.sigmoid(x)


args = get_args()
model = Net(1).to(device)
model.load_state_dict(torch.load('model_31_0.8320754716981132',map_location = 'cpu'))
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
target_layer = model.conv2

cam = CAM(model=model, 
              target_layer=target_layer,
              use_cuda=args.use_cuda)

rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img,(100,100))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
target_category = None
grayscale_cam = cam(input_tensor=input_tensor, 
                        method=args.method,
                        target_category=target_category)

cam_image = show_cam_on_image(rgb_img, grayscale_cam)

gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
gb = gb_model(input_tensor, target_category=target_category)

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask * gb)
gb = deprocess_image(gb)

cv2.imwrite('cam2.jpg', cam_image)
cv2.imwrite('gb.jpg', gb)
cv2.imwrite('cam_gb.jpg', cam_gb)
