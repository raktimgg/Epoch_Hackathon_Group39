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

# class Net1(nn.Module):
# 	def __init__(self,out_dim):
# 		super(Net1,self).__init__()
# 		self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
# 		self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
# 		self.bn1 = nn.BatchNorm2d(32)

# 		self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
# 		self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
# 		self.bn2 = nn.BatchNorm2d(128)

# 		self.fc1 = nn.Linear(8192,512)
# 		self.bn3 = nn.BatchNorm1d(512)
# 		self.drop = nn.Dropout(0.5)
# 		self.fc2 = nn.Linear(512,out_dim)

# 	def forward(self,x):
# 		x = F.relu(self.conv1(x))
# 		x = self.bn1(F.max_pool2d(F.relu(self.conv2(x)),2))
# 		x = F.relu(self.conv3(x))
# 		x = self.bn2(F.max_pool2d(F.relu(self.conv4(x)),2))
# 		# print(x.shape)
# 		x = torch.flatten(x,start_dim = 1)
# 		# print(x.shape)
# 		x = self.bn3(F.leaky_relu(self.fc1(x)))
# 		x = self.drop(x)
# 		x = self.fc2(x)
# 		return F.log_softmax(x, dim=1)


model = Net(1).to(device)
model.load_state_dict(torch.load('model_53_0.9992416582406471', map_location = 'cpu'))
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False).to(device)
# model.fc = nn.Linear(512,43).to(device)
# print(model.fc)
# print(model)

optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

test_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder('data/test',transform=data_transforms),
	batch_size=128, shuffle=False, num_workers=1, pin_memory=use_gpu)

# print(len(test_loader.dataset))

def test():
	model.eval()
	res = []
	for idx, (x, y) in enumerate(test_loader): 
		x = x.to(device)
		y = y.to(device)
		y_pred = model(x)
		# y_pred = y_pred
		max_index = y_pred.max(dim = -1)[1]
		max_index = max_index.cpu().detach().numpy()
		res.append(max_index)
		sys.stdout.write('\r'+str(idx)+'/'+str(len(test_loader)))
	return np.hstack(res)

res = test()