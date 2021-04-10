import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys
import time
from data1 import initialize_data, data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_hflip,data_vflip
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

device = 'cuda'
# device = 'cpu'
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
# 		self.conv1 = nn.Conv2d(3, 50, kernel_size=5)
# 		self.conv2 = nn.Conv2d(50, 100, kernel_size=3)
# 		self.conv3 = nn.Conv2d(100, 150, kernel_size=3)
# 		self.drop = nn.Dropout2d()
# 		self.fc1 = nn.Linear(1350, 600)
# 		self.fc2 = nn.Linear(600, 300)
# 		self.fc3 = nn.Linear(300, 100)
# 		self.fc4 = nn.Linear(100, out_dim)

# 	def forward(self,x):
# 		x = F.max_pool2d(F.leaky_relu(self.conv1(x)),2)
# 		x = F.max_pool2d(F.leaky_relu(self.conv2(x)),2)
# 		x = F.max_pool2d(F.leaky_relu(self.conv3(x)),2)
# 		x = torch.flatten(x,start_dim = 1)
# 		# print(x.shape)
# 		x = F.leaky_relu(self.fc1(x))
# 		x = F.leaky_relu(self.fc2(x))
# 		x = F.leaky_relu(self.fc3(x))
# 		x = self.fc4(x)
# 		return F.log_softmax(x, dim=1)


# class Net1(nn.Module):
# 	def __init__(self,out_dim):
# 		super(Net1,self).__init__()
# 		self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
# 		self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
# 		self.bn1 = nn.BatchNorm2d(32)

# 		self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
# 		self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
# 		self.bn2 = nn.BatchNorm2d(128)

# 		self.fc1 = nn.Linear(61952,512)
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
# 		# print(x.shape)
# 		x = self.bn3(F.leaky_relu(self.fc1(x)))
# 		x = self.drop(x)
# 		x = self.fc2(x)
# 		return F.sigmoid(x)


model = Net(1).to(device)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=False).to(device)
# model.classifier[1] = nn.Conv2d(512, 3, kernel_size=(1,1), stride=(1,1)).to(device)

# model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True).to(device)
# model.classifier[6] = nn.Linear(4096,1).to(device)
# print(model)

# model = torch.hub.load(
#     'moskomule/senet.pytorch',
#     'se_resnet20',
#     num_classes=3).to(device)

# print(model.fc)
# print(model)
# print('Number of trainable parameters :',sum(p.numel() for p in model.parameters() if p.requires_grad))
 
# optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
optimizer = torch.optim.RMSprop(model.parameters(),lr = 0.0001)

# data_transforms = transforms.Compose([
# 	transforms.Resize((45, 45)),
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
# ])

train_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder('/home/scitech/Hackathon/SmileDetection/data/train',transform=data_transforms),
	batch_size=64, shuffle=True, num_workers=1, pin_memory=use_gpu)




# train_loader = torch.utils.data.DataLoader(
# 	torch.utils.data.ConcatDataset([datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_transforms),
# 	datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_jitter_brightness),datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_jitter_hue),datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_jitter_contrast),datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_jitter_saturation),datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_translate),datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_rotate),datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_hvflip),datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_center),datasets.ImageFolder('/home/scitech/Hackathon/data/train',
# 	transform=data_shear)]), batch_size=128, shuffle=True, num_workers=4, pin_memory=use_gpu)

val_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder('/home/scitech/Hackathon/SmileDetection/data/test',transform=data_transforms),
	batch_size=64, shuffle=True, num_workers=1, pin_memory=use_gpu)

# print(len(train_loader.dataset))

loss_fn = nn.BCELoss()

def train(epoch):
	model.train()
	l = 0
	correct = 0
	for idx, (x, y) in enumerate(train_loader): 
		x = x.to(device)
		y = y.to(device)
		y = y.float()
		# print(x[0])
		model.zero_grad()
		y_pred = model(x)[:,0]
		# print(y_pred.shape)
		# print(y[:,None].shape)
		# loss = F.nll_loss(y_pred,y)
		# loss = F.cross_entropy(y_pred,y)
		loss = loss_fn(y_pred,y)
		loss.backward()
		optimizer.step()
		l += loss.item()
		max_index = (y_pred>0.5)*1.0
		correct += ((max_index == y).sum()).item()
		sys.stdout.write('\rEpoch: '+str(epoch)+' Batch: '+str(idx+1)+'/'+str(len(train_loader))+' Training Loss: '+str(loss.item()))
	return (l/idx), correct/len(train_loader.dataset)


def val(epoch):
	model.eval()
	l = 0
	correct = 0
	for idx, (x, y) in enumerate(val_loader): 
		x = x.to(device)
		y = y.to(device)
		y = y.float()
		y_pred = (model(x)[:,0])
		# loss = F.nll_loss(y_pred,y)
		# loss = F.cross_entropy(y_pred,y)
		loss = loss_fn(y_pred,y)
		l += loss.item()
		max_index = (y_pred>0.5)*1.0
		if(idx == 0):
			true = y
			pred = max_index
		else:
			true = torch.cat([true,y],dim = -1)
			pred = torch.cat([pred,max_index],dim = -1)
		correct += ((max_index == y).sum()).item()
		# print(max_index)
		# print(max_index)
		# print(y)
	if(epoch%5==0):
		true = true.detach().cpu().numpy()
		pred = pred.detach().cpu().numpy()
		cm = confusion_matrix(true,pred)
		print(' ')
		print(cm)
	return (l/idx), correct/len(val_loader.dataset)


EPOCH = 100

val1 = 0
sv = []
for epoch in range(EPOCH):
	time1 = time.time()
	tr_loss, tr_acc = train(epoch)
	time2 = time.time()
	val_loss, val_acc = val(epoch)
	print(' Epochs:',epoch,'Training Loss:',tr_loss,'Val Loss:',val_loss,'Training Acc:',tr_acc,'Val Acc:',val_acc,'Time :',time2-time1)

	sv.append([tr_loss,tr_acc,val_loss,val_acc])

	np.savetxt('res.csv',np.array(sv))

	if(val_acc>val1):
		sv_file = 'model_'+str(epoch)+'_'+str(val_acc)
		torch.save(model.state_dict(), sv_file)
		print('Validation Accuracy increased, Model Saved')
		val1 = val_acc

