import torch
import torchvision.transforms as tf

class driverNetMk1(torch.nn.Module):
    def __init__(self, numChannels, numClasses):
        super(driverNetMk1, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=numChannels, out_channels=20, 
                                    kernel_size=(5, 5))
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(10, 10), stride=(10, 10))
		# initialize second set of CONV => RELU => POOL layers
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
        self.fc1 = torch.nn.Linear(in_features=10500, out_features=2000)
        self.relu3 = torch.nn.ReLU()
		# initialize our softmax classifier
        self.fc2 = torch.nn.Linear(in_features=2000, out_features=500)
        self.fc3 = torch.nn.Linear(in_features=500, out_features=300)
        self.fc4 = torch.nn.Linear(in_features=300, out_features=numClasses)
        self.relu4 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x)

        x = self.fc1(x)

        x = self.relu3(x)

        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        output = self.relu4(x)

        return output
