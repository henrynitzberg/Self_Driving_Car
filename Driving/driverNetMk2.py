import torch
import torchvision.transforms as tf

class driverNetMk2(torch.nn.Module):
    def __init__(self, numChannels, numClasses):
        super(driverNetMk2, self).__init__()
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
        self.fc1 = torch.nn.Linear(in_features=10500, out_features=1050)
        self.relu3 = torch.nn.ReLU()
		# initialize our softmax classifier
        self.fc2 = torch.nn.Linear(in_features=1050, out_features=300)
        self.fc3 = torch.nn.Linear(in_features=300, out_features=15)
        self.fc4 = torch.nn.Linear(in_features=16, out_features=numClasses)
        self.relu4 = torch.nn.ReLU()

    def forward(self, x):
        (goal, data) = x
        data = self.conv1(data)
        data = self.relu1(data)
        data = self.maxpool1(data)

        data = self.conv2(data)
        data = self.relu2(data)
        data = self.maxpool2(data)

        data = torch.flatten(data)

        data = self.fc1(data)
        data = self.relu3(data)

        data = self.fc2(data)
        data = self.fc3(data)

        # add short term goal
        torch.cat((data, torch.tensor([goal])))
        data = self.fc4(data)

        output = self.relu4(x)

        return output
