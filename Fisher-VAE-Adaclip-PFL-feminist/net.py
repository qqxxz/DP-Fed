from VAE_utils.cv.resnet_v2 import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(args, num_classes, model_input_channels):
    if args.model.lower() == 'resnet10_v2':
        return ResNet10(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
    elif args.model.lower() == 'resnet18_v2':
        return ResNet18(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
    elif args.model.lower() == 'resnet34_v2':
        return ResNet34(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
    elif args.model.lower() == 'resnet50_v2':
        return ResNet50(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
    elif args.model.lower() == 'resnet101_v2':
        return ResNet101(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
    elif args.model.lower() == 'resnet152_v2':
        return ResNet152(args=args, num_classes=num_classes, model_input_channels=model_input_channels)
    else:
        raise ValueError(f"Unsupported model name: {args.model}")

    
class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    




class cifar10Net(nn.Module):
    def __init__(self):
        super(cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(32*4*4, 32*4*4)
        self.fc2 = nn.Linear(32*4*4, 32*2*2)
        self.fc3 = nn.Linear(32*2*2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class femnistNet(nn.Module):
    def __init__(self):
        super(femnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    
class SVHNNet(nn.Module):
    def __init__(self):
        super(SVHNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # SVHN has 3 color channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128) # Adjusted linear layer size
        self.fc2 = nn.Linear(128, 10)  # 10 classes for the digits 0-9

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



