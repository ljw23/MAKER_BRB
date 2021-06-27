import torch
import torch.nn as nn
from .brb_dataset import BrbDataset
from pathlib import Path
from .data_preprocess import Attribute_builder
from .brb_model import BRB_Model
import json


class Trainer:
    def __init__(
            self,
            data_file: str = 'data/demodata.txt',
            attribute_file: Path = Path('data/attribute.json'),
            label_file=Path('data/label.json'),
            batch_size=8,
            epochs=1000,
            learning_rate=0.0001,
    ):

        self.data_file = data_file
        self.batch_size = batch_size
        self.device = torch.device('cpu')

        self.build_attribute(attribute_file)
        self.build_labels(label_file)
        self.build_dataloader()
        self.model = self.build_model()

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def build_dataloader(self):
        train_dataset = BrbDataset(self.data_file, self.attribute_builder)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=False)

    def build_attribute(self, attribute_file):
        with open(attribute_file, 'r') as f:
            attributes = json.load(f)
        self.attribute_builder = Attribute_builder(attributes)

        self.num_k = self.attribute_builder.num_k
        self.num_A = self.attribute_builder.num_A

    def build_labels(self, label_file: Path):
        with open(label_file, 'r') as f:
            labels = json.load(f)
        self.num_h = len(labels)

    def build_model(self):
        return BRB_Model(self.num_k, self.num_A, self.num_h)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        total_step = len(self.train_loader)
        for epoch in range(self.epochs):
            for i, (input_x, labels) in enumerate(self.train_loader):
                # Move tensors to the configured device
                input_x = input_x.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(input_x)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.model.post_mapping()

                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, self.epochs, i + 1, total_step,
                        loss.item()))

                if (epoch + 1) % 10 == 0:
                    print('')


# Device configuration

# MNIST dataset

# test_dataset = torchvision.datasets.MNIST(root='../../data',
#                                           train=False,
#                                           transform=transforms.ToTensor())

# Data loader

# model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # Move tensors to the configured device
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 100 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
