import torch
from src.vit_br import ViT, CONFIGS

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np


# Training Parameters
batch_size = 64
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

config = CONFIGS["ViT-Ti_16"]

model = ViT(config, img_size, num_classes=len(classes))
model.load_from(np.load(r"weights/Ti_16.npz"))
model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-3)

num_epochs = 3
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train() 
    for inputs, labels in tqdm(trainloader, "Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_loss /= len(trainloader)
    train_acc = 100. * train_correct / train_total

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    model.eval() 
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, "Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(testloader)
    val_acc = 100. * val_correct / val_total

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:.4f}, Val. Acc: {val_acc:.2f}%')

print('Finished Training')

