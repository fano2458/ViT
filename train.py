import torch
from src.vit_br import ViT, CONFIGS
from src.utils import AverageMeter, WarmupCosineSchedule, simple_accuracy

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np
import random
import time

from torchinfo import summary

# Training Parameters
batch_size = 64
img_size = 224
learning_rate = 3e-2
total_steps = 10000
warmup_steps = 500
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# num_epochs = 3
# for epoch in range(num_epochs):
#     train_loss = 0.0
#     train_correct = 0
#     train_total = 0

#     model.train() 
#     for inputs, labels in tqdm(trainloader, "Training"):
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs, _ = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         train_total += labels.size(0)
#         train_correct += predicted.eq(labels).sum().item()

#     train_loss /= len(trainloader)
#     train_acc = 100. * train_correct / train_total

#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0

#     model.eval() 
#     with torch.no_grad():
#         for inputs, labels in tqdm(testloader, "Testing"):
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             outputs, _ = model(inputs)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()
#             _, predicted = outputs.max(1)
#             val_total += labels.size(0)
#             val_correct += predicted.eq(labels).sum().item()

#     val_loss /= len(testloader)
#     val_acc = 100. * val_correct / val_total

#     print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:.4f}, Val. Acc: {val_acc:.2f}%')
#     print(f"Cached GPU Memory: {torch.cuda.max_memory_allocated('cuda') / 1024**2:.3f} MB") #TODO change

# print('Finished Training')

def valid(model, testloader, global_step):
    eval_losses = AverageMeter()
    
    end = time.time()
    model.eval()
    
    all_preds, all_label = [], []
    criterion = torch.nn.CrossEntropyLoss()
    
    for step, batch in tqdm(enumerate(testloader)): # "Testing", total=len(testloader)
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            outputs, _ = model(x)
            
            eval_loss = criterion(outputs, y)
            eval_losses.update(eval_loss.item())
            
            preds = torch.argmax(outputs, dim=-1)
        
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            
    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    
    print("\n")
    print("Validation Results")
    print("Global Steps: {}".format(global_step))
    print("Valid Loss: {}".format(eval_losses.avg))
    print("Valid Accuracy: {}".format(accuracy))
    print("Time spent: {:.2f}".format(time.time() - end))
    
    return accuracy


def train(model, trainloader, testloader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    memory_meter = AverageMeter()
    
    params_list = model.parameters()
    
    optimizer = torch.optim.SGD(params_list,
                                lr=learning_rate,
                                momentum=0.9)
    
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    
    print("Starting training")
    
    losses = AverageMeter()
    global_step = 0
    epoch = 0
    accuracy = -1
    
    while True:
        epoch += 1
        model.train()
        end = time.time()
        
        for step, batch in tqdm(enumerate(trainloader)): #  "Training", total=len(trainloader)
            data_time.update(time.time() - end)
            batch = tuple(t.to(device) for t in batch)
            
            x, y = batch
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            
            loss.backward()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            MB = 1024 * 1024
            memory_meter.update(torch.cuda.max_memory_allocated() / MB)
            losses.update(loss.item())
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            
            if global_step % 300 == 0:
                print("Training ({}/{} Steps)\t(loss={:2.5f})\tData time={:.2f}({:.2f})\tBatch time={:.2f}({:.2f})\tMemory={:.1f}({:.1f})".format(
                        global_step, total_steps, losses.val, data_time.val, data_time.avg, batch_time.val, batch_time.avg, memory_meter.val, memory_meter.avg))
            if global_step % 500 == 0:
                accuracy = valid(model, testloader, global_step)
                model.train()
            if global_step % total_steps == 0:
                break
            
        losses.reset()
        if global_step % total_steps == 0:
            break
        
    print("Final accuracy: \t{}".format(accuracy))
    print("End training")
    

def main():
    
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

    model = ViT(config, img_size, num_classes=len(classes), visualize=False)
    model.load_from(np.load(r"weights/Ti_16.npz"))
    model.to(device)
    
    
    summary(model=model,
        input_size=(64, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )

    train(model, trainloader, testloader)
    

if __name__ == "__main__":
    main()
    