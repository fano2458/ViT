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


def train(model, trainloader, testloader, config):
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
    
    l1_lambda = 0.0001
    
    while False:
        epoch += 1
        model.train()
        end = time.time()
        
        for step, batch in tqdm(enumerate(trainloader)): #  "Training", total=len(trainloader)
            data_time.update(time.time() - end)
            batch = tuple(t.to(device) for t in batch)
            
            #print(model.transformer.encoder.layer[0].attn.channel_importance.importance) # 0-11
            #print(model.transformer.encoder.layer[0].ffn.fc1.weight)
            #print(model.transformer.encoder.layer[0].ffn.mlp_importance1) # 0-11
            #print(model.transformer.encoder.layer[0].ffn.mlp_importance2) # 0-11
            
            x, y = batch
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            
            for i in range(0, 12):
                l1_norm_attn = torch.norm(model.transformer.encoder.layer[i].attn.channel_importance.importance, p=1)
                l1_norm_mlp1 = torch.norm(model.transformer.encoder.layer[i].ffn.mlp_importance1.importance, p=1)
                l1_norm_mlp2 = torch.norm(model.transformer.encoder.layer[i].ffn.mlp_importance2.importance, p=1)
                
                loss = loss + l1_lambda * (l1_norm_attn + l1_norm_mlp1 + l1_norm_mlp2)
                
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
    
    #torch.save(model.state_dict(), "weights.pth")
    
    # importance_scores = []
    # all_scores = []
    # for i in range(0, 12):
        # importance_scores.append(list(model.transformer.encoder.layer[i].attn.channel_importance.importance.detach(),
        #                               model.transformer.encoder.layer[i].ffn.mlp_importance1.importance.detach(),
        #                               model.transformer.encoder.layer[i].ffn.mlp_importance2.importance.detach()))
        # all_scores.extend(model.transformer.encoder.layer[i].attn.channel_importance.importance.detach())
        # all_scores.extend(model.transformer.encoder.layer[i].ffn.mlp_importance1.importance.detach())
        # all_scores.extend(model.transformer.encoder.layer[i].ffn.mlp_importance2.importance.detach())
        
    # sorted_all_scores = sorted(all_scores)
    
    # pruning_ratio = 0.2
    
    # threshold_index = int(pruning_ratio * len(sorted_all_scores))
    # threshold = sorted_all_scores[threshold_index] 
    
    # print(f"Testing of model with masked channels")
    # print(f"Pruning with pruning ratio of {pruning_ratio}")
    # print(f"Threshold value is {threshold}")
    
    model = ViT(config, img_size, num_classes=10, visualize=False, prune=True, ratio=0.05)
    
    model.load_state_dict(torch.load("weights.pth"))
    model.to(device)
    
    accuracy = valid(model, testloader, "Pruned model")
    
    print(f"Accuracy of pruned model is {accuracy}")
    
    
    # print(model.model.transformer.encoder.layer[0].attn.channel_importance.importance.detach())
    
    # for i in range(12):
    #     print(f"Layer {i} mask for D size")
    #     print(model.transformer.encoder.layer[0].attn.channel_importance.importance) # 0-11
    

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

    train(model, trainloader, testloader, config)
    

if __name__ == "__main__":
    main()
    