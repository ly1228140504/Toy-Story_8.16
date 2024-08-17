#%%
import sys
sys.path.append(r"D:\pycharm_software\PyCharm 2024.1\codes\race")
from try_xresnet.data_preprocessing import PTBXLDataLoader, create_dataloaders
from try_xresnet import xresnet1d
import os
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import matplotlib.pyplot as plt
import numpy as np


#%%
"load data and preprocess data"
default_path = r"D:\pycharm_software\PyCharm 2024.1\codes\race\python-example-2024-main"
os.chdir(default_path)
path = 'ptb-xl/'

data_loader = PTBXLDataLoader(path, sampling_rate=100)
X, Y = data_loader.load_data()

#%%
import re
data_folder = r"D:\pycharm_software\PyCharm 2024.1\codes\race\python-example-2024-main\ptb-xl/records100"
records = find_records(data_folder)
print(records)
num_records = len(records)
labels_all = []
for i in range(num_records):
    width = len(str(num_records))
    #print(f'- {i + 1:>{width}}/{num_records}: {records[i]}...')

    data_record = os.path.join(data_folder, records[i])
    data_header = load_header(data_record)
    comments = [l for l in data_header.split('\n') if l.startswith('#')]
    #print(comments)

    # 初始化标签变量
    labels = None

    # 遍历列表，查找包含标签的行
    for line in comments:
        label_match = re.match(r'# Labels: (.+)', line)
        if label_match:
            labels = label_match.group(1).split(', ')
            break

    if labels:
        #print("Labels:", labels)
        labels_all.append(labels)
    else:
        print("Labels not found")
print(len(labels_all))

#Y.iloc[:, -1] = labels_all
#%%
print(Y)
#%%
X_train, y_train, X_val, y_val, X_test, y_test = data_loader.split_data(X, Y)
#%%
print(y_test)
#%%
x_train = data_loader.standardize_signal_data(X_train)
x_val = data_loader.standardize_signal_data(X_val)
x_test = data_loader.standardize_signal_data(X_test)
y_train_one_hot_labels, y_val_one_hot_labels, y_test_one_hot_labels, classes = data_loader.encode_labels(y_train, y_val, y_test)
#%%
train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train_one_hot_labels, X_val, y_val_one_hot_labels, X_test, y_test_one_hot_labels,batch_size=32,w_size=2.5)

print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train_one_hot_labels.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val_one_hot_labels.shape}")
print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test_one_hot_labels.shape}")

#%%
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from sklearn.metrics import f1_score,classification_report,roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
def multi_label_accuracy(output, label, threshold=0.5):
    # Apply sigmoid function to the output to get probabilities
    output = torch.sigmoid(output)

    # Apply threshold to get binary predictions
    predictions = (output > threshold).float()

    # Calculate accuracy
    correct = (predictions == label).float().sum(dim=1)  # Sum along classes axis
    accuracy = correct / label.size(1)  # Normalize by the number of classes

    predictions = predictions.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    f1 = f1_score(label, predictions, average='macro',zero_division=1)

    return accuracy.mean(), f1


def test_multi_label_accuracy(output, label, threshold=0.5):
    # Apply sigmoid function to the output to get probabilities
    output = torch.sigmoid(output)
    # 将每个子列表堆叠为一个张量
    stacked_outputs = [torch.stack(sub_list) for sub_list in output]

    # 将所有堆叠的张量相加
    output = sum(stacked_outputs)

    # Apply threshold to get binary predictions
    predictions = (output > threshold * len(output)).float()

    # Calculate accuracy
    correct = (predictions == label).float().sum(dim=1)  # Sum along classes axis
    accuracy = correct / label.size(1)  # Normalize by the number of classes

    predictions = predictions.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    f1 = f1_score(label, predictions, average='macro',zero_division=1)

    return accuracy.mean(), f1


def train(train_loader, val_loader, model, criterion, optimizer, num_epochs):
    best_val_f1 = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1, val_f1 =[], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        cosine_scheduler.step(epoch)
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_f1 = 0

        # Train the model
        model.train()
        train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train_one_hot_labels, X_val,
                                                                   y_val_one_hot_labels, X_test, y_test_one_hot_labels,
                                                                   batch_size, w_size=5)
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            acc,f1 = multi_label_accuracy(output, label, threshold=0.5)
            epoch_accuracy += acc.item() / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)
            epoch_f1 += f1.item() / len(train_loader)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_f1.append(epoch_f1)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_accuracy = 0
            epoch_val_f1 = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc,f1 = multi_label_accuracy(val_output, label, threshold=0.5)
                epoch_val_accuracy += acc.item() / len(val_loader)
                epoch_val_loss += val_loss.item() / len(val_loader)
                epoch_val_f1 += f1.item() / len(val_loader)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)
            val_f1.append(epoch_val_f1)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], - -lr:{optimizer.param_groups[0]['lr']:.6f}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Train F1: {epoch_f1:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}, val F1: {epoch_val_f1:.4f}"
        )

        # Save the best model
        if epoch_val_f1 > best_val_f1:
            print(f"Validation accuracy improved from {best_val_f1:.4f} to {epoch_val_f1:.4f}. Saving model...")
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), 'best_model.pth')

    # Plotting the training and validation curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_f1, label='Train f1')
    plt.plot(val_f1, label='Validation f1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.show()


def test(test_loader, model):
    model.eval()
    total_acc = 0
    total_f1 = 0
    data_all = []
    with torch.no_grad():
        for data, label in test_loader:
            for data_16 in data:
                data_16 = data_16.to(device)
                label = label.to(device)
                output = model(data_16)
                data_all.append(data_16)
            acc, f1 = multi_label_accuracy(output, label, threshold=0.5)
            total_acc += acc.item() / len(test_loader)
            total_f1 += f1.item() / len(test_loader)
    print(f'Accuracy of the network on the test images: {total_acc * 100:.2f}%')
    print(f'F1 score of the network on the test images: {total_f1 * 100:.2f}%')




#%%
activation = nn.LeakyReLU(inplace=True)
#w_size= 5比2.5更好
w_size =5 # Random Segmentation Size (Unit: seconds)
model_sr = 100 #Sampling Rate for Model Input
stem_k, block_k = 7, 5 # Kernel Size for Stem and ResBlock Conv1d
data_dim = 12 # Dimension for Model Input
batch_size = 256
model_dropout = None # xResNet Dropout Rate
fc_drop = 0.5 # FC Layer Dropout Rate
out_activation = nn.Sigmoid()
loss_fn = torch.nn.BCEWithLogitsLoss()
init_lr = 1e-3
final_lr = 1e-5
num_epochs = 50
train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train_one_hot_labels, X_val, y_val_one_hot_labels, X_test, y_test_one_hot_labels,batch_size,w_size = w_size)


model = xresnet1d.xresnet1d101(model_drop_r=model_dropout,
                     original_f_number=False,
                     # Original xResnet's Resblock Filter Dim if True
                     # PTB-XL Benchmark Article's Filter Dim if False
                     fc_drop=fc_drop,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 计算每个类别的正样本权重
positive_counts = torch.tensor(y_train_one_hot_labels).sum(dim=0)
negative_counts = torch.tensor(y_train_one_hot_labels).size(0) - positive_counts
pos_weight = negative_counts / (positive_counts + 1e-5)
pos_weight = pos_weight.to(device)


# 定义损失函数
criterion =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-2)
# 定义学习率调度器（余弦退火）
cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=5, eta_min=final_lr)
# 训练模型
train(train_loader,val_loader, model, criterion, optimizer, num_epochs)


#%%
test(test_loader, model)

#%%

#%%
from torchsummary import summary
# 打印模型摘要
summary(model, input_size=(12, 1000))

#%%
for data, label in tqdm(test_loader):
    print(len(data))

#%%
print(model)

#%%
print(pos_weight)

#%%
print(torch.tensor(y_train_one_hot_labels).sum(dim=0))

#%%
print(len(test_loader))