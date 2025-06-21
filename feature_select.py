import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import numpy as np
import tqdm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class AttentionModule(nn.Module):
    def __init__(self, input_size, E_node, A_node, set_seed=None):
        super(AttentionModule, self).__init__()
        self.E_W = nn.Parameter(torch.randn(input_size, E_node))
        self.E_b = nn.Parameter(torch.randn(E_node))
        self.A_W = nn.Parameter(torch.randn(input_size, E_node, A_node))
        self.A_b = nn.Parameter(torch.randn((input_size, A_node)))
        if set_seed is not None:
            torch.manual_seed(set_seed)

    def forward(self, x):
        # print(self.E_W)
        # print(x)
        E = torch.tanh(torch.matmul(x, self.E_W) + self.E_b)
        attention_out_list = []
        for i in range(x.size(1)):
            attention_FC = torch.matmul(E, self.A_W[i]) + self.A_b[i]
            attention_out = F.softmax(attention_FC, dim=1)
            attention_out_list.append(attention_out[:, 1])
        A = torch.stack(attention_out_list, dim=1)
        return x * A


class LearningModule(nn.Module):
    def __init__(self, input_size, L_node, output_size, set_seed=None):
        super(LearningModule, self).__init__()
        self.L_W1 = nn.Parameter(torch.randn(input_size, L_node))
        self.L_b1 = nn.Parameter(torch.randn(L_node))
        self.L_W2 = nn.Parameter(torch.randn(L_node, output_size))
        self.L_b2 = nn.Parameter(torch.randn(output_size))
        if set_seed is not None:
            torch.manual_seed(set_seed)

    def forward(self, g):
        L_FC = F.relu(torch.matmul(g, self.L_W1) + self.L_b1)
        O = torch.matmul(L_FC, self.L_W2) + self.L_b2
        return O


class AFSModel(nn.Module):
    def __init__(self, input_size, E_node, A_node, L_node, output_size, set_seed=None):
        super(AFSModel, self).__init__()
        self.attention_module = AttentionModule(input_size, E_node, A_node, set_seed)
        self.learning_module = LearningModule(input_size, L_node, output_size, set_seed)

    def forward(self, x):
        g = self.attention_module(x)
        O = self.learning_module(g)
        return O

    def get_weights(self):
        attention_weights = self.attention_module.E_W.detach().cpu().numpy()
        learning_weights = self.learning_module.L_W1.detach().cpu().numpy()
        return attention_weights, learning_weights


def train_model(train_X, train_Y, input_size, output_size, E_node, A_node, L_node, batch_size, learning_rate_base, regularization_rate, train_step, set_seed=None):
    model = AFSModel(input_size, E_node, A_node, L_node, output_size, set_seed)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate_base, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_base)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    model.to(device)

    # Create data loader
    train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_Y, dtype=torch.float32))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # for epoch in range(train_step):
    for epoch in tqdm.tqdm(range(train_step)):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss += regularization_rate * sum(p.norm() for p in model.parameters() if p.requires_grad)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def test_model(model, test_X, test_Y, input_size, output_size, batch_size):
    model.eval()
    test_dataset = TensorDataset(torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_Y, dtype=torch.float32))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target.long()).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Example usage
input_size = 66
output_size = 2
E_node = 50
A_node = 20
L_node = 50
batch_size = 16
learning_rate_base = 5e-2
regularization_rate = 0.1
train_step = 10000
set_seed = 42

# Assuming train_X and train_Y are numpy arrays

feature = np.load('C-特征打乱并标准化.npy')

# 打乱
num_rows = feature.shape[0]
shuffled_indices = np.random.permutation(num_rows)
shuffled_feature = feature[shuffled_indices]

test_size = 50  # 测试集的大小
num_tests = math.ceil(feature.shape[0] / test_size)  # 计算可以分割的测试集数量

# 初始化列表来存储训练和测试数据
train_X_list = []
train_Y_list = []
test_X_list = []
test_Y_list = []

highest_accuracy = 0.0
highest_attention_weights = None
highest_learning_weights = None

for i in range(num_tests):
    # 计算训练集和测试集的索引
    start_idx = i * test_size
    end_idx = start_idx + test_size
    if i == num_tests - 1:
        # 最后一次迭代，确保包含所有剩余的数据
        end_idx = shuffled_feature.shape[0]
    
    # 分割数据集
    train_X = np.vstack((shuffled_feature[:start_idx, :-1], shuffled_feature[end_idx:, :-1]))
    train_Y = np.concatenate((shuffled_feature[:start_idx, -1], shuffled_feature[end_idx:, -1]))
    test_X = shuffled_feature[start_idx:end_idx, :-1]
    test_Y = shuffled_feature[start_idx:end_idx, -1]
    
    # 将数据添加到列表中
    train_X_list.append(train_X)
    train_Y_list.append(train_Y)
    test_X_list.append(test_X)
    test_Y_list.append(test_Y)
    
    # 训练和测试模型
    model = train_model(train_X, train_Y, input_size, output_size, E_node, A_node, L_node, batch_size, learning_rate_base, regularization_rate, train_step, set_seed)
    accuracy = test_model(model, test_X, test_Y, input_size, output_size, batch_size)
    print(f'Test Accuracy for fold {i+1}: {accuracy}%')

    # 获取并打印权重
    attention_weights, learning_weights = model.get_weights()
    # print("Attention Weights:")
    # print(attention_weights.shape)
    # print("Learning Weights:")
    # print(learning_weights.shape)

    # 检查是否是最高的准确率
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        highest_attention_weights = attention_weights
        highest_learning_weights = learning_weights

# 保存最高准确率时的权重
np.save('./weight/C-attention_weights.npy', highest_attention_weights)
np.save('./weight/C-learning_weights.npy', highest_learning_weights)

print(f'Highest Test Accuracy: {highest_accuracy}%')
print('Weights saved for the highest accuracy.')