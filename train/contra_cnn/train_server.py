'''
Be Ware!
this script is bound to be run on the server(the pytorch for CUDA),
so all the paths and datasets below are related to the server's file system.
and the env to run this script maybe different from the local env.
'''

import torch
import os
import json
import shutil
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from datetime import datetime
from sklearn.preprocessing import StandardScaler

record_dir = f'/root/record/contra_{datetime.now()}'

if not os.path.exists(record_dir):
    os.makedirs(record_dir)
print(f'wiil be at the dir:{record_dir}')


# 定义特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 36)  # 简单的全连接层提取特征
        self.fc2 = nn.Linear(36, 32)
        self.fc3 = nn.Linear(32, feature_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


# # 定义分类器
# class Classifier(nn.Module):
#     def __init__(self, feature_dim, num_classes):
#         super(Classifier, self).__init__()
#         self.fc = nn.Linear(feature_dim, num_classes)  # 简单的线性分类器
#
#     def forward(self, x):
#         return self.fc(x)
# 定义CNN分类器
class Classifier(nn.Module):
    def __init__(self, input_shape):
        super(Classifier, self).__init__()
        self.input_shape = input_shape
        last_channel = 16  # 最后一个卷积层的输出通道数
        # Keras的Conv2D(kernel_size=1)相当于PyTorch的nn.Conv2d(kernel_size=(1,1))
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=last_channel, kernel_size=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(last_channel * input_shape[1] * input_shape[2], 8)  # 输入形状需要调整为16 * H * W
        self.fc2 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 将x转换为4D
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x


# 定义 SupConLoss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, label):
        device = features.device
        n = features.shape[0]  # batch
        T = self.temperature  # 温度参数T
        # label=label.squeeze() no need
        # 这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        # 这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

        # 这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask

        # 这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = (torch.ones(n, n) - torch.eye(n, n)).to(device)

        # 这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = (torch.exp(similarity_matrix / T)).to(device)

        # 这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix * mask_dui_jiao_0

        # 这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask * similarity_matrix

        # 用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim

        # 把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim, dim=1)

        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend
        loss = torch.div(sim, sim_sum)

        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        loss = mask_no_sim.to(device) + loss.to(device) + torch.eye(n, n).to(device)

        # 接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  # 求-log
        tmp = torch.nansum(loss, dim=1)
        # 判断是否是nan
        if torch.isnan(tmp).sum() > 0:
            print('nan exists')
        loss = torch.nansum(tmp) / (2 * n)  # 将所有数据都加起来除以2n
        # 判断是否是nan
        if torch.isnan(loss).sum() > 0:
            print('nan exists')
        return loss


def combine_multi_file_to_df(filenames, basedir=''):
    assert len(filenames) > 0, "No file to combine"
    df_list = []
    data = pd.read_parquet(basedir + '/' + filenames[0])
    df_list.append(data)
    for i in range(1, len(filenames)):
        data = pd.read_parquet(basedir + '/' + filenames[i])
        df_list.append(data)
    data = pd.concat(df_list, ignore_index=True)
    return data


def print_data_distribution_binary(name, y):
    # 取第一列
    assert y.shape[1] == 1
    y_data = y.iloc[:, 0]
    print(name)
    total = y.shape[0]
    defective = sum(y_data)
    print("Total: ", total)
    print("Defective: ", defective)
    print("Non-defective: ", total - defective)
    print("Defective Ratio: ", defective / total)
    print()


def split_data_with_oversampling(data, test_size=0.1):
    X = pd.DataFrame(data.drop(['Defective'], axis=1))
    y = pd.DataFrame(data['Defective'])
    # 创建标准化对象
    scaler = StandardScaler()
    # 分层随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=1234
    )
    # 对 x_train 进行拟合和转换
    # X_train = scaler.fit_transform(X_train)
    print_data_distribution_binary("Train", y_train)
    print_data_distribution_binary("Test", y_test)

    sm = SMOTE(random_state=1234, k_neighbors=5, sampling_strategy=0.7)  # for oversampling minority data
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print_data_distribution_binary("Train after oversampling", y_train)

    return X_train, X_test, y_train, y_test


# dataset_filename = '../dataset/PC5.parquet'
# data = pd.read_parquet(dataset_filename)

# dataset_files =  ['KC3.parquet','MC1.parquet','MC2.parquet', 'MW1.parquet', 'PC1.parquet', 'PC3.parquet', 'PC4.parquet', 'PC5.parquet']
# dataset_files=['MW1.parquet,PC4.parquet']
# data = combine_multi_file_to_df(dataset_files,'./dataset/clean_nasa_d2_inte')
data = pd.read_parquet('./dataset/PC5.parquet')
X_train, X_test, y_train, y_test = split_data_with_oversampling(data)

# 将 DataFrame 转换为张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_train_tensor = y_train_tensor.squeeze()

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
y_test_tensor = y_test_tensor.squeeze()

# 创建训练、测试和验证的 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 确定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = X_train.shape[1]  # 根据你的数据设置输入维度
feature_dim = 30  # 特征维度，可以根据需要调整
num_classes = 2  # 二分类任务

# 创建特征提取器和分类器，并将其移动到设备上
feature_extractor = FeatureExtractor(input_dim, feature_dim).to(device)
input_shape = (1, 1, feature_dim)
classifier = Classifier(input_shape).to(device)

# 定义损失函数
criterion_contrastive = SupConLoss().to(device)
# 设置 pos_weight 增加正类的权重，通常是一个标量，官方建议是负例数/正例数
pos_weight = torch.tensor([3.0])
criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

# 定义优化器
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=1e-3)


# 训练函数
def train_model(train_loader, feature_extractor, classifier, criterion_contrastive, criterion_classifier, optimizer,
                num_epochs=200):
    feature_extractor.train()
    classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in train_loader:
            # 将数据移动到设备上
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 前向传播：特征提取
            features = feature_extractor(batch_x)

            # 对比学习损失
            loss_contrastive = criterion_contrastive(features, batch_y)

            # 分类器前向传播
            logits = classifier(features)
            logits = logits.squeeze()
            loss_classifier = criterion_classifier(logits, batch_y)

            # 总损失
            loss = loss_contrastive * 0.4 + loss_classifier * 0.6

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        print(f"@@@@@@@@Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 调用训练
train_model(train_loader, feature_extractor, classifier, criterion_contrastive, criterion_classifier, optimizer)


# 模型评估函数
def evaluate_model(data_loader, feature_extractor, classifier):
    feature_extractor.eval()
    classifier.eval()

    all_labels = []
    all_predictions = []

    threshold = 0.7

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            # 将数据移动到设备上
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            features = feature_extractor(batch_x)
            logits = classifier(features)
            predicted = torch.where(logits >= threshold, torch.tensor(1), torch.tensor(0))

            all_labels.extend(batch_y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }


# 在测试集上评估模型
test_res = evaluate_model(test_loader, feature_extractor, classifier)

print("Test Result:")
print(test_res)

# 保存结果
with open(f'{record_dir}/record.txt', 'w') as file:
    file.write(json.dumps(test_res, indent=4))
# 保存模型
feature_extractor = feature_extractor.cpu()
classifier = classifier.cpu()
fe_path = f'{record_dir}/feature_extractor.pth'
cl_path = f'{record_dir}/classifier.pth'
torch.save(feature_extractor.state_dict(), fe_path)
torch.save(classifier.state_dict(), cl_path)
# 复制py
source_file_path = '/root/contra.py'
destination_file_path = f'{record_dir}/contra.py'
shutil.copy(source_file_path, destination_file_path)
# 保存测试数据
# 拼接x_test和y_test
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_parquet(f'{record_dir}/test_data.parquet')
