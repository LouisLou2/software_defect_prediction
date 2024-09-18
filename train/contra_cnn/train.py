import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
import preprocess.prep as prep
from sklearn.preprocessing import StandardScaler

from eval.model_eval import eval_model_get_metrics
from model.contra_cnn.classifier import Classifier
from model.contra_cnn.feature_extractor import FeatureExtractor
from model.contra_cnn.suploss import SupConLoss

# dataset_filename = '../dataset/PC5.parquet'
# data = pd.read_parquet(dataset_filename)
# dataset_files=['MW1.parquet','PC1.parquet','PC3.parquet','PC4.parquet']
base_dir = '../../dataset/nasa_mdp/original'
dataset_files=['PC5.parquet']

data = prep.combine_multi_file_to_df(dataset_files,base_dir)
X_train, X_test, y_train, y_test = prep.split_data_with_oversampling(data)

# 创建标准化对象
scaler = StandardScaler()
# 对 x_train 进行拟合和转换
X = scaler.fit_transform(X_train)

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


input_dim = X_train.shape[1]  # 根据你的数据设置输入维度
feature_dim = 30  # 特征维度，可以根据需要调整
num_classes = 2  # 二分类任务

# 创建特征提取器和分类器
feature_extractor = FeatureExtractor(input_dim, feature_dim)
input_shape = (1, 1, feature_dim)
classifier = Classifier(input_shape)

# 定义损失函数
criterion_contrastive = SupConLoss()
# criterion_classifier = nn.BCELoss()
# 设置 pos_weight 增加正类的权重，通常是一个标量，官方建议是负例数/正例数
pos_weight = torch.tensor([3.0])
criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 定义优化器
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=7e-3)


# 训练函数
def train_model(train_loader, feature_extractor, classifier, criterion_contrastive, criterion_classifier, optimizer,
                num_epochs=100):
    feature_extractor.train()
    classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count=0
        for batch_x, batch_y in train_loader:
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
            # print(f'Epoch: {epoch} Batch:{batch_count} loss_contrastive: {loss_contrastive} loss_classifier: {loss_classifier}')
            batch_count+=1
            # 检查total_loss是否为nan
            # if torch.isnan(loss).any():
            #     print(f'Epoch: {epoch} Batch:{batch_count} loss_contrastive: {loss_contrastive} loss_classifier: {loss_classifier}')
            #     break
        print(f"@@@@@@@@Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 调用训练
train_model(train_loader, feature_extractor, classifier, criterion_contrastive, criterion_classifier, optimizer)

# 在测试集上评估模型
test_res = eval_model_get_metrics(test_loader, feature_extractor, classifier)

print("Test Result:")
print(test_res)

# 保存模型
torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')
torch.save(classifier.state_dict(), 'classifier.pth')