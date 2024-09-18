import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import TensorDataset, DataLoader

from data_stat.data_trait import print_data_distribution_binary
from eval.model_eval import eval_model_get_metrics, eval_model_by_visual
from model.contra_cnn.classifier import Classifier
from model.contra_cnn.feature_extractor import FeatureExtractor

# read test data
test_data_file = 'test_data.parquet'
test_data = pd.read_parquet(test_data_file)
X_test = test_data.drop(columns=['Defective'])
y_test = pd.DataFrame()
y_test['Defective'] = test_data['Defective']

# RandomUnderSampler undersampling
undersam_ratio = 0.08
rus = RandomUnderSampler(sampling_strategy=undersam_ratio, random_state=1234)
X_test, y_test = rus.fit_resample(X_test, y_test)

print_data_distribution_binary('Final Test Data',y_test)

# get dataloader
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
y_test_tensor = y_test_tensor.squeeze()

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# initialize the model
input_dim = X_test.shape[1]  # 根据你的数据设置输入维度
feature_dim = 30  # 特征维度，可以根据需要调整
num_classes = 2  # 二分类任务

# 创建特征提取器和分类器
feature_extractor = FeatureExtractor(input_dim, feature_dim)
input_shape = (1, 1, feature_dim)
classifier = Classifier(input_shape)

# load the model
fea_dict_file = 'feature_extractor.pth'
cla_dict_file = 'classifier.pth'
fea_dict = torch.load(fea_dict_file,weights_only=True)
cla_dict = torch.load(cla_dict_file,weights_only=True)

feature_extractor.load_state_dict(fea_dict)
classifier.load_state_dict(cla_dict)

prefered_threshold = 0.75


test_res = eval_model_get_metrics(test_loader, feature_extractor, classifier, prefered_threshold)
print("Test Result:")
print(test_res)

# 以下这些由测试时就确定好的
x_train_0_num = 14847
x_train_1_num = 453
x_train_overs_0_num = 14847
x_train_overs_1_num = 7423

eval_model_by_visual(
        test_loader,
        feature_extractor,
        classifier,
        x_train_0_num,
        x_train_1_num,
        x_train_overs_0_num,
        x_train_overs_1_num,
        'Contrastive Learning & CNN',
        threshold=0.75,)


