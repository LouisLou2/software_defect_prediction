import numpy as np

from visual.stat_page import DataStatPage

# data prepared
logits = np.random.rand(100)  # 假设为模型预测结果
labels = np.random.randint(0, 2, size=100)  # 假设为真实标签
x_train_0_num, x_train_1_num = 50, 50
x_train_overs_0_num, x_train_overs_1_num = 40, 60

DataStatPage(logits,labels,
                 0.75,
                 x_train_0_num,
                 x_train_1_num,
                 x_train_overs_0_num,
                 x_train_overs_1_num,
                 'Contrastive Learning & CNN').show()