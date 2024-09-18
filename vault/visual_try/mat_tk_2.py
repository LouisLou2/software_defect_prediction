import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns

# 创建主窗口
root = tk.Tk()
root.wm_title("Binary Classification Metrics Visualization")

# 示例数据（请用实际的 logits 和 labels 替换）
logits = np.random.rand(100)  # 假设为模型预测结果
labels = np.random.randint(0, 2, size=100)  # 假设为真实标签

# 训练和测试集中的类别计数（根据实际数据修改）
x_train_0_num, x_train_1_num = 50, 50
x_test_0_num, x_test_1_num = 40, 60

# 计算随着阈值变化的指标
thresholds = np.linspace(0, 1, 100)
accuracies, precisions, recalls, f1_scores = [], [], [], []

for threshold in thresholds:
    preds = (logits >= threshold).astype(int)
    accuracies.append(accuracy_score(labels, preds))
    precisions.append(precision_score(labels, preds, zero_division=0))
    recalls.append(recall_score(labels, preds))
    f1_scores.append(f1_score(labels, preds))

# ROC AUC 和 ROC 曲线
fpr, tpr, _ = roc_curve(labels, logits)
prefered_threshold = 0.75
# 阈值为 0.75 时的混淆矩阵
threshold_0_75_preds = (logits >= prefered_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(labels, threshold_0_75_preds).ravel()

# 创建图表
fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 使用3行2列布局

# 折线图（随着阈值变化的指标）
axs[0, 0].plot(thresholds, accuracies, label="Accuracy")
axs[0, 0].plot(thresholds, precisions, label="Precision")
axs[0, 0].plot(thresholds, recalls, label="Recall")
axs[0, 0].plot(thresholds, f1_scores, label="F1 Score")
axs[0, 0].set_title("Metrics vs Threshold")
axs[0, 0].set_xlabel("Threshold")
axs[0, 0].set_ylabel("Score")
axs[0, 0].legend()

# ROC 曲线
axs[0, 1].plot(fpr, tpr, label="ROC Curve")
axs[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axs[0, 1].set_title("ROC Curve")
axs[0, 1].set_xlabel("False Positive Rate")
axs[0, 1].set_ylabel("True Positive Rate")
axs[0, 1].legend()

# 热力图（阈值为 0.75 的混淆矩阵）
conf_matrix = np.array([[tn, fp], [fn, tp]])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axs[1, 0])
axs[1, 0].set_title("Confusion Matrix at Threshold 0.75")
axs[1, 0].set_xlabel("Predicted")
axs[1, 0].set_ylabel("Actual")
axs[1, 0].set_xticklabels(["Negative", "Positive"])
axs[1, 0].set_yticklabels(["Negative", "Positive"])

# 训练集饼图
train_sizes = [x_train_0_num, x_train_1_num]
axs[1, 1].pie(train_sizes, labels=["Train 0", "Train 1"], autopct="%1.1f%%", startangle=90)
axs[1, 1].set_title("Training Data Distribution")

# 测试集饼图
test_sizes = [x_test_0_num, x_test_1_num]
axs[2, 1].pie(test_sizes, labels=["Test 0", "Test 1"], autopct="%1.1f%%", startangle=90)
axs[2, 1].set_title("Testing Data Distribution")

# 设置子图之间的间距
plt.tight_layout()

# 嵌入到 Tkinter 窗口
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# 退出按钮
def _quit():
    root.quit()
    root.destroy()

button = tk.Button(master=root, text="Quit", command=_quit)
button.pack(side=tk.BOTTOM)

tk.mainloop()