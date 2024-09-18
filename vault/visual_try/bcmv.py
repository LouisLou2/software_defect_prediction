import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# 创建主窗口
root = tk.Tk()
window_name = "Binary Classification Metrics Visualization"
method_name = 'Contrastive Learning & CNN'
root.wm_title(window_name)

# data prepared
logits = np.random.rand(100)  # 假设为模型预测结果
labels = np.random.randint(0, 2, size=100)  # 假设为真实标签
x_train_0_num, x_train_1_num = 50, 50
x_test_0_num, x_test_1_num = 40, 60

# calculate metrics
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
preferred_threshold = 0.75
# 阈值为 0.75 时的混淆矩阵
threshold_0_75_preds = (logits >= preferred_threshold).astype(int)
# 注意confusion_matrix返回矩阵的结构
tn, fp, fn, tp = confusion_matrix(labels, threshold_0_75_preds).ravel()

# begin to visualize

# begin to create figure layout
# Create a figure
fig = Figure(figsize=(10, 8), dpi=100)
fig.patch.set_facecolor('lightgrey')  # 设置整个figure的背景色为浅灰色
# Adjust padding and spacing
# fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.5)
# Create a GridSpec layout: 2 rows, 6 columns
gs = GridSpec(2, 6, figure=fig, wspace=2, hspace=0.3)
# Add two elements in the first row, each spanning 3 columns
ax1 = fig.add_subplot(gs[0, :3])  # First subplot in the top-left (span 3 columns)
ax2 = fig.add_subplot(gs[0, 3:])  # Second subplot in the top-right (span 3 columns)
# Add three elements in the second row, each spanning 2 columns
ax3 = fig.add_subplot(gs[1, :2])  # Third subplot in the bottom-left (span 2 columns)
ax4 = fig.add_subplot(gs[1, 2:4])  # Fourth subplot in the bottom-center (span 2 columns)
ax5 = fig.add_subplot(gs[1, 4:])  # Fifth subplot in the bottom-right (span 2 columns)

# plot each figure
# 折线图（随着阈值变化的指标）
ax1.plot(thresholds, accuracies, label="Accuracy")
ax1.plot(thresholds, precisions, label="Precision")
ax1.plot(thresholds, recalls, label="Recall")
ax1.plot(thresholds, f1_scores, label="F1 Score")
ax1.set_title("Metrics vs Threshold")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")
ax1.legend()
# ROC 曲线
ax2.plot(fpr, tpr, label="ROC Curve")
ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
# 热力图（阈值为 preferred_threshold 的混淆矩阵）
conf_matrix = np.array([[tp, fn], [fp, tn]])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax3)
ax3.set_title(f"Confusion Matrix at Threshold {preferred_threshold}")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.set_xticklabels(["Positive","Negative"])
ax3.set_yticklabels(["Positive","Negative"])
# 训练集饼图
train_sizes = [x_train_0_num, x_train_1_num]
ax4.pie(train_sizes, labels=["Train 0", "Train 1"], autopct="%1.1f%%", startangle=90)
ax4.set_title("Training Data Distribution")
# 测试集饼图
test_sizes = [x_test_0_num, x_test_1_num]
ax5.pie(test_sizes, labels=["Test 0", "Test 1"], autopct="%1.1f%%", startangle=90)
ax5.set_title("Testing Data Distribution")

# begin manage the layout of the main window
# create a label
label = tk.Label(root, text=method_name, font=("Arial", 20))
# grid the label
label.grid(row=0, column=0,)
# Embed the figure into the Tkinter canvas
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)
# Key press event handler
def on_key_press(event):
    print(f"you pressed {event.key}")

canvas.mpl_connect("key_press_event", on_key_press)

# Quit button
def _quit():
    root.quit()
    root.destroy()

button = tk.Button(master=root, text="Quit", command=_quit)
button.grid(row=2, column=0)
# begin mainloop
tk.mainloop()