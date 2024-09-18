from datetime import datetime
import threading
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.ma.extras import column_stack
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

# 创建主窗口
root = tk.Tk()
root.configure(bg='white')
window_name = "Binary Classification Metrics Visualization"
method_name = 'Contrastive Learning & CNN'
root.wm_title(window_name)

# data prepared
logits = np.random.rand(100)  # 假设为模型预测结果
labels = np.random.randint(0, 2, size=100)  # 假设为真实标签
x_train_0_num, x_train_1_num = 50, 50
x_train_overs_0_num, x_train_overs_1_num = 40, 60

# calculate metrics
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
threshold_0_75_preds = (logits >= preferred_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(labels, threshold_0_75_preds).ravel()

last_timestamp = 0
# Visualization update function
def update_scale(val):
    global last_timestamp
    current_timestamp = int(datetime.now().timestamp()*1000)
    if current_timestamp - last_timestamp < 500:
        return
    last_timestamp = current_timestamp
    # 更改label
    # Add the label at the top (with proper space allocation)
    label = tk.Label(root, text=f'{last_timestamp}', font=("Arial", 18), background='white')
    label.grid(row=4, column=0, columnspan=1, sticky="nsew", padx=8, pady=0)

# Reduce figure sizes
fig1 = Figure(figsize=(3, 2), dpi=80)
ax1 = fig1.add_subplot(111)
ax1.plot(thresholds, accuracies, label="Accuracy")
ax1.plot(thresholds, precisions, label="Precision")
ax1.plot(thresholds, recalls, label="Recall")
ax1.plot(thresholds, f1_scores, label="F1 Score")
ax1.set_title("Metrics vs Threshold")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")
ax1.legend()

fig2 = Figure(figsize=(3, 2), dpi=80)
ax2 = fig2.add_subplot(111)
ax2.plot(fpr, tpr, label="ROC Curve")
ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()

fig3 = Figure(figsize=(2, 1), dpi=80)
ax3 = fig3.add_subplot(111)
conf_matrix = np.array([[tp, fn], [fp, tn]])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax3)
ax3.set_title(f"Confusion Matrix at Threshold {preferred_threshold}")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.set_xticklabels(["Positive", "Negative"])
ax3.set_yticklabels(["Positive", "Negative"])

fig4 = Figure(figsize=(3, 2), dpi=80)
ax4 = fig4.add_subplot(111)
train_sizes = [x_train_0_num, x_train_1_num]
ax4.pie(train_sizes, labels=["Train 0", "Train 1"], autopct="%1.1f%%", startangle=90)
ax4.set_title("Training Data Distribution")

fig5 = Figure(figsize=(3, 2), dpi=80)
ax5 = fig5.add_subplot(111)
test_sizes = [x_train_overs_0_num, x_train_overs_1_num]
ax5.pie(test_sizes, labels=["Test 0", "Test 1"], autopct="%1.1f%%", startangle=90)
ax5.set_title("Training Data Distribution(OverSampling)")

# Configure grid layout to expand dynamically
root.grid_rowconfigure(0, weight=1)  # Label row
root.grid_rowconfigure(1, weight=4)  # Plot rows
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=4)
root.grid_rowconfigure(4, weight=2)  # Scale row
# Configure column layout to expand dynamically
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(3, weight=1)
root.grid_columnconfigure(4, weight=1)
root.grid_columnconfigure(5, weight=1)

# Add the label at the top (with proper space allocation)
label = tk.Label(root, text=method_name, font=("Arial", 18),background='white')
label.grid(row=0, column=0, columnspan=6, sticky="nsew", padx=15, pady=2)

# Add the canvases for each figure
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.draw()
canvas1.get_tk_widget().grid(row=1, column=0, columnspan=3,sticky="nsew", padx=8, pady=0)

canvas2 = FigureCanvasTkAgg(fig2, master=root)
canvas2.draw()
canvas2.get_tk_widget().grid(row=1, column=3, columnspan=3,sticky="nsew", padx=8, pady=0)

# Add the scale for threshold adjustment
scale = tk.Scale(root, from_=0, to=1, resolution=0.01,orient='horizontal', command=update_scale, length=300, width=6, background='white',highlightbackground='white')
scale.grid(row=2, column=0, columnspan=1, sticky="nsew",padx=25)

canvas3 = FigureCanvasTkAgg(fig3, master=root)
canvas3.draw()
canvas3.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky="nsew", padx=15, pady=0)

# Add the label at the top (with proper space allocation)
label = tk.Label(root, text=method_name, font=("Arial", 18),background='white')
label.grid(row=4, column=0, columnspan=1, sticky="nsew", padx=8, pady=0)

canvas4 = FigureCanvasTkAgg(fig4, master=root)
canvas4.draw()
canvas4.get_tk_widget().grid(row=3, column=2, columnspan=2, sticky="nsew", padx=0, pady=0)

canvas5 = FigureCanvasTkAgg(fig5, master=root)
canvas5.draw()
canvas5.get_tk_widget().grid(row=3, column=4, columnspan=2, sticky="nsew", padx=0, pady=0)

# Main loop
root.mainloop()