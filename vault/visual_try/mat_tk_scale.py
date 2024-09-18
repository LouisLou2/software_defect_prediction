import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns

# Sample data for logits (probabilities) and labels
logits = np.random.rand(100)  # Random predicted probabilities between 0 and 1
labels = np.random.randint(0, 2, 100)  # Random binary labels (0 or 1)

# Class distributions in training and test sets (these values should come from your actual data)
x_train_0_num = 200
x_train_1_num = 300
x_test_0_num = 100
x_test_1_num = 150

# Tkinter setup
root = tk.Tk()
root.title("Classification Visualization")

# Create the figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(14, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)


# Function to update plots based on threshold
def update_plots(threshold):
    threshold = float(threshold)
    preds = (logits >= threshold).astype(int)

    # Accuracy, Precision, Recall, F1 Score vs. Threshold
    thresholds = np.linspace(0, 1, 100)
    acc, prec, rec, f1 = [], [], [], []
    for t in thresholds:
        preds_t = (logits >= t).astype(int)
        acc.append(accuracy_score(labels, preds_t))
        prec.append(precision_score(labels, preds_t, zero_division=1))
        rec.append(recall_score(labels, preds_t))
        f1.append(f1_score(labels, preds_t))

    axs[0, 0].cla()
    axs[0, 0].plot(thresholds, acc, label="Accuracy")
    axs[0, 0].plot(thresholds, prec, label="Precision")
    axs[0, 0].plot(thresholds, rec, label="Recall")
    axs[0, 0].plot(thresholds, f1, label="F1 Score")
    axs[0, 0].set_title('Metrics vs. Threshold')
    axs[0, 0].legend()

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, logits)
    axs[0, 1].cla()
    axs[0, 1].plot(fpr, tpr, label="ROC Curve")
    axs[0, 1].set_title('ROC Curve')
    axs[0, 1].set_xlabel('False Positive Rate')
    axs[0, 1].set_ylabel('True Positive Rate')

    # Confusion Matrix at selected threshold
    conf_matrix = confusion_matrix(labels, preds)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axs[1, 0])
    axs[1, 0].set_title(f'Confusion Matrix (Threshold={threshold:.2f})')
    axs[1, 0].set_xlabel('Predicted Label')
    axs[1, 0].set_ylabel('True Label')

    # Pie chart for training data distribution
    axs[1, 1].cla()
    axs[1, 1].pie([x_train_0_num, x_train_1_num], labels=["Class 0", "Class 1"], autopct='%1.1f%%', startangle=90)
    axs[1, 1].set_title('Training Data Distribution')

    # Pie chart for test data distribution
    axs[1, 2].cla()
    axs[1, 2].pie([x_test_0_num, x_test_1_num], labels=["Class 0", "Class 1"], autopct='%1.1f%%', startangle=90)
    axs[1, 2].set_title('Test Data Distribution')

    fig.canvas.draw_idle()


# Initialize the plots with threshold 0.75
update_plots(0.75)

# Add the figure to the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# Create a slider for the threshold
threshold_slider = ttk.Scale(root, from_=0, to=1, orient='horizontal', command=update_plots)
threshold_slider.set(0.75)  # Set initial value to 0.75
threshold_slider.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# Run the Tkinter main loop
root.mainloop()
