import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import tkinter as tk
import seaborn as sns
import threading

from util.throttler import Throttler


class DataStatPage:
    def __init__(self,
                 logits,
                 labels,
                 preferred_threshold,
                 x_train_0_num,
                 x_train_1_num,
                 x_train_overs_0_num,
                 x_train_overs_1_num,
                 method_name):
        self.logits = logits
        self.labels = labels
        self.preferred_threshold = preferred_threshold
        self.x_train_0_num = x_train_0_num
        self.x_train_1_num = x_train_1_num
        self.x_train_overs_0_num = x_train_overs_0_num
        self.x_train_overs_1_num = x_train_overs_1_num
        self.window_name = "Binary Classification Metrics Visualization",
        self.method_name = method_name

        # 根据阈值计算的指标列表 np.linspace(0, 1, 100)
        self.thresholds=[]
        self.accuracies=[]
        self.precisions=[]
        self.recalls=[]
        self.f1_scores=[]
        # 根据preferred_threshold计算的指标
        self.conf_matrix = None
        self.fpr = None
        self.tpr = None
        self.p_threshold_preds = None # 阈值为preferred_threshold时的预测结果
        self.accuracy=None
        self.precision=None
        self.recall=None
        self.f1_score=None
        # about the ui
        self.root = None

        # about state
        self.throttle_interval = 0.35
        self.throttler = Throttler(self.throttle_interval)

    def calc_metrics(self):
        # calculate metrics
        self.thresholds = np.linspace(0, 1, 100)
        self.thresholds = np.insert(self.thresholds,1,1e-9)
        for threshold in self.thresholds:
            preds = (self.logits >= threshold).astype(int)
            self.accuracies.append(accuracy_score(self.labels, preds))
            self.precisions.append(precision_score(self.labels, preds, zero_division=0))
            self.recalls.append(recall_score(self.labels, preds))
            self.f1_scores.append(f1_score(self.labels, preds))
        # ROC AUC 和 ROC 曲线
        self.fpr, self.tpr, _ = roc_curve(self.labels, self.logits)
        self.p_threshold_preds = (self.logits >= self.preferred_threshold).astype(int)
        self.conf_matrix = self.get_conf_matrix(self.labels, self.p_threshold_preds)
        self.accuracy, self.precision, self.recall, self.f1_score = DataStatPage.get_metrics_by_conf(self.conf_matrix)

    @staticmethod
    def get_metrics_by_conf(conf_matrix):
        tp, fn, fp, tn = conf_matrix.ravel()
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1_score

    @staticmethod
    def get_metrics_str(accuracy, precision, recall, f1_score):
        return (f'Accuracy: {accuracy:.3f}     Precision: {precision:.3f}\n'
                f'Recall: {recall:.3f}       F1 Score: {f1_score:.3f}')


    def on_threshold_change(self, threshold):
        self.throttler.set_target(self.update_on_threshold_change, args=(threshold,))

    @staticmethod
    def get_conf_matrix(labels,preds):
        tn, fp, fn, tp = confusion_matrix(labels,preds).ravel()
        return np.array(
            [[tp, fn],
             [fp, tn]]
        )

    def update_on_threshold_change(self, threshold):
        threshold = float(threshold)
        self.preferred_threshold = threshold
        self.p_threshold_preds = (self.logits >= threshold).astype(int)
        self.conf_matrix = DataStatPage.get_conf_matrix(self.labels, self.p_threshold_preds)
        self.accuracy, self.precision, self.recall, self.f1_score = DataStatPage.get_metrics_by_conf(self.conf_matrix)
        # update conf_matrix figure
        fig3=self.plot_conf_matrix()
        canvas3 = FigureCanvasTkAgg(fig3, master=self.root)
        canvas3.draw()
        canvas3.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky="nsew", padx=15, pady=0)
        # update result string
        metrics_str = DataStatPage.get_metrics_str(self.accuracy, self.precision, self.recall, self.f1_score)
        # metrics result label
        m_label = tk.Label(self.root, text=metrics_str, font=("Arial", 14), background='white',anchor='w',fg='darkblue')
        m_label.grid(row=4, column=0, columnspan=1, sticky="nsew", padx=25, pady=0)


    def plot_conf_matrix(self):
        fig = Figure(figsize=(2, 1), dpi=80)
        ax = fig.add_subplot(111)
        sns.heatmap(self.conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix at Threshold {self.preferred_threshold}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Positive", "Negative"])
        ax.set_yticklabels(["Positive", "Negative"])
        return fig

    def show(self):
        # 创建主窗口
        self.root = tk.Tk()
        self.root.configure(bg='white')
        self.root.wm_title(self.window_name)

        # calc
        self.calc_metrics()

        # Reduce figure sizes
        fig1 = Figure(figsize=(3, 2), dpi=80)
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.thresholds, self.accuracies, label="Accuracy")
        ax1.plot(self.thresholds, self.precisions, label="Precision")
        ax1.plot(self.thresholds, self.recalls, label="Recall")
        ax1.plot(self.thresholds, self.f1_scores, label="F1 Score")
        ax1.set_title("Metrics vs Threshold")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Score")
        ax1.legend()

        fig2 = Figure(figsize=(3, 2), dpi=80)
        ax2 = fig2.add_subplot(111)
        ax2.plot(self.fpr, self.tpr, label="ROC Curve")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2.set_title("ROC Curve")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()

        fig3 = self.plot_conf_matrix()

        fig4 = Figure(figsize=(3, 2), dpi=80)
        ax4 = fig4.add_subplot(111)
        train_sizes = [self.x_train_0_num, self.x_train_1_num]
        ax4.pie(train_sizes, labels=["Train 0", "Train 1"], autopct="%1.1f%%", startangle=90)
        ax4.set_title("Training Data Distribution")

        fig5 = Figure(figsize=(3, 2), dpi=80)
        ax5 = fig5.add_subplot(111)
        test_sizes = [self.x_train_overs_0_num, self.x_train_overs_1_num]
        ax5.pie(test_sizes, labels=["Train 0", "Train 1"], autopct="%1.1f%%", startangle=90)
        ax5.set_title("Training Data Distribution(OverSampling)")

        # manage the layout

        # Configure grid layout to expand dynamically
        self.root.grid_rowconfigure(0, weight=1)  # Label row
        self.root.grid_rowconfigure(1, weight=4)  # Plot rows
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=4)
        self.root.grid_rowconfigure(4, weight=2)  # Scale row
        # Configure column layout to expand dynamically
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)
        self.root.grid_columnconfigure(4, weight=1)
        self.root.grid_columnconfigure(5, weight=1)

        # Method name label
        name_label = tk.Label(self.root, text = self.method_name, font=("Arial", 18), background='white')
        name_label.grid(row=0, column=0, columnspan=6, sticky="nsew", padx=15, pady=2)

        # Add the canvases for each figure
        canvas1 = FigureCanvasTkAgg(fig1, master=self.root)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=1, column=0, columnspan=3, sticky="nsew", padx=8, pady=0)

        canvas2 = FigureCanvasTkAgg(fig2, master=self.root)
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=1, column=3, columnspan=3, sticky="nsew", padx=8, pady=0)

        # Add the scale for threshold adjustment
        scale = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient='horizontal', command=self.on_threshold_change, length=300,
                         width=6, background='white', highlightbackground='white')
        scale.set(value=self.preferred_threshold)
        scale.grid(row=2, column=0, columnspan=1, sticky="nsew", padx=25)

        canvas3 = FigureCanvasTkAgg(fig3, master=self.root)
        canvas3.draw()
        canvas3.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky="nsew", padx=15, pady=0)

        # metrics result label
        metrics_str = DataStatPage.get_metrics_str(self.accuracy, self.precision, self.recall, self.f1_score)
        m_label = tk.Label(self.root, text = metrics_str, font=("Arial", 14), background='white',anchor='w',fg='darkblue')
        m_label.grid(row=4, column=0, columnspan=1, sticky="nsew", padx=25, pady=0)

        canvas4 = FigureCanvasTkAgg(fig4, master=self.root)
        canvas4.draw()
        canvas4.get_tk_widget().grid(row=3, column=2, columnspan=2, sticky="nsew", padx=0, pady=0)

        canvas5 = FigureCanvasTkAgg(fig5, master=self.root)
        canvas5.draw()
        canvas5.get_tk_widget().grid(row=3, column=4, columnspan=2, sticky="nsew", padx=0, pady=0)

        # Main loop
        self.root.mainloop()