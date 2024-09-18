import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from visual.stat_page import DataStatPage


def get_model_res(data_loader, feature_extractor, classifier, threshold=0.7):
    feature_extractor.eval()
    classifier.eval()

    all_labels = []
    all_logits = []
    all_predictions = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            features = feature_extractor(batch_x)
            logits = classifier(features).squeeze()
            predicted = torch.where(logits >= threshold, torch.tensor(1), torch.tensor(0))
            all_labels.extend(batch_y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    return all_labels, all_logits, all_predictions

def eval_model_get_metrics(data_loader, feature_extractor, classifier, threshold=0.7):
    feature_extractor.eval()
    classifier.eval()

    all_labels, all_logits, all_predictions = get_model_res(data_loader, feature_extractor, classifier, threshold)
    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc = roc_auc_score(all_labels, all_logits)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

def eval_model_by_visual(
        data_loader,
        feature_extractor,
        classifier,
        x_train_0_num,
        x_train_1_num,
        x_train_overs_0_num,
        x_train_overs_1_num,
        method_name,
        threshold=0.7,):
    feature_extractor.eval()
    classifier.eval()
    all_labels, all_logits, all_predictions = get_model_res(data_loader, feature_extractor, classifier, threshold)
    logits=np.array(all_logits)
    labels=np.array(all_labels)
    DataStatPage(logits, labels,
                 threshold,
                 x_train_0_num,
                 x_train_1_num,
                 x_train_overs_0_num,
                 x_train_overs_1_num,
                 method_name).show()