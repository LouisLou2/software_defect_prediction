from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, roc_auc_score, recall_score, \
    f1_score

def print_scores(clf_name, y_val, y_val_pred, y_test, y_test_pred):
    print(f'Score for {clf_name}')
    print("-- Validation Set --")
    print("Accuracy: ", accuracy_score(y_val, y_val_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_val, y_val_pred))
    print("Precision: ", precision_score(y_val, y_val_pred))
    # print("AUC: ", roc_auc_score(y_val, y_val_pred))
    print("Recall: ", recall_score(y_val, y_val_pred))
    # f1_score
    print("F1: ", f1_score(y_test, y_test_pred))

    print("-- Test Score --")
    print("Accuracy: ", accuracy_score(y_test, y_test_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_test_pred))
    print("Precision: ", precision_score(y_test, y_test_pred))
    # print("AUC: ", roc_auc_score(y_test, y_test_pred))
    print("Recall: ", recall_score(y_test, y_test_pred))
    # f1_score
    print("F1: ", f1_score(y_test, y_test_pred))