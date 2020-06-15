import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

def num_correctly_classified(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

def eval_pr_per_topics(preds, labels, topics_seq, pr_dict):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    if len(pred_flat) != len(labels_flat):
        raise ValueError("The arrays have different len!")

    size = len(pred_flat)

    for i in range(size):
        topic = topics_seq[i]
        if (pred_flat[i] == 0) and (labels_flat[i] == 0):
            pr_dict[topic]["true_negative"] += 1

        elif (pred_flat[i] == 0) and (labels_flat[i] == 1):
            pr_dict[topic]["false_negative"] += 1

        elif (pred_flat[i] == 1) and (labels_flat[i] == 0):
            pr_dict[topic]["false_positive"] += 1

        else:
            pr_dict[topic]["true_positive"] += 1

def softmax(y_pred):
    y_max = np.max(y_pred, axis=1, keepdims=True)

    y_norm = y_pred - y_max
    return np.exp(y_norm)/np.sum(np.exp(y_norm), axis=1, keepdims=True)

def calc_roc(y_pred, y_truth, num_classes, pos_label=1):
    assert num_classes > 1

    y_probs = softmax(y_pred)

    # Extract the probabilities of class = 1, which is what
    # ROC is expecting.
    y_logits = y_probs[:, 1]

    if num_classes == 2:
        fpr, tpr, threshold = roc_curve(y_truth, y_logits, pos_label)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(
            fpr, tpr,
            color='darkorange', lw=2,
            label='ROC curve (area = %0.2f)' % roc_auc
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC, binary classes')
        plt.legend(loc="lower right")
        plt.show()

def calc_avg_precision_score(y_logits, y_truth, num_classes):
    assert num_classes > 1

    if num_classes == 2:
        return average_precision_score(y_truth, y_logits)

def calc_pr_curve(y_pred, y_truth, num_classes):
    assert num_classes > 1

    y_probs = softmax(y_pred)

    # Extract the probabilities of class = 1, which is what
    # ROC is expecting.
    y_logits = y_probs[:, 1]

    if num_classes == 2:
        avg_prec = calc_avg_precision_score(y_logits, y_truth)

        prec, recall, thresholds = precision_recall_curve(y_truth, y_logits)

        plt.figure()
        plt.plot(
            prec, recall,
            color='darkorange', lw=2,
            label='Average Precision (area = %0.2f)' % avg_prec
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend(loc="lower right")
        plt.show()

        # disp = plot_precision_recall_curve(classifier, X_test, y_test)
        # disp.ax_.set_title('2-class Precision-Recall curve: '
        #                   'AP={0:0.2f}'.format(average_precision))