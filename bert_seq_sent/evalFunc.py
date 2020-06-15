import numpy as np

from collections import defaultdict

from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

def num_correctly_classified(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

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

def pool(doc_id_eval_arr, topics_eval_arr, pred_arr, pool_func):
    y_probs = softmax(pred_arr)
    y_logits = y_probs[:, 1]

    dict_y = defaultdict(list)
    dict_topic = defaultdict(int)

    top_doc_arr = []
    for top, doc in zip(topics_eval_arr, doc_id_eval_arr):
        top_doc_arr.append(f"{doc}_{top}")

    for y, top_doc in zip(y_logits, top_doc_arr):
        dict_y[top_doc].append(y)
        # dict_topic[doc] = top

    print(top_doc_arr)
    print(dict_y.keys())

    # assert list(dict_y.keys()) == list(set(top_doc_arr))

    reg_pred_pooled = []
    topics_id_pooled = []
    doc_id_pooled = []

    print(f"dict_y.keys(): {dict_y.keys()}")
    print(f"len(dict_y.keys(): {len(dict_y.keys())}")

    for k in dict_y.keys():
        # doc_id_pooled.append(k)
        reg_pred_pooled.append(pool_func(dict_y[k]))
        doc_id, topic = k.split("_")
        topics_id_pooled.append(topic)
        doc_id_pooled.append(doc_id)

    return doc_id_pooled, topics_id_pooled, reg_pred_pooled
