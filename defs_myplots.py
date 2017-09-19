from sklearn.calibration import calibration_curve
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns


def auc_plot(y_true, y_pred, size=False):    
    if size:
        fig = plt.figure(1, figsize=(size[0], size[1]))
        
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
def cal_plot(y_true, y_pred, nbins, y_fixed=False, size=False):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, False, n_bins=nbins)
    
    if size:
        fig = plt.figure(1, figsize=(size[0], size[1]))   
        
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)         
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="un-calibrated", linewidth=1, color='blue')
    
    if type(y_fixed) ==  np.ndarray:
        frac, mea = calibration_curve(y_true, y_fixed, False, n_bins=nbins)
        ax1.plot(mea, frac, "s-", label='calibrated', color = 'red', linewidth=2)        

    ax2.hist(y_pred, range=(0, 1), bins=nbins, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    
def ap_plot(y_true, y_pred, size=False):
    if size:
        fig = plt.figure(1, figsize=(size[0], size[1]))
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve{}'.format(average_precision))
    
def conf_matrix(y_test, probs, th=.5, size=False):
    preds = [1 if x>th else 0 for x in probs]
    conf = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # Set up the matplotlib figure
    if size:
        f, ax = plt.subplots(figsize=(size[0], size[1]))
    labels =  np.array([['TN: {}'.format(tn),'FP: {}'.format(fp)],
                        ['FN: {}'.format(fn),'TP: {}'.format(tp)]])
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(conf, annot=labels , fmt= '', annot_kws={"size": 15})