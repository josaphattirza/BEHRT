import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# # Define the AUC value
# auc = 0.9 (produce 0.88 AUC curve)
auc = 0.9
auc2 = 0.864

# Generate random data points for two classes with different means
n = 100000
pos_mean = 0.7 + (0.3 - 0.7) * (1 - auc)  # Adjust the mean of the positive class
neg_mean = 0.3 + (0.7 - 0.3) * (1 - auc)  # Adjust the mean of the negative class
pos_probs = np.random.normal(loc=pos_mean, scale=0.2, size=int(auc * n))
neg_probs = np.random.normal(loc=neg_mean, scale=0.2, size=int((1 - auc) * n))
probs = np.concatenate([pos_probs, neg_probs])

n2 = 100000
# pos_mean2 = 0.7 + (0.3 - 0.7) * (1 -  (produce 0.88 AUC curve)auc2)  # Adjust the mean of the positive class
neg_mean2 = 0.3 + (0.7 - 0.3) * (1 - auc2)  # Adjust the mean of the negative class
pos_probs2 = np.random.normal(loc=pos_mean2, scale=0.2, size=int(auc2 * n2))
neg_probs2 = np.random.normal(loc=neg_mean2, scale=0.2, size=int((1 - auc2) * n2))
probs2 = np.concatenate([pos_probs2, neg_probs2])


# Calculate the threshold
threshold = np.percentile(probs, (1 - auc) * 100)

# Calculate the threshold
threshold2 = np.percentile(probs2, (1 - auc2) * 100)

# Calculate the true positive rate and false positive rate at each threshold
tpr = []
fpr = []
for thresh in np.linspace(0, 1, num=100):
    tp = (np.sum(pos_probs >= thresh)) / len(pos_probs)
    fp = np.sum(neg_probs >= thresh) / len(neg_probs)
    tpr.append(tp)
    fpr.append(fp)

# print(tpr)
# print(fpr)

# Calculate the AUC
roc_auc = np.trapz(tpr, fpr)

# Calculate the true positive rate and false positive rate at each threshold
tpr2 = []
fpr2 = []
for thresh2 in np.linspace(0, 1, num=100):
    tp2 = np.sum(pos_probs2 >= thresh2) / len(pos_probs2)
    fp2 = np.sum(neg_probs2 >= thresh2) / len(neg_probs2)
    tpr2.append(tp2)
    fpr2.append(fp2)

# print(tpr)
# print(fpr)

# Calculate the AUC
roc_auc2 = np.trapz(tpr2, fpr2)


# Plot the ROC curve
plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % auc)
# plt.plot(fpr2, tpr2, lw=2, label='ROC curve (AUC2 = %0.2f)' % roc_auc2)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.show()
