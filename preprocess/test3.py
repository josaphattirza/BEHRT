import numpy as np
import matplotlib.pyplot as plt

# Define the AUC value
auc = 0.8

# Generate random data points for two classes with different means
np.random.seed(0)
n = 1000
pos_mean = 0.7 + (0.3 - 0.7) * (1 - auc)  # Adjust the mean of the positive class
neg_mean = 0.3 + (0.7 - 0.3) * (1 - auc)  # Adjust the mean of the negative class
pos_probs = np.random.normal(loc=pos_mean, scale=0.2, size=int(auc * n))
neg_probs = np.random.normal(loc=neg_mean, scale=0.2, size=int((1 - auc) * n))
probs = np.concatenate([pos_probs, neg_probs])

# Calculate the threshold
threshold = np.percentile(probs, (1 - auc) * 100)

# Calculate the true positive rate and false positive rate at each threshold
tpr = []
fpr = []
for thresh in np.linspace(0, 1, num=100):
    tp = np.sum(pos_probs >= thresh) / len(pos_probs)
    fp = np.sum(neg_probs >= thresh) / len(neg_probs)
    tpr.append(tp)
    fpr.append(fp)

# Calculate the AUC
roc_auc = np.trapz(tpr, fpr)

# Plot the ROC curve
plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
