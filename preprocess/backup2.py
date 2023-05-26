import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset with adjustable class separation
X, y = make_classification(
    n_samples=1000000,
    n_features=10,
    n_informative=8,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.5, 0.5],
    class_sep=0.405,  # Adjust the class separation
    # 0.5 results in AUC 0.93
    # 0.405 results in AUC 0.883 
)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your classifier (replace with your own classifier)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_probs = classifier.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_probs)
print(auc_score)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
# Set the x-axis limits without the gap
plt.xlim([0, 1])
# Set the y-axis limits without the gap
plt.ylim([0, 1])
plt.show()
