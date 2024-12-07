# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from preProcessing import df_filled , df_dropped

# Extract features and labels for both datasets
X_filled = df_filled.iloc[:, :-1].values
y_filled = df_filled.iloc[:, -1].values

X_dropped = df_dropped.iloc[:, :-1].values
y_dropped = df_dropped.iloc[:, -1].values

# Split data into training and testing sets
X_train_filled, X_test_filled, y_train_filled, y_test_filled = train_test_split(
    X_filled, y_filled, test_size=0.20, random_state=0)

X_train_dropped, X_test_dropped, y_train_dropped, y_test_dropped = train_test_split(
    X_dropped, y_dropped, test_size=0.20, random_state=0)

# Scaling: Fit on training set and transform both training and test sets
scaler = StandardScaler()

# For filled dataset
X_train_filled = scaler.fit_transform(X_train_filled)
X_test_filled = scaler.transform(X_test_filled)

# For dropped dataset
X_train_dropped = scaler.fit_transform(X_train_dropped)
X_test_dropped = scaler.transform(X_test_dropped)

# Initialize Naive Bayes Classifier
classifier_filled = GaussianNB()
classifier_dropped = GaussianNB()

# Train and predict for filled data
classifier_filled.fit(X_train_filled, y_train_filled)
y_pred_filled = classifier_filled.predict(X_test_filled)

# Train and predict for dropped data
classifier_dropped.fit(X_train_dropped, y_train_dropped)
y_pred_dropped = classifier_dropped.predict(X_test_dropped)

# Metrics for filled data
cm_filled = confusion_matrix(y_test_filled, y_pred_filled)
accuracy_filled = accuracy_score(y_test_filled, y_pred_filled)
precision_filled = precision_score(y_test_filled, y_pred_filled, pos_label="rain")
recall_filled = recall_score(y_test_filled, y_pred_filled, pos_label="rain")

# Metrics for dropped data
cm_dropped = confusion_matrix(y_test_dropped, y_pred_dropped)
accuracy_dropped = accuracy_score(y_test_dropped, y_pred_dropped)
precision_dropped = precision_score(y_test_dropped, y_pred_dropped, pos_label="rain")
recall_dropped = recall_score(y_test_dropped, y_pred_dropped, pos_label="rain")

# Report
print("Performance Metrics for Naïve Bayes with Filled Data:")
print(f"Confusion Matrix:\n{cm_filled}")
print(f"Accuracy: {accuracy_filled:.8f}")
print(f"Precision: {precision_filled:.8f}")
print(f"Recall: {recall_filled:.8f}\n")

print("Performance Metrics for Naïve Bayes with Dropped Data:")
print(f"Confusion Matrix:\n{cm_dropped}")
print(f"Accuracy: {accuracy_dropped:.8f}")
print(f"Precision: {precision_dropped:.8f}")
print(f"Recall: {recall_dropped:.8f}")

# Create a DataFrame for visualization
metrics_data = {
    "Metric": ["Accuracy", "Precision", "Recall"] * 2,
    "Value": [accuracy_filled, precision_filled, recall_filled, 
              accuracy_dropped, precision_dropped, recall_dropped],
    "Dataset": ["Filled"] * 3 + ["Dropped"] * 3
}
metrics_df = pd.DataFrame(metrics_data)

# Create a comparison bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Metric", y="Value", hue="Dataset", data=metrics_df, palette="Set1")
plt.title("Comparison of Performance Metrics: Filled vs Dropped Datasets", fontsize=14)
plt.ylabel("Metric Value", fontsize=12)
plt.xlabel("Metric", fontsize=12)
plt.ylim(0, 1.1)
plt.legend(title="Dataset")
plt.tight_layout()
plt.show()
