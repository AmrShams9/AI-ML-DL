import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from preProcessing import df_filled

class KNNClassifier:

    def __init__(self, train_features, train_labels, test_features, k_neighbors):
        self.train_features = np.array(train_features)
        self.train_labels = np.array(train_labels)
        self.test_features = np.array(test_features)
        self.k_neighbors = k_neighbors
        self.predicted_labels = []

    def euclidean_distance(self, feature_vector1, feature_vector2):
        squared_difference_sum = np.sum((feature_vector1 - feature_vector2) ** 2)
        return np.sqrt(squared_difference_sum)

    def predict(self):
        for test_point in self.test_features:
            distances = []
            for idx, train_point in enumerate(self.train_features):
                distance = self.euclidean_distance(train_point, test_point)
                distances.append((distance, idx))

            distances = sorted(distances)[:self.k_neighbors]
            neighbor_labels = [self.train_labels[idx] for _, idx in distances]

            class_counts = {0: 0, 1: 0}
            for label in neighbor_labels:
                class_counts[label] += 1

            if class_counts[0] == class_counts[1]:
                predicted_label = neighbor_labels[0]
            elif class_counts[0] > class_counts[1]:
                predicted_label = 0
            else:
                predicted_label = 1

            self.predicted_labels.append(predicted_label)

        return self.predicted_labels

    def calculate_accuracy(self, true_labels, predicted_labels):
        correct_predictions = np.sum(np.array(true_labels) == np.array(predicted_labels))
        return correct_predictions / len(true_labels)


# Encode labels as 0 and 1
label_mapping = {'no rain': 0, 'rain': 1}
df_filled['Rain'] = df_filled['Rain'].map(label_mapping)

# Split data
X = df_filled.drop('Rain', axis=1)
y = df_filled['Rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom kNN
k_values = [3, 5, 7, 9, 11]
metrics_comparison = []

for k in k_values:
    # Custom kNN
    custom_knn = KNNClassifier(X_train_scaled, y_train, X_test_scaled, k)
    custom_predictions = custom_knn.predict()

    # Decode predictions back to original labels
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    custom_predictions_decoded = [reverse_label_mapping[p] for p in custom_predictions]
    y_test_decoded = y_test.map(reverse_label_mapping)

    # Custom kNN Metrics
    custom_accuracy = accuracy_score(y_test_decoded, custom_predictions_decoded)
    custom_precision = precision_score(y_test_decoded, custom_predictions_decoded, pos_label='rain')
    custom_recall = recall_score(y_test_decoded, custom_predictions_decoded, pos_label='rain')
    # print(custom_accuracy)
    # print(custom_precision)
    # print(custom_recall)


    # Scikit-learn kNN
    sklearn_knn = KNeighborsClassifier(n_neighbors=k)
    sklearn_knn.fit(X_train_scaled, y_train)
    sklearn_predictions = sklearn_knn.predict(X_test_scaled)

    # Decode scikit-learn predictions
    sklearn_predictions_decoded = [reverse_label_mapping[p] for p in sklearn_predictions]

    # Scikit-learn kNN Metrics
    sklearn_accuracy = accuracy_score(y_test_decoded, sklearn_predictions_decoded)
    sklearn_precision = precision_score(y_test_decoded, sklearn_predictions_decoded, pos_label='rain')
    sklearn_recall = recall_score(y_test_decoded, sklearn_predictions_decoded, pos_label='rain')

    # Storing the results for each k
    metrics_comparison.append({
        'k': k,
        'custom_accuracy': custom_accuracy,
        'custom_precision': custom_precision,
        'custom_recall': custom_recall,
        'sklearn_accuracy': sklearn_accuracy,
        'sklearn_precision': sklearn_precision,
        'sklearn_recall': sklearn_recall
    })

# Convert results to a DataFrame for easy plotting
df_comparison = pd.DataFrame(metrics_comparison)

# Plotting the comparison of performance metrics for each k
df_comparison.set_index('k').plot(kind='bar', figsize=(10, 8), width=0.8)
plt.title('Performance Comparison for Custom kNN vs. Scikit-learn kNN (Different k values)')
plt.ylabel('Scores')
plt.xlabel('k')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

