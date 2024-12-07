import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from preProcessing import df_filled, df_dropped

# Pairplot visualization for Drop NaN rows
sns.pairplot(df_dropped, hue='Rain', palette='Set2')
plt.title("Pairplot for 'df_dropped' (Rows Dropped)", y=1.02)
plt.show()

# Pairplot visualization
sns.pairplot(df_filled, hue='Rain', palette='Set1')
plt.title("Pairplot for 'df_filled' (filled with the mean of the column)", y=1.02)
plt.show()

# Preparing data for training
X = df_filled.drop('Rain', axis=1)
y = df_filled['Rain']

# Train Decision Tree on df_dropped
X_dropped = df_dropped.drop('Rain', axis=1)
y_dropped = df_dropped['Rain']
X_train_dropped, X_test_dropped, y_train_dropped, y_test_dropped = train_test_split(X_dropped, y_dropped, test_size=0.20, random_state=42)

# Scale the data (fit on the training set, transform on both train and test sets)
scaler = StandardScaler()
X_train_dropped_scaled = scaler.fit_transform(X_train_dropped)  # Fit on training data
X_test_dropped_scaled = scaler.transform(X_test_dropped)        # Use the same statistics for the test data

# Train Decision Tree
dtree_dropped = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtree_dropped.fit(X_train_dropped_scaled, y_train_dropped)

predictions_dropped = dtree_dropped.predict(X_test_dropped_scaled)
print("Performance on df_dropped:")
print(confusion_matrix(y_test_dropped, predictions_dropped))

# Calculate and print precision, recall, and accuracy
precision_dropped = precision_score(y_test_dropped, predictions_dropped, pos_label="rain")
recall_dropped = recall_score(y_test_dropped, predictions_dropped, pos_label="rain")
accuracy_dropped = accuracy_score(y_test_dropped, predictions_dropped)

print(f"Precision on df_dropped: {precision_dropped:.8f}")
print(f"Recall on df_dropped: {recall_dropped:.8f}")
print(f"Accuracy on df_dropped: {accuracy_dropped:.8f}")

# Visualize the decision tree for df_dropped
plt.figure(figsize=(12, 8))
plot_tree(dtree_dropped, 
          feature_names=X_dropped.columns, 
          class_names=dtree_dropped.classes_.astype(str), 
          filled=True, 
          rounded=True, 
          fontsize=5)
plt.title("Decision Tree Visualization for df_dropped")
plt.show()



# Train Decision Tree on df_filled
X_filled = df_filled.drop('Rain', axis=1)
y_filled = df_filled['Rain']
X_train_filled, X_test_filled, y_train_filled, y_test_filled = train_test_split(X_filled, y_filled, test_size=0.20, random_state=42)

# Scale the data (fit on the training set, transform on both train and test sets)
X_train_filled_scaled = scaler.fit_transform(X_train_filled)  # Fit on training data
X_test_filled_scaled = scaler.transform(X_test_filled)        # Use the same statistics for the test data

# Train Decision Tree
dtree_filled = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtree_filled.fit(X_train_filled_scaled, y_train_filled)

predictions_filled = dtree_filled.predict(X_test_filled_scaled)
print("Performance on df_filled:")
print(confusion_matrix(y_test_filled, predictions_filled))

# Calculate and print precision, recall, and accuracy
precision_filled = precision_score(y_test_filled, predictions_filled, pos_label="rain")
recall_filled = recall_score(y_test_filled, predictions_filled, pos_label="rain")
accuracy_filled = accuracy_score(y_test_filled, predictions_filled)

print(f"Precision on df_filled: {precision_filled:.8f}")
print(f"Recall on df_filled: {recall_filled:.8f}")
print(f"Accuracy on df_filled: {accuracy_filled:.8f}")

# Visualize the decision tree for df_filled
plt.figure(figsize=(12, 8))
plot_tree(dtree_filled, 
          feature_names=X_filled.columns, 
          class_names=dtree_filled.classes_.astype(str), 
          filled=True, 
          rounded=True, 
          fontsize=5)
plt.title("Decision Tree Visualization for df_filled")
plt.show()
