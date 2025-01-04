# FeedForward Neural Network for Predicting Cement Strength

Welcome to the **FeedForward Neural Network Project**, where data meets intelligence! This project demonstrates a comprehensive pipeline for processing, training, and predicting concrete compressive strength using a custom-built neural network. 🏗️

---

## 🚀 Features
- **Data Loading & Preprocessing**: Effortlessly handles Excel data, normalizes features, and splits datasets.
- **Custom Neural Network**: A fully connected feedforward neural network with Xavier initialization and sigmoid activation.
- **Training & Evaluation**: Learn from data with backpropagation and evaluate performance using MSE and MAE metrics.
- **Interactive Predictions**: Make real-time predictions for new input data!

---

## 📂 File Structure
- `concrete_data.xlsx`: Example dataset (update the path to match your local setup).
- Python script containing all the magic you see here.

---

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd <repo-folder>
   ```

2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn numpy
   ```

3. Add your dataset (`concrete_data.xlsx`) to the specified path in the script.

4. Run the program:
   ```bash
   python <script-name>.py
   ```

---

## 🧩 Code Walkthrough

### 1. **Data Loading**
```python
input_features, target_values = load_and_process_data(file_path)
```
Loads Excel data and extracts input features and target values.

### 2. **Feature Scaling**
```python
scaled_features, feature_scaler = scale_features(input_features)
scaled_targets, target_scaler = scale_targets(target_values)
```
Normalizes features and target values using Min-Max scaling for faster and stable training.

### 3. **Dataset Splitting**
```python
features_train, features_test, targets_train, targets_test = split_dataset(scaled_features, scaled_targets)
```
Splits the dataset into training and testing sets (default: 75% training).

### 4. **Neural Network Training**
```python
nn_model.train_model(features_train.T, targets_train.reshape(1, -1), num_epochs=1000)
```
Trains a custom feedforward neural network using backpropagation.

### 5. **Model Evaluation**
```python
mse, mae = nn_model.calculate_errors(original_targets, original_predictions)
```
Evaluates the model using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

### 6. **Interactive Predictions**
```python
while True:
    user_input = input("Enter new data...")
```
Provides an interactive interface to predict the compressive strength of new concrete mixes.

---

## 🎯 Sample Usage

1. **Load the Dataset**:
   ```python
   Input Features Shape: (1030, 4)
   Target Values Shape: (1030,)
   ```

2. **Training Progress**:
   ```bash
   Epoch 100/1000, Cost: 0.0345
   Epoch 200/1000, Cost: 0.0256
   ...
   ```

3. **Predictions**:
   ```bash
   Predictions for Test Data: [35.67, 45.23, 28.90, ...]
   Test MSE: 5.3421
   Test MAE: 2.1234
   ```

4. **Interactive Input**:
   ```bash
   Enter new data (cement, water, superplasticizer, age): 300,150,5,28
   Predicted Cement Strength: 42.987654
   ```

---

## 🔍 Key Insights
- **Neural Network Design**: The architecture is flexible, allowing adjustments to layer sizes and learning rates.
- **Data Scaling**: Essential for reducing computational complexity and improving model convergence.
- **Regression Use Case**: Predicting continuous values like compressive strength is made straightforward.

---

## 🤖 How It Works
1. **Forward Pass**:
   - Activates input data through hidden and output layers.
2. **Backward Pass**:
   - Computes gradients for weights and biases using backpropagation.
3. **Weight Update**:
   - Updates parameters using gradient descent.
4. **Prediction**:
   - Scaled predictions are transformed back to their original range.

---

## 🌟 Why You'll Love This Project
- **Educational**: Gain a hands-on understanding of how neural networks are implemented from scratch.
- **Customizable**: Modify network parameters, input features, and target values for your dataset.
- **Interactive**: Predict real-world outcomes instantly using user input.

---

## 🧑‍💻 Contribute
Feel free to fork the repository and improve the model by adding new features or optimizing the architecture.

---

## 🗂️ Dataset
Ensure your dataset (`concrete_data.xlsx`) contains the following columns:
- Cement
- Water
- Superplasticizer
- Age (in days)
- Compressive Strength (target)

---

## 📬 Feedback
We'd love to hear your thoughts or help you troubleshoot issues. Open an issue on the repository or drop us a message!

