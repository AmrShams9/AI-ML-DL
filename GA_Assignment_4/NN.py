import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_and_process_data(file_path):
    data = pd.read_excel(file_path)
    input_features = data.iloc[:, :4].values
    target_values = data.iloc[:, -1].values
    return input_features, target_values

def scale_features(features):
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    return scaled_features, feature_scaler

def scale_targets(targets):
    target_scaler = MinMaxScaler()
    scaled_targets = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    return scaled_targets, target_scaler

def split_dataset(features, targets, test_size=0.25, random_state=42):
    return train_test_split(features, targets, test_size=test_size, random_state=random_state)

def prepare_new_input(new_input, feature_scaler):
    scaled_new_input = feature_scaler.transform(new_input)
    return np.array(scaled_new_input).T

class FeedForwardNeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate=0.01):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.learning_rate = learning_rate

        # Xavier Initialization
        self.weights_input_to_hidden = np.random.randn(hidden_layer_size, input_layer_size) * np.sqrt(1 / input_layer_size)
        self.bias_hidden = np.zeros((hidden_layer_size, 1))
        self.weights_hidden_to_output = np.random.randn(output_layer_size, hidden_layer_size) * np.sqrt(1 / hidden_layer_size)
        self.bias_output = np.zeros((output_layer_size, 1))

    @staticmethod
    def sigmoid_activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(activation_output):
        return activation_output * (1 - activation_output)

    def forward_pass(self, input_data):
        z_hidden = np.dot(self.weights_input_to_hidden, input_data) + self.bias_hidden
        a_hidden = self.sigmoid_activation(z_hidden)
        z_output = np.dot(self.weights_hidden_to_output, a_hidden) + self.bias_output
        a_output = z_output  # Linear activation for regression
        return {"z_hidden": z_hidden, "a_hidden": a_hidden, "z_output": z_output, "a_output": a_output}

    def backward_pass(self, input_data, target_data, cache):
        num_samples = input_data.shape[1]
        a_hidden, a_output = cache["a_hidden"], cache["a_output"]

        error_output_layer = a_output - target_data
        gradient_weights_output = (1 / num_samples) * np.dot(error_output_layer, a_hidden.T)
        gradient_bias_output = (1 / num_samples) * np.sum(error_output_layer, axis=1, keepdims=True)

        error_hidden_layer = np.dot(self.weights_hidden_to_output.T, error_output_layer) * self.sigmoid_derivative(a_hidden)
        gradient_weights_hidden = (1 / num_samples) * np.dot(error_hidden_layer, input_data.T)
        gradient_bias_hidden = (1 / num_samples) * np.sum(error_hidden_layer, axis=1, keepdims=True)

        return {
            "gradient_weights_hidden": gradient_weights_hidden,
            "gradient_bias_hidden": gradient_bias_hidden,
            "gradient_weights_output": gradient_weights_output,
            "gradient_bias_output": gradient_bias_output
        }

    def update_weights_and_biases(self, gradients):
        self.weights_input_to_hidden -= self.learning_rate * gradients["gradient_weights_hidden"]
        self.bias_hidden -= self.learning_rate * gradients["gradient_bias_hidden"]
        self.weights_hidden_to_output -= self.learning_rate * gradients["gradient_weights_output"]
        self.bias_output -= self.learning_rate * gradients["gradient_bias_output"]

    def train_model(self, training_inputs, training_targets, num_epochs):
        for epoch in range(num_epochs):
            for sample_idx in range(training_inputs.shape[1]):
                single_input = training_inputs[:, sample_idx:sample_idx+1]
                single_target = training_targets[:, sample_idx:sample_idx+1]

                cache = self.forward_pass(single_input)
                gradients = self.backward_pass(single_input, single_target, cache)
                self.update_weights_and_biases(gradients)

            if (epoch + 1) % 100 == 0:
                predictions = self.predict(training_inputs)
                cost = np.mean((predictions - training_targets) ** 2)
                print(f"Epoch {epoch + 1}/{num_epochs}, Cost: {cost:.4f}")

    def predict(self, input_data):
        cache = self.forward_pass(input_data)
        return cache["a_output"]

    def calculate_errors(self, true_values, predicted_values):
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        return mse, mae

if __name__ == "__main__":
    file_path = "GA_Assignment_4\concrete_data.xlsx"

    # Load and process dataset
    input_features, target_values = load_and_process_data(file_path)
    print("Input Features Shape:", input_features.shape)
    print("Target Values Shape:", target_values.shape)

    # Normalize features and targets
    scaled_features, feature_scaler = scale_features(input_features)
    scaled_targets, target_scaler = scale_targets(target_values)
    print("Sample of Scaled Features:\n", scaled_features[:5])

    # Split data into training and testing sets
    features_train, features_test, targets_train, targets_test = split_dataset(scaled_features, scaled_targets)
    print("Training Features Shape:", features_train.shape, "Testing Features Shape:", features_test.shape)

    # Initialize and train the neural network
    nn_model = FeedForwardNeuralNetwork(input_layer_size=4, hidden_layer_size=8, output_layer_size=1, learning_rate=0.01)
    nn_model.train_model(features_train.T, targets_train.reshape(1, -1), num_epochs=1000)

    # Evaluate the model on test data
    test_predictions = nn_model.predict(features_test.T)
    original_predictions = target_scaler.inverse_transform(test_predictions.T)
    print("Predictions for Test Data:", original_predictions.flatten()[:10])

    original_targets = target_scaler.inverse_transform(targets_test.reshape(-1, 1))
    mse, mae = nn_model.calculate_errors(original_targets, original_predictions)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # Make predictions for new input
    new_input = np.array([[300, 150, 5, 28]])
    new_input_prediction = nn_model.predict(prepare_new_input(new_input, feature_scaler))
    original_prediction = target_scaler.inverse_transform(new_input_prediction.T)
    print("Predicted Strength for New Input:", original_prediction.flatten()[0])

    # Interactive prediction
    while True:
        user_input = input("Enter new data (cement, water, superplasticizer, age) separated by commas (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try:
            new_input_values = np.array([float(x) for x in user_input.split(',')]).reshape(1, -1)
            if new_input_values.shape[1] != 4:
                print("Invalid input! Please enter exactly 4 values (cement, water, superplasticizer, age).")
                continue
            normalized_input = prepare_new_input(new_input_values, feature_scaler)
            prediction = nn_model.predict(normalized_input)
            original_prediction = target_scaler.inverse_transform(prediction.T)
            print(f"Predicted Cement Strength: {original_prediction.flatten()[0]:.6f}")
        except ValueError:
            print("Invalid input! Please enter numeric values separated by commas.")