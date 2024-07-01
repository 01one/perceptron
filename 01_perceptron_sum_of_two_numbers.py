from random import uniform, randint
import math

# Initialize weights and biases with random values between 0 and 1
w1 = uniform(0, 1)
w2 = uniform(0, 1)
bias = uniform(0, 1)



# Function to perform training
def train_perceptron():
    global w1, w2, bias, accuracies, epochs_list
    
    # Generate training data
    a_train = [randint(1, 100) for _ in range(1001)]
    b_train = [randint(1, 100) for _ in range(1001)]
    y_train = [a + b for a, b in zip(a_train, b_train)]
    m = len(a_train)
    
    # Number of training epochs and learning rate
    epochs = 10000
    learning_rate = 0.00001  # Reduced learning rate
    
    accuracies = []
    epochs_list = []
    
    # Perform training for specified epochs
    for epoch in range(epochs):
        correct_predictions = 0
        
        for i in range(m):
            # Forward pass
            z = w1 * a_train[i] + w2 * b_train[i] + bias
            prediction = z
            
            # Compute the cost (mean squared error)
            error = (prediction - y_train[i])
            
            # Backward pass (calculate gradients)
            dW1 = error * a_train[i]
            dW2 = error * b_train[i]
            dB = error
            
            # Gradient clipping (optional but recommended)
            max_gradient = 10.0
            dW1 = max(min(dW1, max_gradient), -max_gradient)
            dW2 = max(min(dW2, max_gradient), -max_gradient)
            dB = max(min(dB, max_gradient), -max_gradient)
            
            # Update weights and biases using gradients and learning rate
            w1 -= learning_rate * dW1
            w2 -= learning_rate * dW2
            bias -= learning_rate * dB
            
            # Check prediction accuracy
            if abs(prediction - y_train[i]) < 0.1:  # Adjust threshold as needed
                correct_predictions += 1
        
        # Calculate accuracy for current epoch
        accuracy = correct_predictions / m * 100
        accuracies.append(accuracy)
        epochs_list.append(epoch)
        
        # Print accuracy every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Accuracy: {accuracy}%")

# Calculate the result based on the trained weights and biases for a specific pair (a, b)
def test_perceptron(a, b):
    z = w1 * a + w2 * b + bias
    return z

# Train the perceptron model
train_perceptron()

# Test the model with specific pairs of numbers
a_test1, b_test1 = 10, 20
result1 = test_perceptron(a_test1, b_test1)

a_test2, b_test2 = 5, 7.9
result2 = test_perceptron(a_test2, b_test2)

# Print the results and final values of weights and biases
print(f"Result (sum of {a_test1} and {b_test1}):", result1)
print(f"Result (sum of {a_test2} and {b_test2}):", result2)
print("Final values - w1:", w1, "w2:", w2, "bias:", bias)



i1=int(input("Enter the first number: "))

i2=int(input("Enter the second number: "))
print(test_perceptron(i1,i2))
