// digit-recognition-go: A simple neural network implementation in Go for digit recognition
// Built from scratch to understand neural network fundamentals
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Matrix represents a 2D matrix for our neural network computations
// This is the fundamental data structure for storing weights, biases, and intermediate values
type Matrix struct {
	rows, cols int         // Dimensions of the matrix
	data       [][]float64 // 2D slice containing the actual matrix values
}

// NewMatrix creates a new matrix with given dimensions initialized to zero
// Parameters:
//   - rows: number of rows in the matrix
//   - cols: number of columns in the matrix
//
// Returns: pointer to a new Matrix with all values set to 0.0
func NewMatrix(rows, cols int) *Matrix {
	// Create a 2D slice: first allocate slice of row pointers
	data := make([][]float64, rows)
	// Then allocate each row as a slice of float64
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{rows: rows, cols: cols, data: data}
}

// RandomMatrix creates a matrix with random values between -1 and 1
// This is used for weight initialization - random starting weights are crucial
// for breaking symmetry and allowing the network to learn effectively
// Parameters:
//   - rows: number of rows in the matrix
//   - cols: number of columns in the matrix
//
// Returns: pointer to a Matrix with random values in range [-1, 1]
func RandomMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// rand.Float64() returns [0,1), so *2-1 gives us [-1,1)
			m.data[i][j] = rand.Float64()*2 - 1
		}
	}
	return m
}

// NeuralNetwork represents our complete neural network structure
// This contains all the parameters and hyperparameters needed for the network
type NeuralNetwork struct {
	// Network architecture
	inputSize  int // Number of input neurons (784 for 28x28 images)
	hiddenSize int // Number of hidden neurons (affects learning capacity)
	outputSize int // Number of output neurons (10 for digits 0-9)

	// Network parameters (learned during training)
	weightsIH *Matrix // Weights connecting Input to Hidden layer [hiddenSize x inputSize]
	weightsHO *Matrix // Weights connecting Hidden to Output layer [outputSize x hiddenSize]
	biasH     *Matrix // Bias values for Hidden layer neurons [hiddenSize x 1]
	biasO     *Matrix // Bias values for Output layer neurons [outputSize x 1]

	// Training hyperparameter
	learningRate float64 // How fast the network learns (typically 0.01 to 0.3)
}

// NewNeuralNetwork creates and initializes a new neural network
// All weights are randomly initialized to break symmetry
// Parameters:
//   - inputSize: number of input features (e.g., 784 for 28x28 pixel images)
//   - hiddenSize: number of neurons in hidden layer (more = more learning capacity)
//   - outputSize: number of output classes (e.g., 10 for digits 0-9)
//   - learningRate: step size for gradient descent (how fast to learn)
//
// Returns: pointer to initialized NeuralNetwork
func NewNeuralNetwork(inputSize, hiddenSize, outputSize int, learningRate float64) *NeuralNetwork {
	// Seed random number generator for reproducible results during development
	rand.Seed(time.Now().UnixNano())

	return &NeuralNetwork{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		// Initialize weight matrices with random values
		// Note: Matrix dimensions are [output_size x input_size] for matrix multiplication
		weightsIH:    RandomMatrix(hiddenSize, inputSize),  // [128 x 784]
		weightsHO:    RandomMatrix(outputSize, hiddenSize), // [10 x 128]
		biasH:        RandomMatrix(hiddenSize, 1),          // [128 x 1]
		biasO:        RandomMatrix(outputSize, 1),          // [10 x 1]
		learningRate: learningRate,
	}
}

// sigmoid is the activation function that introduces non-linearity to the network
// It squashes any input value to a range between 0 and 1
// Formula: f(x) = 1 / (1 + e^(-x))
// Properties:
//   - Output range: (0, 1)
//   - Smooth and differentiable
//   - S-shaped curve
//
// Parameter: x - input value (can be any real number)
// Returns: sigmoid of x, a value between 0 and 1
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidDerivative computes the derivative of the sigmoid function
// This is needed for backpropagation (the learning algorithm)
// Formula: f'(x) = f(x) * (1 - f(x)) where f(x) is sigmoid(x)
// Note: This function expects x to already be the sigmoid output, not the original input
// Parameter: x - the output of sigmoid function (value between 0 and 1)
// Returns: derivative value used in backpropagation
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// MatrixMultiply performs matrix multiplication: C = A × B
// This is the core operation in neural networks for combining inputs with weights
// Rules: A[m×n] × B[n×p] = C[m×p] (A's columns must equal B's rows)
// Each element C[i][j] = sum of A[i][k] * B[k][j] for all k
// Parameters:
//   - a: first matrix (left operand)
//   - b: second matrix (right operand)
//
// Returns: new matrix containing the result of A × B
// Panics: if matrix dimensions are incompatible for multiplication
func MatrixMultiply(a, b *Matrix) *Matrix {
	// Validate dimensions: a.cols must equal b.rows
	if a.cols != b.rows {
		panic("Matrix dimensions don't match for multiplication")
	}

	// Result matrix has dimensions [a.rows × b.cols]
	result := NewMatrix(a.rows, b.cols)

	// Standard matrix multiplication algorithm
	for i := 0; i < a.rows; i++ { // For each row in A
		for j := 0; j < b.cols; j++ { // For each column in B
			sum := 0.0
			// Compute dot product of row i from A and column j from B
			for k := 0; k < a.cols; k++ {
				sum += a.data[i][k] * b.data[k][j]
			}
			result.data[i][j] = sum
		}
	}
	return result
}

// Add performs element-wise addition of two matrices: this = this + other
// This modifies the receiver matrix in-place
// Used for adding bias values to weighted sums in neural network computations
// Parameters:
//   - other: matrix to add to the receiver
//
// Modifies: the receiver matrix (this matrix is changed)
// Requirements: both matrices must have identical dimensions
func (m *Matrix) Add(other *Matrix) {
	// Add corresponding elements
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] += other.data[i][j]
		}
	}
}

// Sigmoid applies the sigmoid activation function to every element in the matrix
// This modifies the matrix in-place, transforming all values to range (0, 1)
// Used after computing weighted sums to introduce non-linearity
// Modifies: the receiver matrix (all elements are transformed)
func (m *Matrix) Sigmoid() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = sigmoid(m.data[i][j])
		}
	}
}

// Print displays the matrix in a readable format for debugging
// Each row is printed on a separate line with values formatted to 4 decimal places
// Used during development to inspect matrix contents and verify calculations
func (m *Matrix) Print() {
	for i := 0; i < m.rows; i++ {
		fmt.Printf("[ ")
		for j := 0; j < m.cols; j++ {
			fmt.Printf("%.4f ", m.data[i][j])
		}
		fmt.Printf("]\n")
	}
	fmt.Println()
}

// Copy creates a deep copy of the matrix
// Returns a completely independent matrix with the same values
// Used when you need to preserve the original matrix while modifying a copy
// Returns: new Matrix with identical values but separate memory allocation
func (m *Matrix) Copy() *Matrix {
	result := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[i][j] = m.data[i][j]
		}
	}
	return result
}

// FromArray converts a 1D array to a column matrix (vector)
// This is used to convert input data (like pixel values) into matrix format
// for neural network computations
// Parameters:
//   - arr: 1D slice of float64 values
//
// Returns: column matrix [len(arr) × 1] containing the array values
// Example: [1, 2, 3] becomes:
//
//	[1]
//	[2]
//	[3]
func FromArray(arr []float64) *Matrix {
	result := NewMatrix(len(arr), 1)
	for i := 0; i < len(arr); i++ {
		result.data[i][0] = arr[i]
	}
	return result
}

// ToArray converts a matrix to a 1D array (flattened)
// This is used to convert neural network outputs back to simple array format
// Reads matrix elements row by row, left to right
// Returns: 1D slice containing all matrix elements in row-major order
// Example: [1 2]  becomes [1, 2, 3, 4]
//
//	[3 4]
func (m *Matrix) ToArray() []float64 {
	result := make([]float64, m.rows*m.cols)
	index := 0
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result[index] = m.data[i][j]
			index++
		}
	}
	return result
}

// Forward propagation methods for NeuralNetwork

// FeedForward performs forward propagation through the neural network
// This is the prediction process: input → hidden layer → output layer
//
// Steps:
// 1. Convert input to matrix format
// 2. Compute hidden layer: sigmoid(weights_ih × input + bias_h)
// 3. Compute output layer: sigmoid(weights_ho × hidden + bias_o)
// 4. Return output probabilities
//
// Parameters:
//   - inputArray: array of input features (e.g., 784 pixel values for 28×28 image)
//
// Returns: array of output probabilities (e.g., 10 values for digits 0-9)
func (nn *NeuralNetwork) FeedForward(inputArray []float64) []float64 {
	// Step 1: Convert input array to column matrix for matrix operations
	inputs := FromArray(inputArray) // [784 × 1]

	// Step 2: Calculate hidden layer activations
	// Formula: hidden = sigmoid(weights_ih × inputs + bias_h)
	hidden := MatrixMultiply(nn.weightsIH, inputs) // [128 × 784] × [784 × 1] = [128 × 1]
	hidden.Add(nn.biasH)                           // Add bias: [128 × 1] + [128 × 1]
	hidden.Sigmoid()                               // Apply activation function

	// Step 3: Calculate output layer activations
	// Formula: output = sigmoid(weights_ho × hidden + bias_o)
	outputs := MatrixMultiply(nn.weightsHO, hidden) // [10 × 128] × [128 × 1] = [10 × 1]
	outputs.Add(nn.biasO)                           // Add bias: [10 × 1] + [10 × 1]
	outputs.Sigmoid()                               // Apply activation function

	// Step 4: Convert result back to array format for easy use
	return outputs.ToArray() // [10] - probabilities for each digit class
}

// Predict returns the most likely digit (0-9) for the given input
// This is a convenience function that runs forward propagation and finds
// the output neuron with the highest activation (highest probability)
//
// Parameters:
//   - inputArray: array of input features (e.g., pixel values)
//
// Returns: integer from 0-9 representing the predicted digit
func (nn *NeuralNetwork) Predict(inputArray []float64) int {
	// Get probabilities for all 10 digit classes
	outputs := nn.FeedForward(inputArray)

	// Find the index (digit) with highest probability
	maxIndex := 0
	maxValue := outputs[0]
	for i := 1; i < len(outputs); i++ {
		if outputs[i] > maxValue {
			maxValue = outputs[i]
			maxIndex = i
		}
	}

	return maxIndex // Return the predicted digit (0-9)
}

// GetConfidence returns both the predicted digit and the confidence score
// Confidence score is the probability (0-1) that the network assigns to its prediction
// Higher confidence means the network is more certain about its prediction
//
// Parameters:
//   - inputArray: array of input features (e.g., pixel values)
//
// Returns:
//   - int: predicted digit (0-9)
//   - float64: confidence score (0.0 to 1.0, where 1.0 = 100% confident)
func (nn *NeuralNetwork) GetConfidence(inputArray []float64) (int, float64) {
	// Get probabilities for all 10 digit classes
	outputs := nn.FeedForward(inputArray)

	// Find the digit with highest probability and its confidence value
	maxIndex := 0
	maxValue := outputs[0]
	for i := 1; i < len(outputs); i++ {
		if outputs[i] > maxValue {
			maxValue = outputs[i]
			maxIndex = i
		}
	}

	return maxIndex, maxValue // Return predicted digit and its probability
}

func main() {
	// Create a neural network for digit recognition
	// Architecture explanation:
	// - Input: 28×28 = 784 pixels (flattened MNIST image)
	// - Hidden: 128 neurons (good balance of learning capacity vs. speed)
	// - Output: 10 classes (digits 0-9)
	// - Learning rate: 0.1 (moderate learning speed)
	nn := NewNeuralNetwork(784, 128, 10, 0.1)

	// Display network information
	fmt.Println("Neural Network Created!")
	fmt.Printf("Input size: %d\n", nn.inputSize)
	fmt.Printf("Hidden size: %d\n", nn.hiddenSize)
	fmt.Printf("Output size: %d\n", nn.outputSize)
	fmt.Printf("Learning rate: %.2f\n", nn.learningRate)

	// Show weight matrix dimensions (useful for understanding data flow)
	fmt.Println("\nInput to Hidden weights shape:", nn.weightsIH.rows, "x", nn.weightsIH.cols)
	fmt.Println("Hidden to Output weights shape:", nn.weightsHO.rows, "x", nn.weightsHO.cols)

	// Test forward propagation with random data
	fmt.Println("\n=== Testing Forward Propagation ===")

	// Create test input simulating a 28×28 pixel image
	// In real usage, this would be actual pixel values normalized to [0,1]
	// For now, we use random values to test the forward propagation mechanism
	testInput := make([]float64, 784)
	for i := range testInput {
		testInput[i] = rand.Float64() // Random pixel values between 0 and 1
	}

	// Test the prediction functions
	prediction := nn.Predict(testInput)
	_, confidence := nn.GetConfidence(testInput)

	fmt.Printf("Predicted digit: %d\n", prediction)
	fmt.Printf("Confidence: %.4f (%.2f%%)\n", confidence, confidence*100)

	// Show detailed output probabilities for all digits
	// This helps understand how the network "thinks" about each possibility
	outputs := nn.FeedForward(testInput)
	fmt.Println("\nAll output probabilities:")
	for i, prob := range outputs {
		fmt.Printf("Digit %d: %.4f (%.2f%%)\n", i, prob, prob*100)
	}

	// Test with multiple random inputs to see variety in predictions
	fmt.Println("\n=== Testing Multiple Predictions ===")
	for i := 0; i < 5; i++ {
		// Generate new random input for each test
		testInput := make([]float64, 784)
		for j := range testInput {
			testInput[j] = rand.Float64()
		}

		digit, confidence := nn.GetConfidence(testInput)
		fmt.Printf("Test %d - Predicted: %d, Confidence: %.2f%%\n",
			i+1, digit, confidence*100)
	}
}
