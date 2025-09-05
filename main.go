package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Matrix represents a 2D matrix for our neural network
type Matrix struct {
	rows, cols int
	data       [][]float64
}

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{rows: rows, cols: cols, data: data}
}

// RandomMatrix creates a matrix with random values between -1 and 1
func RandomMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.data[i][j] = rand.Float64()*2 - 1 // Random value between -1 and 1
		}
	}
	return m
}

// NeuralNetwork represents our neural network
type NeuralNetwork struct {
	inputSize  int
	hiddenSize int
	outputSize int

	// Weights and biases
	weightsIH *Matrix // Input to Hidden weights
	weightsHO *Matrix // Hidden to Output weights
	biasH     *Matrix // Hidden layer bias
	biasO     *Matrix // Output layer bias

	learningRate float64
}

// NewNeuralNetwork creates a new neural network
func NewNeuralNetwork(inputSize, hiddenSize, outputSize int, learningRate float64) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	return &NeuralNetwork{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		outputSize:   outputSize,
		weightsIH:    RandomMatrix(hiddenSize, inputSize),
		weightsHO:    RandomMatrix(outputSize, hiddenSize),
		biasH:        RandomMatrix(hiddenSize, 1),
		biasO:        RandomMatrix(outputSize, 1),
		learningRate: learningRate,
	}
}

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of sigmoid function
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// MatrixMultiply multiplies two matrices
func MatrixMultiply(a, b *Matrix) *Matrix {
	if a.cols != b.rows {
		panic("Matrix dimensions don't match for multiplication")
	}

	result := NewMatrix(a.rows, b.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < b.cols; j++ {
			sum := 0.0
			for k := 0; k < a.cols; k++ {
				sum += a.data[i][k] * b.data[k][j]
			}
			result.data[i][j] = sum
		}
	}
	return result
}

// Add matrices element-wise
func (m *Matrix) Add(other *Matrix) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] += other.data[i][j]
		}
	}
}

// Apply sigmoid function to all elements
func (m *Matrix) Sigmoid() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = sigmoid(m.data[i][j])
		}
	}
}

// Print matrix for debugging
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

func main() {
	// Create a neural network for digit recognition
	// Input: 28x28 = 784 pixels (for MNIST-like data)
	// Hidden: 128 neurons
	// Output: 10 classes (digits 0-9)
	nn := NewNeuralNetwork(784, 128, 10, 0.1)

	fmt.Println("Neural Network Created!")
	fmt.Printf("Input size: %d\n", nn.inputSize)
	fmt.Printf("Hidden size: %d\n", nn.hiddenSize)
	fmt.Printf("Output size: %d\n", nn.outputSize)
	fmt.Printf("Learning rate: %.2f\n", nn.learningRate)

	fmt.Println("\nInput to Hidden weights shape:", nn.weightsIH.rows, "x", nn.weightsIH.cols)
	fmt.Println("Hidden to Output weights shape:", nn.weightsHO.rows, "x", nn.weightsHO.cols)
}
