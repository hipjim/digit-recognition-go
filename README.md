# go-neural-digits

A neural network implementation in Go built from scratch to recognize handwritten digits (0-9). This project is designed as a learning exercise to understand the fundamentals of neural networks without using external ML libraries.

## Features

- **Pure Go implementation** - No external ML frameworks
- **Educational focus** - Well-commented code explaining each step
- **MNIST-compatible** - Designed to work with 28x28 pixel digit images
- **Configurable architecture** - Easy to modify network structure

## Architecture

- **Input Layer**: 784 neurons (28×28 pixels)
- **Hidden Layer**: 128 neurons (configurable)
- **Output Layer**: 10 neurons (digits 0-9)
- **Activation Function**: Sigmoid
- **Learning Algorithm**: Backpropagation with gradient descent

## Project Structure

```
go-neural-digits/
├── main.go           # Neural network implementation
├── README.md         # Project documentation
├── go.mod           # Go module file
└── data/            # Training/test data (to be added)
```

## Getting Started

### Prerequisites

- Go 1.19 or later

### Installation

```bash
git clone https://github.com/yourusername/digit-recognition-go.git
cd digit-recognition-go
go mod init digit-recognition-go
```

### Running

```bash
go run main.go
```

## How It Works

### 1. Forward Propagation
Input → Hidden Layer → Output Layer
- Matrix multiplication with weights
- Add biases
- Apply sigmoid activation

### 2. Backward Propagation
Calculate errors and update weights:
- Compute output error
- Propagate error backward through network
- Update weights using gradient descent

### 3. Training Loop
Repeat forward and backward propagation on training data to minimize error.

## Learning Objectives

This project demonstrates:
- **Matrix operations** for neural networks
- **Forward propagation** (prediction)
- **Backward propagation** (learning)
- **Gradient descent** optimization
- **Activation functions** (sigmoid)
- **Weight initialization** strategies

## Roadmap

- [x] Basic neural network structure
- [x] Matrix operations
- [x] Forward propagation
- [ ] Backward propagation
- [ ] Training loop
- [ ] Test data generation
- [ ] MNIST data loading
- [ ] Performance metrics
- [ ] Visualization tools

## Contributing

This is an educational project! Feel free to:
- Add comments for clarity
- Implement additional activation functions
- Add data preprocessing utilities
- Improve performance metrics

## License

MIT License - Feel free to use this for learning!

## Resources

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

---

**Note**: This is a learning implementation optimized for understanding, not production performance.