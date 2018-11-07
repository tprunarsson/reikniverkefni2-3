#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon

"""
Neural Networks consist of the following components

An input layer, x - the state, a board, the dice and which player is to move.
An arbitrary amount of hidden layers
An output layer, ŷ - The best move according to the policy.
A set of weights and biases between each layer, W and b
A choice of activation function for each hidden layer, σ. - Sigmoid activation function.

Each iteration of the training process consists of the following steps:

Calculating the predicted output ŷ, known as feedforward
Updating the weights and biases, known as backpropagation
"""

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)   
# The output y of a simple 2-layer Neural Network is y = sigma(w_2 * sigma(w_1*x + b_1) + b_2)
        self.y          = y
        self.output     = np.zeros(y.shape)
        
# Calculating the predicted output y.
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

# The loss function evaluates how far off our predictions are.
# Sum-of-squares-error loss function
# Our goal in training is to find the best set of weights and biases that 
# minimizes the loss function.

# Backpropagation propagates the error back and updates our weights and biases.
# Use the derivative of the loss function with respect ot the weights and biases
# to know the appropiate amount to adjust the weights and biases by.

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        

if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)


"""
Here’s a brief overview of how a simple feedforward neural network works:

1.Takes inputs as a matrix (2D array of numbers) 

Spurning: Hvernig komum við teningnum inn í inputið ásamt boardinu?

2.Multiplies the input by a set weights (performs a dot product aka matrix multiplication)

3.Applies an activation function

4.Returns an output

5.Error is calculated by taking the difference from the desired output from the data and the predicted output. This creates our gradient descent, which we can use to alter the weights

Spurning: Hvernig vitum við predicted output?

6.The weights are then altered slightly according to the error.

7.To train, this process is repeated 1,000+ times. The more the data is trained upon, the more accurate our outputs will be.

Spurning: Við erum með eitthvað input, sem ég myndi giska á að væri boardið plús teningurinn en 
hvað ætti outputið að vera? Reward, nýja staðan á boardinu, moves sem eru í boði, array með þeim moves
sem eru í boði þar sem hvert move hefur gildi sem táknar hversu gott er að taka viðkomandi move?
Við viljum nota neural networkið til að ákvarða move? 
Þegar neural networkið er þjálfar ætti það þá ekki að taka inn board og tening og skila
því movei sem er best að taka?
"""
"""
Here’s how we will calculate the incremental change to our weights:

1.Find the margin of error of the output layer (o) by taking the difference of the predicted output and the actual output (y)

2.Apply the derivative of our sigmoid activation function to the output layer error. We call this result the delta output sum.

3.Use the delta output sum of the output layer error to figure out how much our z2 (hidden) layer contributed to the output error by performing a dot product with our second weight matrix. We can call this the z2 error.

4.Calculate the delta output sum for the z2 layer by applying the derivative of our sigmoid activation function (just like step 2).

5.Adjust the weights for the first layer by performing a dot product of the input layer with the hidden (z2) delta output sum. For the second layer, perform a dot product of the hidden(z2) layer and the output (o) delta output sum.
"""

# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
xPredicted = np.array(([4,8]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
y = y/100 # max test score is 100

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print("Predicted data based on trained weights: ");
    print("Input (scaled): \n" + str(xPredicted));
    print("Output: \n" + str(self.forward(xPredicted)));

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
  print("# " + str(i) + "\n")
  print("Input (scaled): \n" + str(X))
  print("Actual Output: \n" + str(y))
  print("Predicted Output: \n" + str(NN.forward(X)))
  print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()






def action(board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    # make the best move according to the policy
    
    # policy missing, returns a random move for the time being
    #
    #
    #
    #
    #
    move = possible_moves[np.random.randint(len(possible_moves))]

    return move