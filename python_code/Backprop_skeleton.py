import math
import random
import copy


# The transfer function of neurons, g(x)
def logFunc(x):
    return (1.0 / (1.0 + math.exp(-x)))


# The derivative of the transfer function, g'(x)
def logFuncDerivative(x):
    return math.exp(-x) / (pow(math.exp(-x) + 1, 2))


def randomFloat(low, high):
    return random.random() * (high - low) + low


# Initializes a matrix of all zeros
def makeMatrix(I, J):
    m = []
    for i in range(I):
        m.append([0] * J)
    return m


class NN:  # Neural Network
    def __init__(self, numInputs, numHidden, learningRate=0.001):
        # Inputs: number of input and hidden nodes. Assuming a single output node.
        # +1 for bias node: A node with a constant input of 1. Used to shift the transfer function.
        self.numInputs = numInputs + 1
        self.numHidden = numHidden

        # Current activation levels for nodes (in other words, the nodes' output value)
        self.inputActivation = [1.0] * self.numInputs
        self.hiddenActivations = [1.0] * self.numHidden
        self.outputActivation = 1.0  # Assuming a single output.
        self.learningRate = learningRate

        # create weights
        # A matrix with all weights from input layer to hidden layer
        self.weightsInput = makeMatrix(self.numInputs, self.numHidden)
        # A list with all weights from hidden layer to the single output neuron.
        self.weightsOutput = [0 for i in range(self.numHidden)]  # Assuming single output
        # set them to random vaules
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weightsInput[i][j] = randomFloat(-0.5, 0.5)
        for j in range(self.numHidden):
            self.weightsOutput[j] = randomFloat(-0.5, 0.5)

        # Data for the backpropagation step in RankNets.
        # For storing the previous activation levels (output levels) of all neurons
        self.prevInputActivations = []
        self.prevHiddenActivations = []
        self.prevOutputActivation = 0
        # For storing the previous delta in the output and hidden layer
        self.prevDeltaOutput = 0
        self.prevDeltaHidden = [0 for i in range(self.numHidden)]
        # For storing the current delta in the same layers
        self.deltaOutput = 0
        self.deltaHidden = [0 for i in range(self.numHidden)]

    def propagate(self, inputs):
        if len(inputs) != self.numInputs - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.prevInputActivations = copy.deepcopy(self.inputActivation)
        for i in range(self.numInputs - 1):
            self.inputActivation[i] = inputs[i]
        self.inputActivation[-1] = 1  # Set bias node to -1.

        # hidden activations
        self.prevHiddenActivations = copy.deepcopy(self.hiddenActivations)
        for j in range(self.numHidden):
            sum = 0.0
            for i in range(self.numInputs):
                # print self.ai[i] ," * " , self.wi[i][j]
                sum = sum + self.inputActivation[i] * self.weightsInput[i][j]
            self.hiddenActivations[j] = logFunc(sum)

        # output activations
        self.prevOutputActivation = self.outputActivation
        sum = 0.0
        for j in range(self.numHidden):
            sum = sum + self.hiddenActivations[j] * self.weightsOutput[j]
        self.outputActivation = logFunc(sum)
        return self.outputActivation

    def computeOutputDelta(self):
        p_ab = logFunc(self.prevOutputActivation - self.outputActivation)
        self.prevDeltaOutput = logFuncDerivative(self.prevOutputActivation) * (1 - p_ab)
        self.deltaOutput = logFuncDerivative(self.outputActivation) * (1 - p_ab)

    def computeHiddenDelta(self):
        # Implement the delta function for the hidden layer (see exercise text)
        for h in range(self.numHidden):
            self.prevDeltaHidden[h] = logFuncDerivative(self.prevHiddenActivations[h]) * self.weightsOutput[h] * (
            self.prevDeltaOutput - self.deltaOutput)
            self.deltaHidden[h] = logFuncDerivative(self.hiddenActivations[h]) * self.weightsOutput[h] * (
            self.prevDeltaOutput - self.deltaOutput)

    def updateWeights(self):
        # Update the weights of the network using the deltas (see exercise text)
        for i in range(len(self.weightsInput)):
            for j in range(len(self.weightsInput[i])):
                self.weightsInput[i][j] += self.learningRate * (
                    self.prevDeltaHidden[j] * self.prevInputActivations[i] - self.deltaHidden[j] * self.inputActivation[
                        i])
        for h in range(len(self.weightsOutput)):
            self.weightsOutput[h] += self.learningRate * (
                self.prevDeltaOutput * self.prevHiddenActivations[h] - self.deltaOutput * self.hiddenActivations[h])

    def backpropagate(self):
        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()

    # Prints the network weights
    def weights(self):
        print('Input weights:')
        for i in range(self.numInputs):
            print(self.weightsInput[i])
        print()
        print('Output weights:')
        print(self.weightsOutput)

    def train(self, patterns, iterations=1):
        for pair in patterns:
            self.propagate(pair[0])
            self.propagate(pair[1])
            self.backpropagate()

    def countMisorderedPairs(self, patterns):
        numRight = 0.0
        numMisses = 0.0

        for pair in patterns:
            a = self.propagate(pair[0])
            b = self.propagate(pair[1])

            """if self.prevOutputActivation > self.outputActivation:
                if pair[0].rating > pair[1].rating:
                    numRight += 1.0
                else:
                    numMisses += 1.0
            else:
                if pair[1].rating > pair[0].rating:
                    numRight += 1.0
                else:
                    numMisses += 1.0
            """

            if a > b:
                numRight += 1.0
            else:
                numMisses += 1.0

        rate_of_error = numMisses / (numRight + numMisses)
        print "Error: " + str(rate_of_error)

        return rate_of_error
