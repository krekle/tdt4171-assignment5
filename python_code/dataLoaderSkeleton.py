import Backprop_skeleton as Bp
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# Class for holding your data - one object for each line in the dataset
class dataInstance:
    def __init__(self, qid, rating, features):
        self.qid = qid  # ID of the query
        self.rating = rating  # Rating of this site for this query
        self.features = features  # The features of this query-site pair.

    def __str__(self):
        return "Datainstance - qid: " + str(self.qid) + ". rating: " + str(self.rating) + ". features: " + str(
            self.features)


# A class that holds all the data in one of our sets (the training set or the testset)
class dataHolder:
    def __init__(self, dataset):
        self.dataset = self.loadData(dataset)

    def loadData(self, file):
        # Input: A file with the data.
        # Output: A dict mapping each query ID to the relevant documents, like this: dataset[queryID] = [dataInstance1, dataInstance2, ...]
        data = open(file)
        dataset = {}
        for line in data:
            # Extracting all the useful info from the line of data
            lineData = line.split()
            rating = int(lineData[0])
            qid = int(lineData[1].split(':')[1])
            features = []
            for elem in lineData[2:]:
                if '#docid' in elem:  # We reached a comment. Line done.
                    break
                features.append(float(elem.split(':')[1]))
            # Creating a new data instance, inserting in the dict.
            di = dataInstance(qid, rating, features)
            if qid in dataset.keys():
                dataset[qid].append(di)
            else:
                dataset[qid] = [di]
        return dataset


def runRanker(iterations, trainingset='datasets/train.txt', testset='datasets/test.txt'):
    # Dataholders for training and testset
    dhTraining = dataHolder(trainingset)
    dhTesting = dataHolder(testset)

    # Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    nn = Bp.NN(46, 10, 0.001)

    trainingPatterns = []  # For holding all the training patterns we will feed the network
    testPatterns = []  # For holding all the test patterns we will feed the network

    # Find unique patterns for testing and training
    for qid in dhTraining.dataset.keys():
        dataInstance = dhTraining.dataset[qid]
        dataInstance.sort(key=lambda obj: obj.rating, reverse=True)

        for i in xrange(len(dataInstance)):
            for j in xrange(i + 1, len(dataInstance)):
                if not (dataInstance[i].rating == dataInstance[j].rating):
                    trainingPatterns.append([dataInstance[i].features, dataInstance[j].features])

    for qid in dhTesting.dataset.keys():
        dataInstance = dhTesting.dataset[qid]
        dataInstance.sort(key=lambda obj: obj.rating, reverse=True)

        for i in xrange(len(dataInstance)):
            for j in xrange(i + 1, len(dataInstance)):
                if not (dataInstance[i].rating == dataInstance[j].rating):
                    testPatterns.append([dataInstance[i].features, dataInstance[j].features])

    test_error = list()
    training_error = list()
    print 'Training Initiated ... '
    training_error.append(nn.countMisorderedPairs(trainingPatterns))
    test_error.append(nn.countMisorderedPairs(testPatterns))

    for i in xrange(iterations):
        print '[Epoch: {iteration}]'.format(iteration=i + 1)
        # Train
        nn.train(trainingPatterns)

        # Check ann performance
        training_error.append(nn.countMisorderedPairs(trainingPatterns))
        test_error.append(nn.countMisorderedPairs(testPatterns))

    return [training_error, test_error]


def runner():
    runs = 5
    iterations = 20

    # Run first to initialize lists
    training, test = runRanker(iterations)

    # Remove one iteration form the loop for initialization
    for i in xrange(runs - 1):
        tr, te = runRanker(iterations)
        print tr
        print te

        # Plot runs
        #plt.plot(tr, "k--", label="Training: " + str(i))
        #plt.plot(te, "k-", label="Test: " + str(i))

        # Accumulate for avgs
        for itemIndex in xrange(len(tr)):
            training[itemIndex] += tr[itemIndex]
            test[itemIndex] += te[itemIndex]

    # Average the results
    for avgIndex in xrange(len(training)):
        training[avgIndex] /= runs
        test[avgIndex] /= runs

    # Plotting avg
    plt.plot(training, label="Avg Training")
    plt.plot(test, label="Avg Test")

    # Axis labels
    plt.xlabel("Error Rate")
    plt.ylabel("Epoch")

    # Show legend
    plt.legend()

    # Set y window
    plt.ylim(0, 1)

    # Show
    plt.show()


runner()
