from random import seed
from random import randrange
from csv import reader
from collections import Counter
import math
import operator
import numpy

# Load a CSV file
def load_csv(filename):
    file = open(filename, "rb")
    lines = reader(file,delimiter=';')
    headers = lines.next()
    dataset = list(lines)
    return headers, dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

#Calculate Euclidean Distance, dist((x, y), (a, b)) = sqrt(x - a)^2 + sqrt(y - b)^2
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += math.pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

#Calculate Manhattan Distance, dist((x, y), (a, b)) = | x - a | + | y - b |
def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs(instance1[x] - instance2[x])
    return  distance

#Calculate Cosine Distance, dist((x, y), (a, b)) = 1- [( ax + by ) / (sqrt( a^2 +  b^2 )*sqrt( x^2 +  y^2 )]
def CosineDistance(instance1, instance2, length):
    dot_product, magnitude_1, magnitude_2 = 0, 0, 0
    for x in range(length):
        dot_product += instance1[x]*instance2[x]
        magnitude_1 += math.pow(instance1[x],2)
        magnitude_2 += math.pow(instance2[x],2)
    distance = 1- dot_product/(math.sqrt(magnitude_1)*math.sqrt(magnitude_2))
    return distance

#get k neighbor for the test instance from training set
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    # Calcualte distance for each test instance
    for x in range(len(trainingSet)):
        # three different type of distance measures
        #dist = manhattanDistance(testInstance, trainingSet[x], length)
        #dist = CosineDistance(testInstance, trainingSet[x], length)
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    # Sort neighbors by their distances
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    # Add k nearest neighbors
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# get prediction base on it's neighbors
def getResponse(neighbors):
    predictions = {}
    # count responses for it's neighbors
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in predictions:
            predictions[response] += 1
        else:
            predictions[response] = 1
    # Sort and get the majority response from it's neighbors
    sortedPredictions = sorted(predictions.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedPredictions[0][0]


# Split the dataset into n folds, return the splited data sets
def cross_validation_split(dataset, n_folds):
    # Create a dataset to store the splited data, and a copy of original dataset
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
    # Add 1/n of dataset to each fold
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
    # Add the remaining data from the first fold to the remainder fold
	if len(dataset_copy) > 0:
		for i in range(0, len(dataset_copy)-1):
			dataset_split[i].append(dataset_copy.pop(i))
	return dataset_split

# run knn on the n folds data sets.
def cross_validation_knn(dataset, k, n_folds):
    folds = cross_validation_split(dataset, n_folds)
    # keep track test_scores, validation_scores for each fold
    test_scores, validation_scores= list(), list()
    # Use one fold as test and other three combined as training set each time.
    # Each time keep some data points aside from the training set as validation set
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list(fold)
        validation_set = list()
        while len(validation_set) < (len(train_set)/4):
            index = randrange(len(train_set))
            validation_set.append(train_set.pop(index))
        f1_vali = knn(train_set, validation_set, k)
        validation_scores.append(f1_vali)
        f1_test = knn(train_set, test_set, k)
        test_scores.append(f1_test)
    return validation_scores, test_scores

#KNN classifier
def knn(train, test, k):
    predicted, actual= [], []
    for x in range(len(test)):
        neighbors = getNeighbors(train, test[x], k)
        result = getResponse(neighbors)
        predicted.append(result)
        actual.append(test[x][-1])
    f1_Accuracy = f1_metric(actual, predicted)
    return f1_Accuracy

# Performance metrics of macro-F1 score and average accuracy
def f1_metric(actual, predicted):
    # keep track of the correct count, true_positive count, recall, and precision
    correct, true_positive, recall, precision = 0, [], [], []
    actual_count = Counter(actual)
    predicted_count = Counter(predicted)
    # Count the number of True Positive for each value.
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
            true_positive.append(actual[i])
    true_positove_counter = Counter(true_positive)
    #Calculate each recall and precision.
    for x in true_positove_counter:
        recall.append(float(true_positove_counter.get(x))/float(actual_count.get(x)))
        precision.append(float(true_positove_counter.get(x))/float(predicted_count.get(x)))
	recall_macro = sum(recall)/len(actual_count)
    precision_macro = sum(precision)/len(predicted_count)
    f1 = 2*recall_macro*precision_macro/(recall_macro+precision_macro)
    return (f1*100, correct / float(len(actual)) * 100.0)

# chage data to array by attributes
def array(dataset, headers):
	column = {}
	for h in headers:
		column[h] = []
	for row in dataset:
		for h, v in zip(headers, row):
			column[h].append(float(v))
	return column

#Normalization
def normalize(array):
	array_max = max(array)
	array_min = min(array)
	for i in range(len(array)):
		array[i] = (array[i]-array_min)/(array_max-array_min)
	return array

#Prepare the data to be normalized, then out put normalized data set
def norm(data,features):
    norm_dataset = data
    dataset = array(data, features)
    for h in range(len(features)):
        feature = features[h]
        x_array = numpy.array(dataset[feature])
        normalized_X = normalize(x_array)
        dataset[h] = normalized_X
        for i in range(len(dataset[h])):
            norm_dataset[i][h] = dataset[h][i]
    return norm_dataset

#Print result as required output format
def print_reslut(scores, k, n_folds, distance_measure):
    Average_Validation_F1, Average_Validation_Accuracy, Average_test_F1, Average_test_Accuracy= 0, 0, 0, 0
    print "Hyper-parameters:"
    print "K:", k
    print "Distance measure:", distance_measure
    print
    for i in range(n_folds):
        print 'Fold-', i + 1, ":"
        print 'Validation: F1 Score:', '{:04.1f}'.format(scores[0][i][0]),',', 'Accuracy:', '{:04.1f}'.format(
            scores[0][i][1])
        Average_Validation_F1 += scores[0][i][0]
        Average_Validation_Accuracy += scores[0][i][1]
        print 'Test: F1 Score:', '{:04.1f}'.format(scores[1][i][0]),',','Accuracy:', '{:04.1f}'.format(scores[1][i][1])
        Average_test_F1 += scores[1][i][0]
        Average_test_Accuracy += scores[1][i][1]
        print
    print 'Average:'
    print 'Validation: F1 Score:', '{:04.1f}'.format(Average_Validation_F1 / n_folds),',', 'Accuracy:', '{:04.1f}'.format(
        Average_Validation_Accuracy / n_folds)
    print 'Test: F1 Score:', '{:04.1f}'.format(Average_test_F1 / n_folds),',', 'Accuracy:', '{:04.1f}'.format(
        Average_test_Accuracy / n_folds)
    print

def main():
    #seed(123)
    # load and prepare data
    filename = 'winequality-white.csv'
    features, dataset = load_csv(filename)
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    norm_dataset = norm(dataset, features[0:-1])
    k= 5
    n_folds = 4
    distance_measure = 'Euclidean Distance'
    scores = cross_validation_knn(norm_dataset, k, n_folds)
    print_reslut(scores,k,n_folds, distance_measure)

main()

