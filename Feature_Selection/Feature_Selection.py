import random
import math
from sklearn import svm
import os
import sys
import subprocess


def feature_ext(X, cols):
    Val = []
    columns = list(zip(*X))
    for i in cols:
        Val.append(columns[i])
    Val = list(zip(*Val))
    return Val

def sub_sample(dataset, labels, ratio):
    sData = []
    sLabel = []
    nSample = round(len(dataset) * ratio)

    row_index = [random.randint(0, nSample - 1) for _ in range(0, nSample)]

    for i in row_index:
        sData.append(dataset[i])
        sLabel.append(labels[i])
    return sData, sLabel



def bag(model, row):
    predictions = [list(m.predict([row])) for m in model]
    pred = [i[0] for i in predictions]
    return max(set(pred), key=pred.count)

def Accuracy(actual, predicted):
    count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            count += 1
    return count / float(len(actual)) * 100.0


def chiSquare(data, n_features):

    label = [row[-1] for row in data]
    rows = len(data)
    cols = len(data[0]) - 1
    T = []
    for j in range(0, cols):
        ct = [[1, 1], [1, 1], [1, 1]]

        for i in range(0, rows):
            if label[i] == 0:
                if data[i][j] == 0:
                    ct[0][0] += 1
                elif data[i][j] == 1:
                    ct[1][0] += 1
                elif data[i][j] == 2:
                    ct[2][0] += 1
            elif label[i] == 1:
                if data[i][j] == 0:
                    ct[0][1] += 1
                elif data[i][j] == 1:
                    ct[1][1] += 1
                elif data[i][j] == 2:
                    ct[2][1] += 1

        col_totals = [sum(x) for x in ct]
        row_totals = [sum(x) for x in zip(*ct)]
        total = sum(col_totals)
        exp_value = [[(row * col) / total for row in row_totals] for col in col_totals]
        sqr_value = [[((ct[i][j] - exp_value[i][j]) ** 2) / exp_value[i][j] for j in range(0, len(exp_value[0]))] for i in range(0, len(exp_value))]
        chi_2 = sum([sum(x) for x in zip(*sqr_value)])
        T.append(chi_2)
    indices = sorted(range(len(T)), key=T.__getitem__, reverse=True)
    idx = indices[:n_features]
    return idx
    
def usingSVM(sData, sLabel):
    model = svm.SVC(kernel='linear', C=1)
    model.fit(sData, sLabel)
    model.score(sData, sLabel)
    return model
    
print("Reading Data.........")
datafile = sys.argv[1]
f = open(datafile, 'r')
data = []
i = 0
l = f.readline()
while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f.readline()

print('Training data:', len(data))

datafile = sys.argv[2]
f = open(datafile, 'r')
label = []
i = 0
l = f.readline()
while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    label.append(l2)
    l = f.readline()

print('Training labels:', len(label))

datafile = sys.argv[3]
f = open(datafile, 'r')
testData = []
i = 0
l = f.readline()
while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    testData.append(l2)
    l = f.readline()

print('Test data:', len(testData))


trainLabels, index = zip(*label)
trainLabels = list(trainLabels)

for i in range(0, len(data)):
    data[i].append(trainLabels[i])

del (index, i, label)
print('Data loaded')


ratio = 0.70
dataLength = len(data)
size = int(dataLength * ratio)

train_index = random.sample(range(dataLength), size)

train_subset = []
test_subset = []

for i in range(len(data)):
    if i in train_index:
        train_subset.append(data[i])
    else:
        test_subset.append(data[i])
        

col_Feature = chiSquare(train_subset, 15)
real_Test_Data = feature_ext(testData, col_Feature)
col_Feature.append(len(train_subset[0]) - 1)
new_Training_Data = feature_ext(train_subset, col_Feature)
newTestData = feature_ext(test_subset, col_Feature)

print('Feature Selection done!')

new_Training_Data = [list(elem) for elem in new_Training_Data]
newTestData = [list(elem) for elem in newTestData]
real_Test_Data = [list(elem) for elem in real_Test_Data]

new_Training_Label = [row[-1] for row in new_Training_Data]
for row in new_Training_Data:
    del (row[-1])

newTestLabel = [row[-1] for row in newTestData]
for row in newTestData:
    del (row[-1])

###SVM (Bagging)####

bags = 50
models = [] * bags

for _ in range(0, bags):
    sData, sLabel = sub_sample(new_Training_Data, new_Training_Label, 1)
    # SVM linear Model
    m = usingSVM(sData, sLabel)
    models.append(m)

predictions = []
for row in newTestData:
    predictions.append(bag(models, row))

model_Accuracy = Accuracy(newTestLabel, predictions)

print('\nACCURACY OF THE MODEL IS', model_Accuracy, '%\n')

real_Test_DataPredict = [bag(models, row) for row in real_Test_Data]

file_path = os.path.dirname(os.path.abspath('__file__'))

#### The predicted labels are saved in file called testlabels #####

path = file_path + '/OutputFiles/testlabels'

f = open(path, "a")
w = 0
for i in real_Test_DataPredict:
    f.write(' '.join(map(str, [int(i)])) + ' ' + str(w) + "\n")
    w += 1
f.close()

path1 = file_path + '/OutputFiles/features'
f=open(path1, "a")
for i in range(len(col_Feature)-1):
    f.write(' '+str(col_Feature[i]))
f.close()

del (col_Feature[-1])
print('Number of features =', str(len(col_Feature)))
print('Feature columns =', str(col_Feature))

score = model_Accuracy / (100 * len(col_Feature))
#print('Score of the output =', str(score))
print('Predicted labels file Saved in: ', path)
