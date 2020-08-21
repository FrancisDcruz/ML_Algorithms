import sys
import random
import math

eta_list = [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001,0.0000000001,0.00000000001]
theta = 0.001

#loading test data file

input_file = sys.argv[1]
features = []
for line in open(input_file):
    line = line.rstrip()
    test_data = line.split(" ")
    aa = []
    aa.append(float(1))
    for td in test_data:
        aa.append(float(td))
    features.append(aa)


#loading training labels

output_file = sys.argv[2]
test_label = dict()
for line in open(output_file):
    line= line.rstrip()
    labels = line.split(" ")
    test_label[int(labels[1])] = int(labels[0])
    if(test_label[int(labels[1])] == 0):
        test_label[int(labels[1])] = -1

theta=float(sys.argv[3])
len_data = len(features)
num_features = len(features[0])
num_tlabel= len(test_label)

#random number generator to initilaize values for w
w = []
for j in range(num_features):
    w.append(random.uniform(-0.1, 0.1))


#computing hinge loss
def get_hingeloss():
    hingeLoss = 0
    for i in range(0,len_data):
        if(test_label.get(i) != None):
            sum = 0
            for j in range(0,num_features):
                sum = sum + (w[j] * features[i][j])
            temp = max(0,1-(test_label[i] * sum))
            hingeLoss += temp
    return hingeLoss

def get_hingeloss1(w_temp):
    hingeLoss = 0
    for i in range(0,len_data):
        if(test_label.get(i) != None):
            sum = 0
            for j in range(0,num_features):
                sum = sum + (w_temp[j] * features[i][j])
            temp = max(0,1-(test_label[i] * sum))
            hingeLoss += temp
    return hingeLoss

#computing weight
def get_weight():
    x = []
    for i in range(0, len_data):
        x.append(0)
    for i in range (0, len_data):
        for j in range(0, num_features):
            x[i] = x[i] + (features[i][j] * w[j])
    for i in range(0, num_tlabel):
        if (test_label.get(i) != None):
            x[i] = x[i] * test_label[i]

    delw = []
    for i in range(0,len_data):
        delw.append(0)
    for i in range(0,num_tlabel):
        for j in range(0,num_features):
            if(x[i] < 1):
                if test_label.get(i) != None:
                    delw[j] =delw[j] - (features[i][j] * test_label[i])
            else:
                delw[j] = delw[j] +0
    bestobj = 1000000000
    for i in range(0,len(eta_list)):
                   eta_temp=eta_list[i]
                   w_temp = w[:]
                   for i in range(0,num_features):
                       w_temp[i] = w_temp[i] - (eta_temp * delw[i])
                   obj = get_hingeloss1(w_temp)
                   if(obj < bestobj):
                       bestobj = obj
                       best_eta = eta_temp
    eta = best_eta
    #update w
    for i in range(num_features):
        w[i] = w[i] - (eta * delw[i])

prevError = get_hingeloss()
while(True):
    get_weight()
    hingeloss = get_hingeloss()
    if(abs(prevError - hingeloss) < theta):
       break
    prevError = hingeloss


#predicting training labels
for i in range(0,len_data):
    p = 0
    if (test_label.get(i) == None):
        for j in range(0,num_features):
            p = p + (w[j] * features[i][j])
        if(p < 0):
            print("0", i)
        else:
            print("1", i)
