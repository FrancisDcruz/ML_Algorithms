import sys
import random
import math

eta_list = [1,.1,.01,.001,.0001,.00001,.000001,.0000001,.00000001,.000000001,.0000000001,.00000000001]
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

#computing least squares error
def get_error():
    sum =[]
    error = 0
    for i in range(0,len_data):
            sum1 = 0
            for j in range(0,num_features):
                sum1 = sum1 + (w[j] * features[i][j])
            sum.append(sum1)
    for i in range(0,num_tlabel):
        if (test_label.get(i) != None):
            error = error + (test_label[i] - sum[i]) **2
    return error

def get_error1(w_temp):
    sum =[]
    error = 0
    for i in range(0,len_data):
            sum1 = 0
            for j in range(0,num_features):
                sum1 = sum1 + (w_temp[j] * features[i][j])
            sum.append(sum1)
    for i in range(0,num_tlabel):
        if (test_label.get(i) != None):
            error = error + (test_label[i] - sum[i]) **2
    return error

#computing weight
def get_weight():
    delw = []
    for i in range(0,num_features):
        delw.append(0)
    for i in range(0,len_data):
        if (test_label.get(i) != None):
            sum = 0
            for j in range(0,num_features):
                sum = sum + (w[j] * features[i][j])
            temp = test_label[i] - sum
            for j in range(0, num_features):
                delw[j] = delw[j] + (temp*features[i][j])
    bestobj = 1000000000
    for i in range(0,len(eta_list)):
                   eta_temp=eta_list[i]
                   w_temp = w[:]
                   for i in range(0,num_features):
                       w_temp[i] = w_temp[i] + (eta_temp * delw[i])
                   obj = get_error1(w_temp)
                   if(obj < bestobj):
                       bestobj = obj
                       best_eta = eta_temp
    eta = best_eta
    #update w
    for i in range(num_features):
        w[i] = w[i] + (eta * delw[i])

prevError = get_error()
while(True):
    get_weight()
    error = get_error()
    if((prevError - error) <= theta):
        break
    prevError = error


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
