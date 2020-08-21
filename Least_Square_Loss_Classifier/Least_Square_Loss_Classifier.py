import sys
import random
import math



#Retrival of feature data from file as matrix
def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        rVec=[]
        row = line.split()
        for item in row:
            rVec.append(float(item))
        rVec.append(float(1))
        x.append(rVec)
    dFile.close()
    return x

#Retrival label data from file as a dictionary with key
def getLabelData(labelFile):   
    lFile = open(labelFile, 'r')  
    lDict = {}  
    for line in lFile: 
        row = line.split() 
        lDict[row[1]] = int(row[0]) 
        if(lDict[row[1]]==0):
            lDict[row[1]]=int(-1)
    lFile.close()  
    return lDict



dataFileName = sys.argv[1]
data = getFeatureData(dataFileName)
labelFileName = sys.argv[2]
lDict = getLabelData(labelFileName)

rows=(len(data))
cols=(len(data[0]))

w = []
for j in range(cols):
    w.append(0.02 * random.random() - 0.01)
    
#calculation of dot product
def dot_product(a, b):
    dtp = 0
    for i in range(cols):
        dtp += a[i] * b[i]
    return dtp
    
#Gradient decent
eta = 0.0001
error = rows + 10
diff = 1
count = 0
while ((diff) > 0.001):
    d = []
    for m in range(cols):
        d.append(0)
    for j in range(rows):
        if (lDict.get(str(j)) != None):
            dtp = dot_product(w, data[j])
            for k in range(cols):
                d[k] += (lDict.get(str(j)) - dtp) * data[j][k]
                

#Updating weight w
    for j in range(cols):
        w[j] = w[j] + eta * d[j]

    prev_error = error
    error = 0


#Error value
    for j in range(rows):
        if (lDict.get(str(j)) != None):
            error += (lDict.get(str(j)) - dot_product(w, data[j])) ** 2
    if (prev_error > error):
        diff = prev_error - error
    else:
        diff = error - prev_error
    count = count + 1
    if (count % 100 == 0):
        print(error)

print("error = " + str(error))

norm = 0
for i in range((cols - 1)):
    norm += w[i] ** 2
    


norm = math.sqrt(norm)

dist_origin = abs(w[len(w) - 1] / norm)
print ("Distance to  the origin = " + str(dist_origin))

#Prediction  vaoflues
for i in range(rows):
    if (lDict.get(str(i)) == None):
        dtp = dot_product(w, data[i])
        if (dtp > 0):
            print("1 " + str(i))
        else:
            print("0 " + str(i))