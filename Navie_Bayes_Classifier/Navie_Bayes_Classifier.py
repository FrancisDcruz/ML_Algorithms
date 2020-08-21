import sys


'''Get feature data from file as a matrix with a row per data instance'''
def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        x.append(rVec)
    dFile.close()
    return x

'''Get label data from file as a dictionary with key as data instance indexand value as the class index'''
def getLabelData(labelFile):   
    lFile = open(labelFile, 'r')  
    print(lFile)
    lDict = {}  
    for line in lFile: 
        
        row = line.split()   
        lDict[int(row[1])] = int(row[0])  
    lFile.close()  
    return lDict


dataFileName = sys.argv[1]
data = getFeatureData(dataFileName)
labelFileName = sys.argv[2]
lDict = getLabelData(labelFileName)



key0=[]
key1=[]
testdata=[]
allkeys=[]
pre_count=[]
#getting the training data for label 0  and label 1
for key,value in lDict.items(): 
    if(value==0):
        key0.append(data[key])
        allkeys.append(key)
    elif(value==1):
        key1.append(data[key])
        allkeys.append(key)
 

#Getting the test data        
for i in range(len(data)):
    
    if( i not in allkeys):
        testdata.append(data[i])
        pre_count.append(i)
 
  
sum_0=[]
sum_1=[]
mean_0=[]
mean_1=[]
sd_0=[]
sd_1=[]

#Adding all the numbers in the list
sum_0=[sum(i) for i in zip(*key0)]
sum_1=[sum(i) for i in zip(*key1)]


#Taking the mean of class 0 and 1
mean_0=[i/len(key0) for i in sum_0]
mean_1=[i/len(key1) for i in sum_1]
length_0=len(key0)


for i in range(len(key0[0])):
    sd=[(j[i]-mean_0[i])**2 for j in key0]
    var=sum(sd)/len(key0)
    if var == 0:
        var = 1
        sd = var ** 0.5
        sd_0.insert(len(sd_0), sd)
    else:
        sd = var ** 0.5
        sd_0.insert(len(sd_0), sd)



for i in range(len(key1[0])):
    sd=[(j[i]-mean_1[i])**2 for j in key1]
    var=sum(sd)/len(key1)
    if var == 0:
        var = 1
        sd = var ** 0.5
        sd_1.insert(len(sd_1), sd)
    else:
        sd = var ** 0.5
        sd_1.insert(len(sd_1), sd)
        

pre_labels={}
nv_0=[]
nv_1=[]
for var in testdata:
    n0=0
    n1=0
    for c in range(len(testdata[0])):
        n0+=((var[c]-mean_0[c])/sd_0[c])**2
        n1+=((var[c]-mean_1[c])/sd_1[c])**2
    nv_0.insert((len(nv_0)),n0)
    nv_1.insert((len(nv_1)),n1)

i=0   
for val in pre_count:
    if(nv_0[i] < nv_1[i]):
        pre_labels[val]=0
    else:
        pre_labels[val]=1
    i=i+1  

for key,value in pre_labels.items():
    print(value," ",key)    
       

