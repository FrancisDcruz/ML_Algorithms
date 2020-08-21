from sklearn import svm
import random
import sys
import math
import numpy as np 

def d_p(x,y):
    d_p=0
    for i in range(len(x)):
        d_p+=x[i]*y[i]
    return d_p

def sign(x):
    sign =-1 if x<=0 else 1
    return sign


def getbestC(train,labels):
                
        random.seed()
        allCs = [.001, .01, .1, 1, 10, 100]
        error = {}
        for j in range(0, len(allCs), 1):
                error[allCs[j]] = 0
        rowIDs = []
        for i in range(0, len(train), 1):
                rowIDs.append(i)
        nsplits = 10
        for x in range(0,nsplits,1):        
                #### Making a random train/validation split of ratio 90:10
                newtrain = []
                newlabels = []
                validation = []
                validationlabels = []

                random.shuffle(rowIDs) #randomly reorder the row numbers      
                #print(rowIDs)

                for i in range(0, int(.9*len(rowIDs)), 1):
                        newtrain.append(train[i])
                        newlabels.append(labels[i])
                for i in range(int(.9*len(rowIDs)), len(rowIDs), 1):
                        validation.append(train[i])
                        validationlabels.append(labels[i])
                        

                #### Predict with SVM linear kernel for values of C={.001, .01, .1, 1, 10, 100} ###
                for j in range(0, len(allCs), 1):
                        C = allCs[j]
                        clf = svm.LinearSVC(C=C)
                        #newlabels=np.reshape(newlabels,(1,-1))
                        #newtrain=np.reshape(newtrain,(1,-1))
                        clf.fit(newtrain, newlabels)
                        prediction = clf.predict(validation)
                        
                        err = 0
                        for i in range(0, len(prediction), 1):
                                if(prediction[i] != validationlabels[i]):
                                        err = err + 1
                        err = err/len(validationlabels)
                        error[C]+=err
                        #print("err=",err,"C=",C,"split=",x)


        bestC = 0
        minerror=100
        keys = list(error.keys())
        for i in range(0, len(keys), 1):
                key = keys[i]
                error[key] = error[key]/nsplits
                if(error[key] < minerror):
                        minerror = error[key]
                        bestC = key
  
        #print(bestC,minerror)
        return [bestC,minerror]
        
def K_value(k):
    Z0=[]
    for i in range(len(training_data)):
        Z0.append([])
    Z1=[]
    for i in range(len(test_data)):
        Z1.append([])
    for x in range(k):
        #########Create random w vector for training data##########
        w =[]
        for j in range(0, len(training_data[0]),1):
            w.append(0)
        for j in range(0, len(w),1):
            w[j] = random.uniform(-1,1)
        X0=[]
        for i in range(len(training_data)):
            X0.append(d_p(w,training_data[i]))
        w0=-random.choice(X0)
        ####projecting training data to w###########
        for i in range(len(training_data)):
            Z0[i].append((1+sign(X0[i]+w0))/2)
        ####projecting test data to w###########
        X1=[]
        for i in range(len(test_data)):
            X1.append(d_p(w,test_data[i]))
        for i in range(len(test_data)):
            Z1[i].append((1+sign(X1[i]+w0))/2)
    return Z0,Z1

def getError(label,predict,id):
    n0=0
    n1=0
    t0=0
    t1=0
    b_err=0
    for i in range(len(id)):
        if label[id[i]]==0:
            t0+=1
            if predict[i] !=label[id[i]] :
                n0+=1
        if label[id[i]]==1:
            t1+=1
            if predict[i] !=label[id[i]] :
                n1+=1
    #print(n0,n1,t0,t1)
    b_err=0.5*(n0/t0+n1/t1)
    accuracy=(n0+n1)/(t0+t1)        
    return  b_err  


def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        x.append(rVec)
    dFile.close()
    return x
    
def getLabel(labelFile):   
    lFile = open(labelFile, 'r')  
    print(lFile)
    llabel = {}  
    for line in lFile: 
        
        row = line.split()   
        llabel[int(row[1])] = int(row[0])  
    lFile.close()  
    return llabel
    

    
def getLabelData(labelFile):   
    lFile = open(labelFile, 'r')  
    lDict = {}  
    for line in lFile: 
        
        row = line.split()   
        lDict[int(row[1])] = int(row[0])  
    lFile.close()  
    return lDict



dataFileName = sys.argv[1]
data = getFeatureData(dataFileName)
labelFileName = sys.argv[2]
trainlabels = getLabelData(labelFileName)
dataFile = sys.argv[3]
lab = getLabel(dataFile)

name_of_dataSet=dataFileName[:-5]+" split 0:"


training_data=[]
training_labels=[]
test_data=[]
test_IDs=[]

for i in range(len(data)):
    if trainlabels.get(i)!=None:
        training_data.append(data[i])
        training_labels.append(trainlabels[i])
    else:
        test_data.append(data[i])
        test_IDs.append(i)

bestc,minerror=getbestC(training_data,training_labels)
clf=svm.LinearSVC(C=bestc)
clf.fit(training_data,training_labels)
predict_ori=clf.predict(test_data)
error_ori=getError(lab,predict_ori,test_IDs)
f=open('final_output.txt','a+')
f.write(name_of_dataSet+'\n')
f.write('Original data: LinearSVC Best C='+str(bestc)+' ,  Best CV Error='+str(round(minerror*100,2))+'% ,  test error='+str(round(error_ori*100,2))+'%'+'\n')
f.write('random hyperplane data:\n')

k_list=[10,100,1000,10000]
k_scores={}
rowIDs=list(range(len(training_data)))
for k in k_list:
    k_scores[k]=0
for k in k_list:  
    z0z1=K_value(k)
    Z0=z0z1[0]
    Z1=z0z1[1]
    bestc,minerror=getbestC(Z0,training_labels)
    clf=svm.LinearSVC(C=bestc,max_iter=10000)
    clf.fit(Z0,training_labels)
    prediction=clf.predict(Z1)
    err=getError(lab,prediction,test_IDs)
    f.write('For k ='+str(k)+'\n')
    f.write('LinearSVC Best C='+str(bestc)+' ,  Best CV Error='+str(round(minerror*100,2))+'% ,  test error='+str(round(err*100,2))+'%'+'\n')   

f.close()


