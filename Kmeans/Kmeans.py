import sys;
import math;
import random as rnd;
from collections import defaultdict

def distance(x, K):
    s = sum([(m-n)**2 for m,n in zip(x,K)])
    s_sqrt = math.sqrt(s);
    return s

def fcluster(x, K):
    d = defaultdict(list);
    m = [];
    for i in range(0, len(x)):
        temp = [];
        for j in range(0, len(K)):
            temp.append(distance(x[i],K[j]));
        index = temp.index(min(temp));
        d[index].append(x[i])
    for k in d:
        temp = d[k]
        m.append([sum(n)/len(n) for n in zip(*temp)])
    return m

def hs(O,N):
    return(set([tuple(a) for a in O]) == set([tuple(a) for a in N]))

def kmean(X,K):
    m = rnd.sample(X,K);
    while True:
        mn = fcluster(X, m);
        if hs(m,mn):
            m = mn
            break
        m = mn;
    
    for i in range(0, len(X)):
        temp = [];
        for j in range(0, len(m)):
            temp.append(distance(X[i],m[j]))
        index = temp.index(min(temp));
        print('{} {}'.format(index, i));
    return m


datafile = sys.argv[1];
noCluster = int(sys.argv[2]);
f = open(datafile, 'r');
data = [];
l = f.readline();
#Read the Data File
while(l != ''):
    a = l.split();
    l2 = [];
    for j in range(0, len(a), 1):
        l2.append(float(a[j]));
    data.append(l2);
    l = f.readline();
rows = len(data);
cols = len(data[0]);
f.close();
m = kmean(data, noCluster);
