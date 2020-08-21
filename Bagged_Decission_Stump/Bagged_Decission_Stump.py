import argparse
import sys
import os
from random import randint as rand

#this will store lists of all the predictions needed for the labels
preds = {}

def dataReader(data_file):
   
    f = open(data_file)
    data = []
    i = 0
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        data.append([])
        for j in range(len(line)):
            data[i].append(line[j])
        i += 1
    f.close()
    return data

def labelReader(labels_file):
  
    f = open(labels_file)
    label_lines = []
    
    for line in f.readlines():
        a = [int(x) for x in line.split()]
        label_lines.append(a)
       
    f.close()
    return label_lines

def classMaker(label_lines):
   
    class_d = {}
    class_size = [0,0]
    for line in label_lines:
        class_d[line[1]] = line[0]
        class_size[line[0]] = class_size[line[0]] + 1
        
    return class_d, class_size

def filterdata(data, labels):
    global preds 
    row_indeces = []  
    total_pres = 0
    nrow = len(data)
    for i in range(nrow):
        if i not in labels:
            preds[i] = {0:0,1:0}
            total_pres += 1
        else:
            row_indeces.append(i)
    return row_indeces
            
def Bagging(data, indeces, labels):
   
    nrow, ncol = len(data), len(data[0])
    new_data = []
    new_labs = {}
    cur = 0
    while(len(new_data) < len(data)):
        row_idx = indeces[rand(0,len(indeces)-1)]
        if labels.get(row_idx) == None: 
            print("Unexpected bagged data (unclassified) row {}".format(row_idx))
            continue
        new_data.append(data[row_idx])
        new_labs[cur] = labels[row_idx]
        cur += 1
    return new_data, new_labs
        
def gini_sel(data, labels):

    
    nrow, ncol = len(data), len(data[0])
    ginivals = [[0, 0] for j in range(ncol)]
    temp, c, s = 0, 0, 0


    for j in range(ncol):

        listcol = [item[j] for item in data]
        keys = sorted( range( len(listcol) ), key=lambda col: listcol[col])
        listcol = sorted(listcol)  

        ginis = []
        prevrow = 0

        for k in range(1,nrow):

            lsize, rsize = k, (nrow - k)
 
            lp, rp = 0, 0
            for l in range(k):
                if (labels.get(keys[l]) == 0):
                    lp += 1
            for r in range(k, nrow):
                if (labels.get(keys[r]) == 0):
                    rp += 1
 
            gini = float((lsize / nrow) * (lp / lsize) * (1 - lp / lsize) + (rsize / nrow) * (rp / rsize) * (1 - rp / rsize))
            ginis.append(gini)
            if (ginis[k - 1] == float(min(ginis))):
                ginivals[j][0] = ginis[k - 1]
                ginivals[j][1] = k

        if (j == 0):
 
            temp = ginivals[j][0]
        if (ginivals[j][0] <= temp):

            temp = ginivals[j][0]
            c = j
            s = ginivals[j][1]
            if (s != 0):
                s = float((listcol[s] + listcol[s - 1]) / 2)
 

    left_count, right_count = 0, 0
    left_label, right_label = 0, 0
    for i in range(nrow):
        if labels.get(i) != None:
            if data[i][c] < s: #for all points left of the split
                if labels[i] == 0: #check if more 0 or 1 labels exist
                    left_count += 1 
                else:
                    right_count += 1

    if left_count > right_count:
        right_label = 1
    else:
        left_label = 1

  #  print("gini index: {}\ncolumn with best split: {}\nbest split: {}".format(temp,c,s))
    return c, s, left_label, right_label

def tally_predictions(col, split, data, labels, left, right):
    global preds
    nrow = len(data)
    for i in range(nrow):
        point = data[i][col]
        if labels.get(i) == None:
           
            if point < split:
                preds[i][left] += 1  
            else:
                preds[i][right] += 1 

def print_predictions():
    global preds
    actual = {}
    for key in preds:
        if preds[key][0] > preds[key][1]:
            print("{} {}".format(key, 0))
            actual[key] = 0
        else:
            print("{} {}".format(key, 1))
            actual[key] = 1
    #return actual

def compare_predictions(ap, labels_path):
    f = open(labels_path)
    d = {}
    for line in f:
        l = line.split()
        d[int(l[1])] = int(l[0])
    f.close()
    num_wrong = 0
    num_correct = 0
    for key in ap:
        if ap[key] == d[key]:
            num_correct += 1
        else:
            num_wrong += 1
    print("error: {}/{} = {}".format(num_wrong, len(ap), 100 * num_wrong/len(ap)))
    
def parse_options():
    parser = argparse.ArgumentParser(description="Bagging on the HW06 Decision Stump")
    parser.add_argument("data_file", help="path to the data file")
    parser.add_argument("labels_file", help="path to the training labels file")
    parser.add_argument("--labs", help="path to the labels file")
    ret_args = parser.parse_args()
    return ret_args

if __name__ == "__main__":
    args = parse_options()
    data_filepath, labels_filepath = args.data_file, args.labels_file    
    data_content = dataReader(data_filepath)
    label_content = labelReader(labels_filepath)
    classes, class_sizes = classMaker(label_content)
    training_indeces = filterdata(data_content, classes)
    
    for i in range(101):
       # print("_______iteration:{}________".format(i))
        bag, bag_labs = Bagging(data_content, training_indeces, classes)
        best_col, best_split, leftlab, rightlab = gini_sel(bag, bag_labs)
        tally_predictions(best_col, best_split, data_content, classes, leftlab, rightlab)

    print_predictions()
   