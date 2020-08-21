Write a Python program that implements the Naive Bayes classifier.
Your program should take as input a dataset file name(name with extension .data) and file with a set of training labels (name with extension .trainlabels.x) in the format given in the UCI datasets.

It should be run  from the command line as follows:

python <your python program file> <data file name> <training file name>

Sample files are given below. Your output for this sample should be

1 10

0 11

testBayes.data

testBayes.trainlabels.0

 

Note that the feature values in the data file should be read as real numbers.

As output your program should produce predicted labels for the test
dataset which are feature vectors whose labels are not given for training.
Each output line should be in the same format as label file, that is, <label> <rowId>

Note rowId starts from 0.

Produce output using println().

