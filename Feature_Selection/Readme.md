In this course project we encourage you to develop your own set of methods for learning and classifying. 

This is a simulated dataset of single nucleotide polymorphism (SNP) genotype datacontaining 29623 SNPs (total features).

Amongst all SNPs are 15 causalones which means they and neighboring ones discriminate between case and controls while remainder are noise.

In the training are 4000 cases and 4000 controls. Your task is to predictthe labels of 2000 test individuals whose true labels are known only tothe instructor and TA.
Both datasets and labels are immediately following the link for thisproject file. 

The training dataset is called traindata.gz (in gzippedformat), training labels are in trueclass, and test dataset is calledtestdata.gz (also in gzipped format).
You may use cross-validation to evaluate the accuracy of your method and forparameter estimation. 

The winner would have the highest accuracy in the testset with the fewest number of features.Your project must be in Python. 
You cannot use numpy or scipy. 

You may usethe support vector machine, logistic regression, naive bayes, linearregression and dimensionality reduction modules but not the featureselection ones. 
These classes are available by importing the respectivemodule. 

For example to use svm we do from sklearn import svmYou may also make system calls to external C programs for classification such as svmlight, liblinear, fest, and bmrm.
Your program would take as input the training dataset, thetrueclass label file for training points, and the test dataset.
The output would be a prediction of the labels of the test dataset in thesame format as in the class assignments. 
Also output the total number offeatures and the feature column numbers that were used for final prediction.
If all features were used just say "ALL" instead of listing all columnnumbers.The score of your output is measured by accuracy/(#number of features).
In order to qualify for full points you would need to achieve an accuracyof at least 63%.