(a) You should accept only the following as arguments, <data file > < training label file> <stop criterion>

     e.g. python least_squares_adaptive_eta.py <data file> <training file> 0.0001

(b) Modify the gradient descent to use an adaptive eta setting between the compute dellf (gradient) and updatew code portions during each gradient descent iteration.

 
In other words, insert the following pseudocode in each iteration of gradient-descent:

eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
bestobj = 1000000000000 # infinity 

besteta = 1

best_w = w

for k in range(0, len(eta_list), 1):

               eta = eta_list[k]

                ##insert code here for temp_w = w + eta*dellf

               ##compute  new training error "obj" for the eta using temp_w

               ##update bestobj based on this value "obj" and update corresponding best_eta, best_w

eta = best_eta

w = best_w

error = best_obj

After you have the adapative step size solutions working, obtain
the average test error of least squares and hinge on the six
datasets on the course website. For this install in your directory, the perl
script avg_test_error.plPreview the document. This script also needs the perl script error.plPreview the document in the same directory.

To get average error for a data set using your classifier, do the following:

perl avg_test_error.pl <your classifier file>  <directory for UCI data repository> <data name>

e.g. in your directory in AFS unix,

perl   avg_test_error.pl  least_squares_adaptive_eta.py   ../ta.dir/DataSets   ionosphere

