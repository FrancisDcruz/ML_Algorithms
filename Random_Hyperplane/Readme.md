In this project we will experiment with random hyperplanes and stacking
for classification. Your program will take a dataset as input and produce new
features following the procedure below. The input is in the same format as for
previous assignments.
Inputs:
n × m training data matrix X and p × m test data matrix X1. Here n and p
are number of rows and m is number of columns (attributes)
Training labels: Y
Level : k
Let Z and Z1 be an empty list initially.
For i = 0 to k do the following:
(a) Create random vector w where each wj
, 1 ≤ j ≤ m is uniformly sampled
between -1 and 1.
(b) Determine the largest and smallest w
T xj across all training data rows xj
in
X, 1 ≤ j ≤ n. Select w0 randomly in the range [minj w
T xj
, maxj w
T xj
].
(c) Project training data X onto w. Let projection vector zi be Xw + w0 ∗ 1n
where 1n is n × 1 vector of all 1’s. (Note X is n × m matrix and w is
m × 1 vector). Append (1n + sign(zi))/2 as new column to the right end
of Z. Remember that zi
is a n × 1 vector and so for each row zki of zi
,
(1 + sign(zki))/2 is 0 if zki < 0 and 1 otherwise.
(d) Project test data X1 (each row is datapoint x1
j
) onto w. Let projection
vector z
1
i be X1w. Append z
1
i
as new column to the right end of Z1.
1. Run linear SVM on Z and predict on Z1.
2. Repeat the procedure for values of k = 10, 100, 1000, and 10000.
3. How does the error compare to liblinear on original data X and X1 for
each k?
