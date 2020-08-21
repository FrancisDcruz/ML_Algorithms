import sys

# Read The Data
data_file = sys.argv[1];
label_file = sys.argv[2];
f = open(data_file);
l = f.readline();
data = [];

while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]));
    data.append(l2);
    l = f.readline();

rows = len(data);
cols = len(data[0]);
f.close();

# Read the Training Labels
train_labels = {};
n = [];
f = open(label_file);
n = [];
n.append(0);
n.append(0);
l = f.readline();


while (l != ''):
    a = l.split();
    train_labels[int(a[1])] =  int(a[0]);
    n[int(a[0])] += 1;
    l = f.readline()
f.close();

# Gini Index Implementation
g_vals = [];
split = 0;
l3 = [0,0];
for j in range(0, cols, 1):
    g_vals.append(l3);
ginit = 0;
col = 0;
for j in range(0, cols, 1):
    listcol = [item[j] for item in data]
    keys = sorted(range(len(listcol)), key = lambda k: listcol[k])
    listcol.sort();
    gv = [];
    prev_gini = 0;
    prevrow = 0;
    for k in range(1, rows, 1):
        lsize = k;
        rsize = rows - k;
        lp = 0;
        rp = 0;

        for i in range(0, k, 1):
            if (train_labels.get(keys[i]) == 0):
                lp += 1
        for m in range(k, rows, 1):
            if (train_labels.get(keys[m]) == 0):
                rp += 1
        gini = (lsize/rows)*(lp/lsize)*(1 - lp/lsize)+(rsize/rows)*(rp/rsize)*(1-rp/rsize);
        gv.append(gini);
        prev_gini = min(gv);

        if(gv[k - 1] == float(prev_gini)):
            g_vals[j][0] = gv[k - 1];
            g_vals[j][1] = k;
    if(j == 0):
        ginit = g_vals[j][0];
    if(g_vals[j][0] <= ginit):
        ginit = g_vals[j][0];
        col = j;
        split = g_vals[j][1];
        if(split != 0):
            split = (listcol[split] + listcol[split - 1]) / 2;
print("Best split for column ",col," at ",split," with gini index ",ginit)