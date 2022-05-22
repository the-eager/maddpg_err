import logging

import numpy as np
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(message)s')
fhandler = logging.FileHandler('test_UAV.log', 'w')
fhandler.setLevel(logging.INFO)# DEBUG
fhandler.setFormatter(formatter)

chandler = logging.StreamHandler()
chandler.setLevel(logging.INFO)
chandler.setFormatter(formatter)

logger.addHandler(fhandler)
logger.addHandler(chandler)
logger.setLevel(logging.INFO)


import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
X = [-1.1, 0.2, 100.1, 0.3]
print(X)
X = np.reshape(X,(-1,1))
print(X)

clf = LOF(n_neighbors=2)
res = clf.fit_predict(X)
num = 0
for i in range(len(res)):
       if res[i] == -1:
              X[i] = np.nan
              num += 1
print(res)
print(num)
print(X)
Y = X[~np.isnan(X)]
print(Y)
print("ok")