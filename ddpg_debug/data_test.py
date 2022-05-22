<<<<<<< HEAD
import numpy as np
import sklearn
from sklearn.neighbors import LocalOutlierFactor as LOF 
import re



f = open('r.txt','w')
fener = open('ener.txt','w')
f_lof = open('r_lof.txt','w')

rew_energy_all = []
for str1 in open("multi_UAV.log"):  
    
    if re.findall('(?<=rew_energy:).*$', str1) != []:
          f.write(re.findall('(?<=rew_energy:).*$', str1)[0]+'\n')
    if re.findall('(?<=rew_energy:).*$', str1) != []:
        rew_energy_all.append([float(re.findall('(?<=rew_energy:).*$', str1)[0])])
rew_energy_all = [i for j in range(len(rew_energy_all)) for i in rew_energy_all[j]]

import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
rew_energy_all = np.reshape(rew_energy_all,(-1,1))

clf = LOF(n_neighbors=2)
res = clf.fit_predict(rew_energy_all)
num = 0
print("begin")

for i in range(len(res)):
       if res[i] == -1:
              rew_energy_all[i] = np.nan
              num += 1

rew_energy_all_lof = rew_energy_all[~np.isnan(rew_energy_all)]
str_lof = str(rew_energy_all_lof)
f_lof.write(str_lof)


mean = np.mean(rew_energy_all_lof)
var = np.var(rew_energy_all_lof)
fener.write(f"{mean}+{var}\n")
print(mean,var)
print(num)

=======
import numpy as np
import sklearn
from sklearn.neighbors import LocalOutlierFactor as LOF 
import re



f = open('r.txt','w')
fener = open('ener.txt','w')
rew_energy_all = []
for str in open("multi_UAV.log"):  
    
    if re.findall('(?<=env-_get_reward-r:).*$', str) != []:
          f.write(re.findall('(?<=env-_get_reward-r:).*$', str)[0]+'\n')
    if re.findall('(?<=rew_energy:).*$', str) != []:
        rew_energy_all.append([float(re.findall('(?<=rew_energy:).*$', str)[0])])
rew_energy_all = [i for j in range(len(rew_energy_all)) for i in rew_energy_all[j]]
mean = np.mean(rew_energy_all)

var = np.var(rew_energy_all)
fener.write(f"{mean}+{var}\n")
print(mean,var)

>>>>>>> d5972d95f410bcede2e240f9ef5a6dc44ac9c4bc
        