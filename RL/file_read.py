import numpy as np

lastpath1 =""
file1 = open(lastpath1, 'r')
line = file1.readline()
data_list1 = []

while line:
    num = list(map(float, line.split()))
    data_list1.append(num)
    line = file1.readline()
file1.close()
data_array1 = np.array(data_list1)
print(data_array1)




