import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy as sp
import scipy.interpolate



import random
# set seed to reproducible
# random.seed(1)
# data_size = 100
# max_value_range = 132
# x = np.array([random.random()*max_value_range for p in range(0,data_size)])
# y = np.array([random.random()*max_value_range for p in range(0,data_size)])
# z = 2*x*x*x + np.sqrt(y)*y + random.random()
# fig = plt.figure(figsize=(10,6))
# ax = axes3d.Axes3D(fig)
# ax.scatter3D(x,y,z, c='r')


pd.set_option('display.max_row',300)
pd.set_option('display.max_column',30)
data = pd.read_csv('SuperOx GdBCO 2G HTS 0° Field Dependence.csv')
data = data.dropna()
B = data['Applied field (T)']
T = data['Temperature (K)']
Ic = data['Critical current (A)']

data2 = []
for i in range(0,len(data)):
    data2.append([int(100*B.iloc[i]),int(T.iloc[i]),int(Ic.iloc[i])])
data2 = np.array(data2)
print(data2)

x = data2[:,0]
y = data2[:,1]
z = data2[:,2]

print(f'{x}\n{y}\n{z}')
x, y, z = zip(*sorted(zip(x, y, z)))
print(f'{x}\n{y}\n{z}')

spline = sp.interpolate.Rbf(x,y,z,function='thin_plate',smooth=5, episilon=5)

x_grid = np.linspace(min(x),max(x), 4*len(x))
y_grid = np.linspace(min(y),max(y),4*len(y))
B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')

Z = spline(B1,B2)
error_perc = 5
B_ref = 1.7*100
T_ref = 50

# for i in B1[1]:
#     for j in B2[1]:
#
#         error_perc = abs(B_ref-i)/(B_ref/100) + abs(T_ref-j)/(T_ref/100)
#         print(f'error = {error_perc} %')
#
#         if error_perc <= 0.5:
#             index_B = np.where(B1[0]==i)
#             print(f'B={i} ind={index_B[0][0]} T = {B2[0][index_B[0][0]]}')
#
# print(B1)

fig = plt.figure(figsize=(10,6))
ax = axes3d.Axes3D(fig)
ax.scatter3D(x, y, z, color = 'blue')
# ax.plot_wireframe(B1, B2, Z)
ax.plot_surface(B1, B2, Z,alpha=0.15, color = 'blue')

plt.xlabel('B, 10mT')
plt.ylabel('T, K')
ax.set_zlabel('Ic, A')
plt.show()