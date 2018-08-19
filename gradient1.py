# @Author: Atul Sahay <atul>
# @Date:   2018-08-05T17:47:50+05:30
# @Email:  atulsahay01@gmail.com
# @Filename: gradient1.py
# @Last modified by:   atul
# @Last modified time: 2018-08-07T18:08:34+05:30



# In[]
import numpy as np
from matplotlib import pyplot as plt

% matplotlib inline

# In[]

x_points = [1,1,2,3,4,5,6,7,8,9,10,11]
y_points = [1,2,3,1,4,5,6,4,7,10,15,9]

# In[]

plt.plot(x_points, y_points, 'bo')
# In[]

# y = mx + b
m = 0
b = 0
y = lambda x : m*x + b

# In[]

def plot_line(y, data_points):
    x_values = [ i for i in range(int(min(data_points)-1),int(max(data_points)+2)) ]
    y_values = [ y(x) for x in x_values ]
    plt.plot(x_values, y_values, 'r')

# In[]

plot_line(y, x_points)
plt.plot(x_points, y_points, 'bo')

# In[]

learn = 0.001

# In[]

def summation(y, x_points, y_points):
    total1 = 0
    total2 = 0

    for i in range(1,len(x_points)):
        #print(total1, total2)
        total1 += (y(x_points[i]) - y_points[i])
        total2 += ((y(x_points[i]) - y_points[i])*x_points[i])

    return total1/len(x_points), total2/len(x_points)

# In[]

for i in range(1000):
    s1, s2 = summation(y, x_points, y_points)
    m = m - learn*s2
    b = b - learn*s1
    plot_line(y, x_points)
    plt.plot(x_points, y_points, 'bo')
# In[]

m

# In[]

b

# In[]

plot_line(y, x_points)
plt.plot(x_points, y_points, 'bo')

# In[]
import pandas as pd

# In[]
train_data = pd.read_csv('/home/atul/college/cs725/Assignment/train.csv')

# In[]
train_data.head()
# In[]
train_data.shape
# In[]
train_data.iloc[:,[ i for i in range(5)]].head(6)
