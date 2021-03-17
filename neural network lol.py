#!/usr/bin/env python
# coding: utf-8

# In[6]:


print("helloworld")


# In[7]:


x = list(range(1,11))

for i in range(0,10):
    x[i] = x[i]**2
    print(x[i])


# In[3]:


for n in range(1,11):
    print(n)
    pass
    n+=1000
print(n)
print("ready")


# In[5]:


print("square of", n, "is", n**2)
print("square root of", n, "is", n**0.5)


# In[11]:


def medium(x,y,z):
    print( (x+y+z)//3 )
medium(int(input()),int(input()),int(input()))


# In[114]:


import numpy
import scipy.special
import matplotlib.pyplot
import random

a = numpy.zeros([10, 10])
for i in range(10):
    for j in range(10):
        if i%2==0:
            if j%2==0:
                a[i,j]=1

matplotlib.pyplot.imshow(a,interpolation="lanczos")

print(a)

class Dog():
    def __init__(self,petname,pettemp):
        self.name = petname
        self.temp = pettemp
        
    def bark(self):
        print("гау гау")
        
    def status(self):
            print("name:", self.name)
            print("temp:",self.temp)
            
    def settemp(self,newtemp):
        self.temp = newtemp
        
x = Dog("Mike", 37)
y = Dog("stas", 10)
x.bark()
x.status()
x.settemp(10)
x.status()
y.bark()
y.status()


# In[ ]:





# In[58]:


field = list(numpy.zeros[0,8])


# In[120]:


class Neuralnetwork:
    def __init__(self,innode,hidnode,outnode,learnrate):
        self.inp_n=innode
        self.hid_n=hidnode
        self.out_n=outnode
        self.l_r=learnrate
        
        #генерация матриц слоев
        #weights input-hidden
        self.wih = (numpy.random.normal(0, pow(self.hid_n, -0.5),(self.hid_n,self.inp_n)) )
        #weights hidden-output
        self.who = (numpy.random.normal(0, pow(self.hid_n, -0.5),(self.out_n,self.hid_n)) )
        
    #тренировка сети                      
    def train(self):
        pass
    #опрос сети
    def query(self):
        self.activ_func = lambda x: scipy.special.expit(x)
        #outputs
        self.out_i = numpy.dot(self.who,inputs)
        out_o=self.activ_func(out_i)
        
        self.out_i = numpy.dot(self.who,inputs)
        out_o=self.activ_func(out_i)
        pass
inpn=3
hidn=3
outn=3
lr=0.3

nn1 = Neuralnetwork(inpn,hidn,outn,lr)

print(nn1.wih)
print()
print(nn1.who)


# In[ ]:




