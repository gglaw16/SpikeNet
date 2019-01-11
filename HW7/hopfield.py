
from math import *
from matplotlib import *
from numpy import *
from pylab import *
from random import *

#graphic functions to create the images I used



def circle(radius, x, y, value):
    t=linspace(0,2*pi,1000);
    a=cos(t);
    b=sin(t);
    if value == 1.0:
        fill(x+radius*a,y+radius*b,'b')
    if value == -1.0:
        fill(x+radius*a,y+radius*b,'w')
    if value == 2.0:
        fill(x+radius*a,y+radius*b,'r')
    if value == 3.0:
        fill(x+radius*a,y+radius*b,'g')

        
#draw a network of neurons and their
def net (dim1, values):
    k = 0
    for i in range (dim1):
        for j in ran
        circle (0.4, i, j, values[k])
        k=k+1

        
# one iteration -- impose activities (activity vector) and calculate new output based on
# synaptic weight matrix (weights), return a new vector of activities (a)
def recall (activity, weights):
    outt = dot (weights, activity)
    a=outt
    for n in range(len(activity)):
        if outt[n] >= 0.0:
            a[n] = 1.0
        if outt[n] < 0.0:
            a[n] = -1.0
        if outt[n] == 0.0:
            a[n]=outt[n]
    return (a)
                            
                
# calculate energey for a given activity state given the synaptic weight matrix:
# calculate voltages first, then calculate energy from voltages and weights. return energy
def ener (activity, weights):
    e=0
    volt=dot(activity, weights)
    for k in range (49):
        for n in range (49):
            e = e+weights[n,k]*volt[n]*volt[k]
    e=-0.5*e
    return (e)


# update neuron n, calculate its new activity value from a vector of activity values
# and a weight matrix. Return activity      
def updateone (n, activity, weights):
    a=0.0
    for k in range(len(activity)):
        a = a+weights[n,k]*activity[k]
    if a >= 0.0:
        act = 1.0;
    if a < 0.0:
        act = -1.0
    if a == 0.0:
        act = activity[n]
    return (act)

                    
# this function runs the simulation I showed. It loads a vector of activities (grim or grim2),
# calculates weights, updates network and plots. You probably wont need most of this
def main ():  
    #load activity patterns from file
    In1 = loadtxt('grim.txt')
    In2 = loadtxt('grim2.txt')
    test1 = In1;
    #calculate weights to store pattern In1
    w=1/49.0*outer(In1, In1)
    #plot pattern In1
    figure (1)
    subplot (1,4,1)
    net (7, In1)

    
#now create a distorted version of In1 by randomly choosing 20% of neurons to be 'flipped'
for n in range (49):
    r=random()
    if 0.2 > r:
        if test1[n] == 1:
            test1[n] = -1
        if test1[n] == -1:
            test1[n] = 1

    #plot the distorted pattern twice        
    subplot (1,4,2)
    net (7, test1)
    subplot (1,4,3)
    net (7, test1)
    pause (1)
    rec=test1

    # now use the distorted pattern as starting point and update one neuron at a time given
    # the previously calculated synaptic weights and calculate the corresponding energy values.          
    e=np.zeros(49)
    for n in range (49):
        rec[n]=updateone (n, rec,w)
        e[n]=ener(rec, w)
                    
                    
    #plot the resulting activity pattern and the energy function over time
    figure (1)
    subplot (1,4,3)
    net(7,rec)
    subplot (1,4,4)
    plot (e)
                
    #next the non-smiley face is used to initialize the network and again each neuron is updated one at a time
    In1 = loadtxt('grim.txt')
    test1 = loadtxt('grim2.txt')
                
    figure (2)
    subplot (1,4,1)
    net (7, In1)
    subplot (1,4,2)
    net (7, test1)
                
                
