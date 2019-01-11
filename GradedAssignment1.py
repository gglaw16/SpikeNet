# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 00:04:18 2018

@author: gwenda
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from neuronpy.graphics import spikeplot

"""
TODO:


  
  
- Add an arugment "end_time" to the controllers "run" method.  In your loop (before the pop)
  peek at the next events time  self.input_queue[0]['time'] > end_time
  If it is past the end time,  return from the method

+ Make a network:  Make a controller, make 7 sensors with the same selective orientation.  Give them all differnt stimulus orientations.
  make a spike_recorder, add it as an output to all 7 sensors.
  call run on the controller, call plot_recordings on the spike recorder.
  
  :)





- Done --- add a method "plot_recordings" to the "spike_recorder" class.  It should make a plot similar to the one you used for your original homework.

- Done --- add funcionality to the sensor.process_event method: it should call process_event on all outputs in the sensors outputs list.
  Dad comment: Show not have created a new method with the same name. I combined the two methods.

- Done --- add an ivar "outputs" to class sensor.  It will be an empty list [] at first.
  add a method "add_output(output)" to the sensor class,  it will append the output to the self.outputs list.

- Done --- add a method "process_event(neuron)" to the recorder class.  It will add the events "time" to the appropriate recording.
  If this is confusing, I will clarify more.

- Done --- Comment on your implementation: Dictionaries do not have an append method.  You have to say self.recordings[neuron] = []
-  make a class called "spike_recorder"  it will have a ivar recordings
  which is initially an empty dictionary {}, but when full, will have structure {neuron:[], neuron2:[]}
  recorder will have a method add_neuron(neuron), which will add a new pair neuron:[]  to the dictionary


- Done --- Comment on your implementation: add a run method (TO THE CONTROLLER). This is the main loop for all neurons. 
  It needs to be a while len(queue) > 0   because processing events will cause n oe events to be added
- add a method that takes events off the input queue one at a time (event = queue.pop(0))
  and calls event['neuron'].process_event(event).  It returns when the queue is empty.

- Done ---add a "process_event(event)" method to the sensor
    It should add another event (at spike_time = event['time'] + (1/rate) ) to the controllers input queue

- done --- Since we cannot control the order input events are added,
  have "add_input_event" sort the queue after an append.

- done --- Add a start_time argument to "set_stimulus_orientation" 
  Have the "set_stimulus_orientaiton" method add an event (spike) to the controllers input queue.
  An event should look like {'neuron':sensor, 'time': spike_time}
  Choose a spike time randomly in the interval [start_time, start_time+(1/rate)]

- done --- Add an instance varaiable to sensor "max_rate"
  Add a method to the sensor "set_stimulus orientation(angle)"
  it should compute the rate  (use your gaussian function)

- done ---- Write methods "" and "" to the controller class.  They should just append the event to the queues.

- Done ---- Make a "controller" class that has two queues:
  self.input_queue = []
  self.interernal_queue = []

- done -----make a "sensor" class that has ivars
  the constructor "__init__" should take a controller as an arguemnt
"""







#object that manages the event queues.

class controller():
    """
    These queues hold events that need to be processed.
    The reason we need two queues is that events in the input queue can
    be added in any order and we need to keep it sorted.
    Since events are added to the interal queue in order, we do not
    need to sort it.
    """
    def __init__(self):
        self.input_queue = []
        self.internal_queue = []
    
    def add_input_event(event):
        self.input_queue.append(event)
        self.input_queue = sorted(self.input_queue.iteritems(), key=lambda (k,v): (v,k))
        
    def add_invernal_event(event):
        self.internal_queue.append(event)
    
    #this is the main loop for all neurons
    def run():
        while len(self.input_queue) > 0:
            if self.input_queue[0]['time'] > end_time:
                return
            event = self.input_queue.pop(0)
            event['neuron'].process_event(event)
            
    
class sensor():
    
    def __init__(self,controller):
        self.controller = controller
        self.best_orientation = 0 #the angle at which it fires the most
        self.stimulus_orientation = None
        self.rate = 0 #in Hz
        self.max_rate = 60 #in Hz
        self.selectivity_width = 45 #the standard deviation of the gaussian
        self.outputs = [] 

    #this sets the stimulus orientation and the rate based on the angle
    def set_stimulus orientation(angle, start_time):
        self.stimulus_orientation = angle
        self.rate = gaussian(angle,self.best_orientation,self.selectivity_width)
        spike_time = random.random(start_time, start_time+(1/self.rate))
        self.controller.add_input_event({'neuron':self, 'time': spike_time})
        
    def process_event(event):
        spike_time = event['time'] + (1/self.rate) 
        self.controller.add_input_event({'neuron':self, 'time': spike_time})
        # This is ok for now (as long as the outputs are only recorders),
        # but we will probably have to rethink this when outputs are synapses (with delays)
        for event in self.outputs:
            process_event(event)


    def add_output(output):
        self.outputs.append(output)
                       

class spike_recorder():
                       
    def __init__(self):
        self.recordings = {} #initially an empty dictionary, but when full, will have structure {neuron:[], neuron2:[]}

    #adds a neuron pair to the dictionary
    def add_neuron(neuron):
        self.recordings[neuron] = []

    #add the events "time" to the appropriate recording
    def process_event(neuron):
        self.recordings[neuron].append(time)
    
    #returns the recordings in the format of a list of lists
    def get_spiketrains():
        #spiketrains should be all the times of the spikes in a list separated by neuron, a list of lists
        spiketrains = [] 
        for neuron in self.recordings:
            times = []
            for time in self.recordings[neuron]:
                times.append(time)
            spiketrains.append(times)
        return spiketrains
    
    def plot_recordings():
        #spiketrains should be all the times of the spikes in a list separated by neuron, a list of lists
        spiketrains = self.get_spiketrains()
  
                       
        sp = spikeplot.SpikePlot()
        sp.set_markerscale(0.5)

        sp.plot_spikes(spiketrains)


#====================================================================================
# old code








#a
"""
I used gaussians to generate the frequency becuase I thought it would be interesting
I also used spikeplot from neuronpy.graphics to plot some very pretty spike trains, so I wrote
a function to generate the spikes in the format that it needs
"""

#this is just a gaussian function, the neurons use this to give an output frequency for a degree
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
def sensNeuron0(location):
    return gaussian(location,0,45)
    
def sensNeuron90(location):
    return gaussian(location,90,45)
    
def sensNeuron_90(location):
    return gaussian(location,-90,45)
    
def sensNeuron180(location):
    if (location > 0):
        return gaussian(location,180,45)
    else :
        return gaussian(location,-180,45)

#takes the frequency and returns spikes in a format for neuronpy spikeplot
def returnSpikes(freq,n=1):
    spikes = []
    idx = 0
    while idx < n:
        spikes.append(idx)
        idx = idx+((1/freq)/100)
            
    return spikes

#creating 6 spike trains for the 90 degree neuron 
spiketrains = [returnSpikes(sensNeuron90(-90)),
              returnSpikes(sensNeuron90(-30)),
              returnSpikes(sensNeuron90(60)),
              returnSpikes(sensNeuron90(90)),
              returnSpikes(sensNeuron90(120)),
              returnSpikes(sensNeuron90(150)),
              returnSpikes(sensNeuron90(180))]
print("Spike trains for 90 degree neuron at -90, -30, 60, 90, 120, 150, and 180 degrees (from bottom to top):")
#plot these three spike trains!
sp = spikeplot.SpikePlot()
sp.set_markerscale(0.5)

sp.plot_spikes(spiketrains)

#b
"""
This is a simple calculation of population vector by adding all the vectors, then reverse tangent to get angle
"""

#this calculates the angle from the 4 frequencies of the sensory neurons
def calcPopVector(rate0,rate90,rate_90,rate180):
    vector = [rate0-rate180,rate90-rate_90]
    if vector[0] == 0:
        if vector[1] > 0:
            return 90
        else:
            return -90
    guess = math.degrees(math.atan(vector[1]/vector[0]))
    if vector[0] < 0:
        if vector[1] > 0:
            guess = guess+180
        else:
            guess = guess-180
    return guess

#this uses calcPopVector except inputs the original angle
def popVectorFromAngle(inputAngle):
    rate0 = sensNeuron0(inputAngle)
    rate90 = sensNeuron90(inputAngle)
    rate_90 = sensNeuron_90(inputAngle)
    rate180 = sensNeuron180(inputAngle)
    return calcPopVector(rate0,rate90,rate_90,rate180)

print("population vector estimates")
print("-30 degrees:")
print(popVectorFromAngle(-30))
print("60 degrees:")
print(popVectorFromAngle(60))
print("120 degrees:")
print(popVectorFromAngle(120))
print("both 90 and -90 degrees:")
print(calcPopVector(1,0,0,.8))
#c
"""
I wrote a few functions that process all of the inputs and create a singular input for the RUN_LIF function
The 12 neruons are defined as weights for the 4 inputs
The graphs could probably be better but they all peak at their assigned degrees
"""

#this function takes the spike train data from part a and converts it to a list of voltages
def spikesToInput(spikes,amp):
   duration = 1
   sample_step = .001
   out = np.zeros(int(duration / sample_step))

   for spike_time in spikes:
       idx = int(round(spike_time / sample_step))
       if idx >=0 and idx < len(out):
           out[idx] += amp

   return out
#puts together an input y of voltages from all of the neurons for the RUN_LIF function to use
def getInput(degree,weights):
    n_90 = 5*spikesToInput(returnSpikes(sensNeuron_90(degree),1),weights[0])
    n0 = 5*spikesToInput(returnSpikes(sensNeuron0(degree),1),weights[1])
    n90 = 5*spikesToInput(returnSpikes(sensNeuron90(degree),1),weights[2])
    n180 = 5*spikesToInput(returnSpikes(sensNeuron180(degree),1),weights[3])
    return np.array(n_90)+np.array(n0)+np.array(n90)+np.array(n180)


#standard LIF neuron for 1 timestep
def LIF (i, oldv, tau, thresh):
    sum=(1-math.exp(-1.0/tau))*i+math.exp(-1.0/tau)*oldv
    if sum >= thresh:
        return ([0,5])
    else:
        return ([sum, 0])
        
#runs the LIF neuron for n time steps and returns voltage and spike train output
def RUN_LIF (y, tau=20, thresh=.3):
    n=len(y)
    x=np.zeros(n)
    v=np.zeros(n)
    
    i=1
    while i<n:
        [v[i],x[i]]=LIF(y[i],v[i-1],tau, thresh)
        i=i+1
    return([v,x])

#these are weights for all the 12 neurons
# 0:-90, 1:0, 2:90, 3:180
neuron_150 = [.33,0,0,.67]
neuron_120 = [.67,0,0,.33]
neuron_90 = [1,0,0,0]
neuron_60 = [.67,.33,0,0]
neuron_30 = [.33,.67,0,0]
neuron0 = [0,1,0,0]
neuron30 = [0,.67,.33,0]
neuron60 = [0,.33,.67,0]
neuron90 = [0,0,1,0]
neuron120 = [0,0,.67,.33]
neuron150 = [0,0,.33,.67]
neuron180 = [0,0,0,1]

#d

#takes the output spike train from the RUN_LIF function and returns a frequency
def outputToFreq(x):
    totalSpikes = np.sum(x)/5.0
    return totalSpikes

#this inputs the weights of a neuron from above, and generates a list of output freq for a tuning curve
def neurFuncFromWeights(weights):
    freqs = []
    for degree in range(-179,180):
        v, x = RUN_LIF(getInput(degree,weights))
        freq = outputToFreq(x)
        freqs.append(freq)
    return freqs, x

#the x axis of all the degrees
d = range(-179,180)

#the response curve for each neuron (and the spike train for use later)
n_150, ns_150 = neurFuncFromWeights(neuron_150)
n_120, ns_120 = neurFuncFromWeights(neuron_120)
n_90, ns_90 = neurFuncFromWeights(neuron_90)
n_60, ns_60 = neurFuncFromWeights(neuron_60)
n_30, ns_30 = neurFuncFromWeights(neuron_30)
n0, ns0 = neurFuncFromWeights(neuron0)
n30, ns30 = neurFuncFromWeights(neuron30)
n60, ns60 = neurFuncFromWeights(neuron60)
n90, ns90 = neurFuncFromWeights(neuron90)
n120, ns120 = neurFuncFromWeights(neuron120)
n150, ns150 = neurFuncFromWeights(neuron150)
n180, ns180 = neurFuncFromWeights(neuron180)

print("Plot of the tuning curves of all 12 neurons")



plt.figure(1)
plt.plot(d, n_150, 'r--', d, n_120, 'b--', d, n_90, 'g--', d, n_60, 'r--', d, n_30, 'b--', d, n0, 'g--', d, n30, 'r--', d, n60, 'b--', d, n90, 'g--', d, n120, 'r--', d, n150, 'b--', d, n180, 'g--')
#plt.plot(d,n30,'r--')
plt.ylabel('Frequency')
plt.xlabel('Degree')

plt.show()

#e
"""
For this part, I subtracted the spike trains of the previous neighboring neurons from the input voltages 

However, something is going wrong with this, and I am not sure what, all the tuning curves are being pushed to the center
"""

#puts together an input y of voltages from all of the neurons for the RUN_LIF function to use
#subtracts spike trains from each neighboring neuron
def getInputInhibitory(degree,weights,inh1,inh2):
    n_90 = 5*spikesToInput(returnSpikes(sensNeuron_90(degree),1),weights[0])
    n0 = 5*spikesToInput(returnSpikes(sensNeuron0(degree),1),weights[1])
    n90 = 5*spikesToInput(returnSpikes(sensNeuron90(degree),1),weights[2])
    n180 = 5*spikesToInput(returnSpikes(sensNeuron180(degree),1),weights[3])
    return np.array(n_90)+np.array(n0)+np.array(n90)+np.array(n180)-2*(inh1)-2*(inh2)

#this inputs the weights of a neuron from above, and generates a list of output freq for a tuning curve using inhibitory getInput
def neurFuncFromWeights(weights,inh1,inh2):
    freqs = []
    for degree in range(-179,180):
        v, x = RUN_LIF(getInputInhibitory(degree,weights,inh1,inh2))
        freq = outputToFreq(x)
        freqs.append(freq)
    return freqs

#all of the neuron frequency response curves except also subtracts spike trains from each neighboring neuron
n_150i = neurFuncFromWeights(neuron_150,ns180,ns_120)
n_120i = neurFuncFromWeights(neuron_120,ns_150,ns_90)
n_90i = neurFuncFromWeights(neuron_90,ns_120,ns_60)
n_60i = neurFuncFromWeights(neuron_60,ns_90,ns_30)
n_30i = neurFuncFromWeights(neuron_30,ns_60,ns0)
n0i = neurFuncFromWeights(neuron0,ns_30,ns30)
n30i = neurFuncFromWeights(neuron30,ns0,ns60)
n60i = neurFuncFromWeights(neuron60,ns30,ns90)
n90i = neurFuncFromWeights(neuron90,ns60,ns120)
n120i = neurFuncFromWeights(neuron120,ns90,ns150)
n150i = neurFuncFromWeights(neuron150,ns120,ns180)
n180i = neurFuncFromWeights(neuron180,ns150,ns_150)

print("Plot of the tuning curves of all 12 neurons after inhibitory connections")

plt.figure(2)
plt.plot(d, n_150i, 'r--', d, n_120i, 'b--', d, n_90i, 'g--', d, n_60i, 'r--', d, n_30i, 'b--', d, n0i, 'g--', d, n30i, 'r--', d, n60i, 'b--', d, n90i, 'g--', d, n120i, 'r--', d, n150i, 'b--', d, n180i, 'g--')
#plt.plot(d,n30i,'r--')
plt.ylabel('Frequency')
plt.xlabel('Degree')

plt.show()

