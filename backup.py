# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 00:04:18 2018

@author: gwenda
"""
#import matplotlib.pyplot as plt
import numpy as np
import math
import random
#from neuronpy.graphics import spikeplot

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



#this is just a gaussian function, the neurons use this to give an output frequency for a degree
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    




#object that manages the event queues.

class Controller():
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
    
    def add_input_event(self,event):
        self.input_queue.append(event)
        self.input_queue.sort(key=lambda e:e["time"])
        
    def add_invernal_event(self,event):
        self.internal_queue.append(event)
    
    #this is the main loop for all neurons
    def run(self,end_time):
        while len(self.input_queue) > 0:
            if self.input_queue[0]['time'] > end_time:
                return
            event = self.input_queue.pop(0)
            event['neuron'].process_event(event)
            
    
class Sensor():
    
    def __init__(self,controller):
        self.controller = controller
        self.best_orientation = 0 #the angle at which it fires the most
        self.stimulus_orientation = None
        self.rate = 0 #in Hz
        self.max_rate = 60 #in Hz
        self.selectivity_width = 45 #the standard deviation of the gaussian
        self.outputs = [] 
        self.weights = []

    #this sets the stimulus orientation and the rate based on the angle
    def set_stimulus_orientation(self,angle, start_time):
        self.stimulus_orientation = angle
        self.rate = gaussian(angle,self.best_orientation,self.selectivity_width)
        spike_time = start_time+(1/self.rate)*random.random()
        self.controller.add_input_event({'neuron':self, 'time': spike_time})
        
    def process_event(self,event):
        print(event["time"])
        spike_time = event['time'] + (1/self.rate) 
        self.controller.add_input_event({'neuron':self, 'time': spike_time})
        # This is ok for now (as long as the outputs are only recorders),
        # but we will probably have to rethink this when outputs are synapses (with delays)
        for idx in range(len(self.outputs)):
            outputs[idx].process_event(event,self.weights[idx])


    def add_output(self,output,weight):
        self.outputs.append(output)
        self.weights.append(weight)
                       
class lifNeuron():
    
    def __init__(self,controller):
        self.controller = controller
        self.resting_voltage = -70
        self.voltage = self.resting_voltage
        self.time = 0
        self.decay = .05 #this is a decay constant with units one over milliseconds
        self.inh_decay
        self.threshold = -55
        
        self.outputs = [] 
        self.weights = []
       
    def process_event(self,event, weight):
        print(event["time"])
        spike_time = event['time']
        delta_time = spike_time - self.time
        
        self.voltage = (self.voltage-self.resting_voltage)*math.exp(-delta_time*self.decay)+self.resting_voltage
        self.voltage = self.voltage+5*weight

            
        self.time = spike_time
        
        if self.voltage >= self.threshold:
            self.voltage = self.resting_voltage
            spike = {'neuron':self, 'time': self.time}
            for idx in range(len(self.outputs)):
                outputs[idx].process_event(spike,self.weights[idx])
            print("LIF spike")


    def add_output(self,output,weight):
        self.outputs.append(output)
        self.weights.append(weight)
            

def makeSynapse(input,output,weight):
    input.add_output(output,weight)
            
controller = Controller()
neuron = Sensor(controller)
lif = lifNeuron(controller)
makeSynapse(neuron,lif)
neuron.set_stimulus_orientation(80,1)
controller.run(100)
        
class SpikeRecorder():
                       
    def __init__(self):
        self.recordings = {} #initially an empty dictionary, but when full, will have structure {neuron:[], neuron2:[]}

    #adds a neuron pair to the dictionary
    def add_neuron(self,neuron):
        self.recordings[neuron] = []

    #add the events "time" to the appropriate recording
    def process_event(self,neuron):
        self.recordings[neuron].append(time)
    
    #returns the recordings in the format of a list of lists
    def get_spiketrains(self):
        #spiketrains should be all the times of the spikes in a list separated by neuron, a list of lists
        spiketrains = [] 
        for neuron in self.recordings:
            times = []
            for time in self.recordings[neuron]:
                times.append(time)
            spiketrains.append(times)
        return spiketrains
    
    def plot_recordings(self):
        #spiketrains should be all the times of the spikes in a list separated by neuron, a list of lists
        spiketrains = self.get_spiketrains()

        print(spiketrains)
        
                       
        #sp = spikeplot.SpikePlot()
        #sp.set_markerscale(0.5)

        #sp.plot_spikes(spiketrains)


