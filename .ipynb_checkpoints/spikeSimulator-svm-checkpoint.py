# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 00:04:18 2018

@author: gwenda
"""
#import matplotlib.pyplot as plt
import numpy as np
import math
import random
import cv2
import pdb
#from neuronpy.graphics import spikeplot

random.seed(0)
np.random.seed(10)

#this is just a gaussian function, the neurons use this to give an output frequency for a degree
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    

def load_mnist():
    with open('/home/claw/Preston/cs574/p3_mnist/mnist/t10k-labels-idx1-ubyte', 'rb') as fp:
        magic = fp.read(4)
        num = fp.read(4)
        num = int(num.encode('hex'), 16)
        test_labels = np.fromfile(fp, dtype=np.uint8, count=num)
    with open('/home/claw/Preston/cs574/p3_mnist/mnist/train-labels-idx1-ubyte', 'rb') as fp:
        magic = fp.read(4)
        num = fp.read(4)
        num = int(num.encode('hex'), 16)
        train_labels = np.fromfile(fp, dtype=np.uint8, count=num)
    with open('/home/claw/Preston/cs574/p3_mnist/mnist/t10k-images-idx3-ubyte', 'rb') as fp:
        magic = fp.read(4)
        num = fp.read(4)
        num = int(num.encode('hex'), 16)
        rows = fp.read(4)
        rows = int(rows.encode('hex'), 16)
        cols = fp.read(4)
        cols = int(cols.encode('hex'), 16)
        test_images = np.fromfile(fp, dtype=np.uint8, count=num*rows*cols)
        test_images = test_images.reshape(num, rows, cols)
    with open('/home/claw/Preston/cs574/p3_mnist/mnist/train-images-idx3-ubyte', 'rb') as fp:
        magic = fp.read(4)
        num = fp.read(4)
        num = int(num.encode('hex'), 16)
        rows = fp.read(4)
        rows = int(rows.encode('hex'), 16)
        cols = fp.read(4)
        cols = int(cols.encode('hex'), 16)
        train_images = np.fromfile(fp, dtype=np.uint8, count=num*rows*cols)
        train_images = train_images.reshape(num, rows, cols)
    #cv2.imwrite('images/mnist.png', test_images[0, ...])
    return train_images, train_labels, test_images, test_labels


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
        self.queue = []
        self.end_time = 0
        self.neuron_has_won = False
        self.neuron_that_won = None
    
    def add_event(self,event):
        self.queue.append(event)
        self.queue.sort(key=lambda e:e["time"])
    
    #this is the main loop for all neurons
    def run(self):
        while len(self.queue) > 0:
            event = self.queue.pop(0)
            if event['type'] == 'input':
                event['neuron'].process_input_event(event)
                self.end_time = event['time']
            elif event['type'] == 'forced':
                event['neuron'].process_forced_event(event)
                self.end_time = event['time']
            elif event['type'] == 'learn':
                event['neuron'].update_weights(event)
    
    def run_to_test(self):
        while len(self.queue) > 0:
            event = self.queue.pop(0)
            if event['type'] == 'input':
                event['neuron'].process_input_event(event,True)
                self.end_time = event['time']
            elif event['type'] == 'forced':
                event['neuron'].process_forced_event(event)
                self.end_time = event['time']
                
            #no learning for testing phase
            
            if self.neuron_has_won:
                self.neuron_that_won = event['neuron']
                self.queue = []

            
        
        
class InputNeuron():
    
    def __init__(self,controller):
        self.controller = controller
        
        self.output_synapses = []

    #this sets the pixel value
    def set_value(self, value, time):
        self.value = value
        if value > 125:
            spike_time = time+random.random()
            self.controller.add_event({'neuron':self, 'time': spike_time, 'type': 'forced', 'synapse':None})

    def process_forced_event(self,event):
        for idx in range(len(self.output_synapses)):
            spike = {'neuron':self.output_synapses[idx][1], 'time': event['time'], 'type':'input', 'synapse':self.output_synapses[idx]}
            self.controller.add_event(spike)


    def add_output(self,synapse):
        self.output_synapses.append(synapse)  
        
                       
class lifNeuron():
    
    def __init__(self,controller):
        self.id = random.randrange(0,100)
        self.controller = controller
        self.resting_voltage = -70
        self.voltage = self.resting_voltage
        self.time = 0
        self.decay = .05 #this is a decay constant with units one over milliseconds
        self.ahp_decay = 1/5000 #this is for slow afterhyperpolarization
        self.ahp = 0
        self.threshold = -55
        self.input_synapses = []
        self.inh_input_synapses = []
        self.output_synapses = [] 
        self.learning_rate = 0.001
        self.forward_window = 1
       
    def process_forced_event(self,event):
        spike_time = event['time']
        
        #update voltage to the time of the input
        self.update_voltage(spike_time)
        
        self.fire()
        
            
    def process_input_event(self,event,toTest = False):
        spike_time = event['time']
        synapse = event['synapse']
        
        #update voltage to the time of the input
        self.update_voltage(spike_time)

        synapse[3] = spike_time
        
        #add the input to the voltage
        self.voltage = self.voltage+ 5*synapse[2]
        
        if self.voltage >= self.threshold:
            if toTest:
                self.controller.neuron_has_won = True
            self.fire()
            
            
    def fire(self):
        self.voltage = self.resting_voltage
        for idx in range(len(self.output_synapses)):
            spike = {'neuron':self.output_synapses[idx][1], 'time': self.time, 'type':'input', 'synapse':self.output_synapses[idx]}
            self.controller.add_event(spike)
            
        learning_event = {'neuron':self, 'time': self.time + self.forward_window, 'type':'learn', 'synaspse':None}
        self.controller.add_event(learning_event)
        
        print("LIF spike %d"%self.id)

    def update_voltage(self,time):
        delta_time = time - self.time
        #have to calculate ahp effect
        self.ahp = self.ahp*math.exp(-delta_time*self.ahp_decay)
        # AHP cunductance affect the equilibrium voltage (closer to K reversal potential)
        veq = self.resting_voltage - self.ahp;
        # Decay the voltage toward the reversal potential
        self.voltage = (self.voltage-veq)*math.exp(-delta_time*self.decay)+veq
        #update to current time
        self.time = time
            
    def update_weights(self,event):
        #this only gets called when the neuron spikes
        time_of_spike = event['time']-self.forward_window
        for synapse in self.input_synapses:
            last_spike_time = synapse[3]
            dt = time_of_spike - last_spike_time
            if dt > -1 and dt < .6:
                synapse[2] -= self.learning_rate
                if synapse[2] < 0:
                    synapse[2] = 0
            elif dt > 1 and dt < 2:
                synapse[2] += self.learning_rate
            
            
    def normalize_input_weights(self):
        #compute the magnitude of the weight vector
        sum = 0
        for synapse in self.input_synapses:
            sum += synapse[2]*synapse[2]
        magnitude = math.sqrt(sum)        
        for synapse in self.input_synapses:
            synapse[2] /= magnitude
            
    def add_input(self,synapse):
        if synapse[2] < 0 :
            self.inh_input_synapses.append(synapse)
        else :
            self.input_synapses.append(synapse)
        
        
    def add_output(self,synapse):
        self.output_synapses.append(synapse)
        
    #this will have to take the weights matrix and change it to 10 28*28 images
    def weights_to_images(self):
        weights = np.array([s[2] for s in self.input_synapses])
        image = weights.reshape((28,28))
        image = image*255/np.max(image)
        cv2.imwrite("images/image%d.png"%self.id,image)

            

def makeSynapse(in_neuron,out_neuron,weight):
    synapse = [in_neuron,out_neuron,weight,-100]
    in_neuron.add_output(synapse)
    out_neuron.add_input(synapse)

class Network():
    
    def __init__(self):
        self.ending_time = 0
        self.controller = Controller()
        
        #make the ten neruons that represent clusters
        self.neurons = []
        for i in range(10):
            self.neurons.append(lifNeuron(self.controller))
            self.neurons[i].id = i
        
        #make the 28 by 28 grid of input neurons  
        self.inputNeurons = []
        for i in range(784): 
            self.inputNeurons.append(InputNeuron(self.controller))
            #fully connect the input neurons with the ten hidden neurons
            for j in range(10):
                #makeSynapse(self.inputNeurons[i],self.neurons[j],random.random())
                makeSynapse(self.inputNeurons[i],self.neurons[j],.000000001)
              
        
        #for i in range(10):
        #    self.neurons[i].normalize_input_weights()

        
        
    def present_image(self,img,time):
        img_flat = img.reshape((784))
        for i in range(784):
            self.inputNeurons[i].set_value(img_flat[i],time)
            
            
    def supervise(self,label,time):
        spike = {'neuron':self.neurons[label], 'time': time, 'type':'forced', 'synapse':None}
        self.controller.add_event(spike)
            
            
    def train(self):
        images = load_mnist()[0]
        labels = load_mnist()[1]
        time = 0
        for j in range(1):
            for i in range(len(images)):

                self.present_image(images[i],time)
                self.supervise(labels[i],time+2)
                self.controller.run()
                #net.create_images()
                
                time = time + 1000
                print("image %d"%i)
                #pdb.set_trace()
        self.ending_time = time
                
    
    def test(self):
        images = load_mnist()[2]
        labels = load_mnist()[3]
        time = self.ending_time
        winners = []
        self.controller.neuron_has_won = False
        num_correct = 0

        for i in range(len(images)):
            print("image %d"%i)
            print("label is %d"%labels[i])
            winner = None

            while not self.controller.neuron_has_won:
                self.present_image(images[i],time)
                self.controller.run_to_test()
                time = self.controller.end_time + .01

                
            winner = self.controller.neuron_that_won.id
            if winner == labels[i]:
                num_correct += 1
            winners.append(winner)
            self.controller.neuron_has_won = False
            time = time + 1000
            
        

                
        percent_correct = 100*(num_correct/len(images))
        print("The number correct is %d"%num_correct)
        print("The percent correct is %d"%percent_correct)

            
    def create_images(self):
        for neuron in self.neurons:
            neuron.weights_to_images()
        
    
net = Network()
net.train()
net.create_images()
net.test()
        




