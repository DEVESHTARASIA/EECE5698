#Imports
import numpy as np
import pandas as pd
import math as maths
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import copy

def valueIter(classObject,theta,gamma):
    condition = True
    actions = classObject.getActionList()
    
    while condition:
        policies = classObject.getPolicy()
        oldPolicy = copy.deepcopy(policies)
        values = classObject.getValue()
        oldValues = copy.deepcopy(values)
        delta = 0
        for t in classObject.stateGenerator():
            highVal = -10000
            for action in actions:
                prob,r,adjacentStates = classObject.reward((t),action)
                v = oldValues[adjacentStates]
                v = gamma*v
                v = r + v
                ans = sum(np.multiply(prob,v))

                if ans > highVal:
                    classObject.setStatePolicy(t,action)
                    highVal = ans
                    classObject.setValue(t,ans)
                    #delta = max(delta,abs(oldValues[t]-ans))

            delta = max(delta,abs(oldValues[t]-classObject.getStateValue(t)))
        
        if delta < theta:
            condition = False

                     


