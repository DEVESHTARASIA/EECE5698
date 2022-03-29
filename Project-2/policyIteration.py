#Imports
from cmath import inf
import numpy as np
import pandas as pd
import math as maths
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import copy

def policyEvaluation(classObject,theta,gamma):
    
    condition = True
    while condition == True:
        delta = 0
        vold = classObject.getValue()
        vk = copy.deepcopy(vold)
        for t in classObject.stateGenerator():
            prob,r,adjacentStates = classObject.reward(t)
            v = vk[adjacentStates]
            v = gamma*v
            v = r + v
            ans = sum(np.multiply(prob,v))
            classObject.setValue(t,ans)
            delta = max(delta,abs(vk[t]-ans))
        if delta < theta:
            condition = False
    
def policyIter(classObject,theta,gamma):
    stopAfter = 0
    same = False
    actions = classObject.getActionList()

    while same == False:
        policyEvaluation(classObject,theta,gamma)
        vOld = classObject.getValue()
        vk = copy.deepcopy(vOld)
        policy = classObject.getPolicy()
        policyOld = copy.deepcopy(policy)
        
        for t in classObject.stateGenerator():
            highVal = vk[t]
            for action in actions:
                prob,r,adjacentStates = classObject.reward(t,action)
                v = vOld[adjacentStates]
                v = gamma*v
                v = r + v
                ans = sum(np.multiply(prob,v))

                if ans > highVal:
                    classObject.setStatePolicy(t,action)
                    highVal = ans
        
        if np.array_equal(policyOld,classObject.getPolicy()):
            same = True
            print('Stopped')
                

