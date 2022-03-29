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
    vk = classObject.getValue()
    vOld = copy.deepcopy(vk)
    
    condition = True
    while condition:
        for i,j in classObject.stateGenerator():
            delta = 0
            prob,r,adjacentStates = classObject.reward((i,j))
            v = vOld[adjacentStates]
            v = gamma*v
            v = r + v
            ans = sum(np.multiply(prob,v))
            classObject.setValue((i,j),ans)
            delta = max(delta,abs(vOld[i,j]-ans))
            print(delta)
            if delta < theta:
                condition = False

def policyIter(classObject,theta,gamma):
    same = False
    while same == False:
        policyEvaluation(classObject,theta,gamma)
        vOld = classObject.getValue()
        policy = classObject.getPolicy()
        policyOld = copy.deepcopy(policy)
        
        for i,j in classObject.stateGenerator():
            print(i,j)
            actions = classObject.getActionList()
            highVal = classObject.getStateValue((i,j))
            print(i,j)
            for action in actions:
                prob,r,adjacentStates = classObject.reward((i,j),action)
                v = vOld[adjacentStates]
                v = gamma*v
                v = r + v
                ans = sum(np.multiply(prob,v))

                if ans > highVal:
                    classObject.setPolicy((i,j),action)
        
        if policyOld == classObject.getPolicy():
            same = True
                

