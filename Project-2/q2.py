#Imports
import numpy as np
import pandas as pd
import math as maths
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd

def XOR(a, b):
    if a != b:
        return 1
    else:
        return 0

def to_binary(n):
  bin_arr = [0,0,0,0]
  i = 0
  while (n>0):
      bin_arr[i] = n%2
      n = int(n/2)
      i = i+1
  bin_arr.reverse()
  return np.array(bin_arr)

def to_decimal(arr):
  bin_arr = list(arr)
  bin_arr.reverse()
  dec_val = 0
  for i in len(bin_arr):
    dec_val = dec_val + bin_arr[i] * (2**i)
  
  return dec_val

class Genes():
  def __init__(self,actionList,prob,connectivityMatrix):
    self.actions = actionList
    self.p = prob
    self.C = connectivityMatrix
    self.Rsas = np.array([np.zeros((16,16))] * len(self.actions))
    self.transistion = np.array([np.zeros((16,16))] * len(self.actions))
    self.Ras = np.zeros((len(self.actions),16))
    self.values = np.zeros((16,1))

    for k in range(len(self.actions)):
      for i in range(16):
        for j in range(16):
          statei = to_binary(i).reshape(4,1)
          statej = to_binary(j).reshape(4,1)
          v = np.matmul(self.C,statei)
          v = np.array(list(map(XOR,v,self.actions[k]))).reshape(4,1)
          v = statej - v
          value = abs(int(sum(v)))
          self.transistion[k,i,j] = (self.p**value)*((1-self.p)**(4-value))
          self.Rsas[k,i,j] = 5*sum(statej) - sum(self.actions[k])
    
    for k in range(len(self.actions)):
      ans = np.multiply(self.transistion[k],self.Rsas[k])
      self.Ras[k] = np.sum(ans,axis=1)
    
    self.Ras = np.reshape(self.Ras,(len(self.actions),16,1))

  def getTransition(self):
    return self.transistion      

  def getReward(self):
    return self.Ras

  def getActions(self):
    return self.actions
  
  def getValues(self):
    return self.values


p = 0.05
C = np.array([[0,0,-1,0],[1,0,-1,-1],[0,1,0,0],[-1,1,1,0]])
actionList = np.array([[0,0,0,0],[1,0,0,0],[0,0,0,1]])

gene1 = Genes(actionList,p,C)
print(gene1.getReward().shape)