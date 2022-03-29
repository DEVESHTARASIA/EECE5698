#Imports
import numpy as np
import pandas as pd
import math as maths
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import policyIteration
import valueIteration
from matplotlib.patches import Rectangle

class Maze():
    def __init__(self,nRows,nCols,actionList,prob,startPosn,goalPosn,oilPosn,bumpPosn,wallPosn,oilR,bumpR,goalR,emptyR,actionR):
        self.nRows = nRows
        self.nCols = nCols
        self.actionList = actionList
        self.startPosn = startPosn
        self.goalPosn = goalPosn
        self.oilPosn = oilPosn
        self.bumpPosn = bumpPosn
        self.wallPosn = wallPosn
        self.oilR = oilR
        self.bumpR = bumpR
        self.goalR = goalR
        self.emptyR = emptyR
        self.actionR = actionR
        self.p = prob
        self.maze = np.zeros((self.nRows,self.nCols))
        self.values = np.zeros((self.nRows,self.nCols))
#        self.policy = [[rnd.choice(self.actionList)] * self.nCols] * self.nRows
        self.policy = np.random.choice(self.actionList,(self.nRows,self.nCols))
        self.currentI = 1
        self.currentJ = 1

        self.maze[self.goalPosn] = self.goalR
        self.values[self.goalPosn] = self.goalR
        for i,j in self.oilPosn:
            self.maze[i,j] = self.oilR
        for i,j in self.bumpPosn:
            self.maze[i,j] = self.bumpR
        for i,j in self.wallPosn:
            self.maze[i,j] = 0.0

    def getActionList(self):
        return self.actionList

    def getStateValue(self,currentState): #May not be needing it
        return self.values[currentState]

    def getValue(self):
        return self.values
    
    def setValue(self,currentState,value):
        self.values[currentState] = value
    
    def getStatePolicy(self,currentState):
        return self.policy[currentState]

    def getPolicy(self):
        return self.policy

    def setStatePolicy(self,currentState,action):
        self.policy[currentState] = action

    def stateGenerator(self):
        i = 1
        j = 1
        while i < self.nRows - 1:
            if (i,j) not in self.wallPosn:
                yield (i,j)
            j = j + 1
            if j == self.nCols - 1:
                j = 1
                i = i + 1

    def reward(self,currentState,action=None):
        i,j = currentState
        if action == None:
            action = self.policy[currentState]
        probabilities = []
        futureStates = [currentState,(i,j+1),(i,j-1),(i+1,j),(i-1,j)] #current,R,L,D,U
        rewards = np.array([self.maze[k]+self.actionR for k in futureStates])
        
        if action == 'R':
            probabilities = [0,1-self.p,self.p/3,self.p/3,self.p/3]
        if action == 'L':
            probabilities = [0,self.p/3,1-self.p,self.p/3,self.p/3]
        if action == 'D':
            probabilities = [0,self.p/3,self.p/3,1-self.p,self.p/3]
        if action == 'U':
            probabilities = [0,self.p/3,self.p/3,self.p/3,1-self.p]
        
        for k in range(1,len(futureStates)): #to add probabilities if wall is in side
            if futureStates[k] in wallPosn: 
                probabilities[0] += probabilities[k]
                probabilities[k] = 0

        xaxis = [i[0] for i in futureStates]
        yaxis = [i[1] for i in futureStates]
        adjacentStates = (xaxis,yaxis)
        probs = np.array(probabilities)

        return probs, rewards, adjacentStates


gamma = 0.55
theta = 0.01
nrows = 20
ncols = 20
prob = 0.02
actionList = ['R','L','U','D']
oilRew = -5
bumpRew = -10
goalRew = 200
actionRew = -1
emptyRew = 0
goalPosn = (3,13)
startPosn = (15,4)
oilPosn = [(2,8),(2,16),(4,2),(5,6),(9,18),(15,10),(16,10),(17,14),(17,17),(18,7)]
bumpPosn = [(1,11),(1,12),(2,1),(2,2),(2,3),(5,1),(5,9),(5,17),(6,17),(7,17),(8,17),(7,10),(7,11),(7,2),(12,11),(12,12),(14,1),(14,2),(15,17),(15,18),(16,7)]
wallPosn = [(0,i) for i in range(ncols)] + [(i,0) for i in range(nrows)] + [(i,ncols-1) for i in range(nrows)] + [(nrows-1,i) for i in range(ncols)]
wallPosn = wallPosn + [(2,5),(3,5)] + [(4,i) for i in range(3,17)] + [(i,6) for i in range(6,13)]
wallPosn = wallPosn + [(7,12),(7,13),(7,14),(15,14),(15,15),(15,16),(11,16),(11,17),(12,17),(13,17),(17,1),(17,2),(10,1),(10,2),(10,3),(10,4),(5,3),(6,3),(7,3),(12,3),(12,4),(12,5),(12,7),(13,7),(14,7),(15,7)] + [(17,i) for i in range(7,13)] + [(i,9) for i in range(6,11)] + [(i,10) for i in range(10,15)] +\
            [(i,15) for i in range(6,12)] + [(i,13) for i in range(11,16)]

mazeq1 = Maze(nrows,ncols,actionList,prob,startPosn,goalPosn,oilPosn,bumpPosn,wallPosn,oilRew,bumpRew,goalRew,emptyRew,actionRew)

policyIteration.policyIter(mazeq1,theta,gamma)
#valueIteration.valueIter(mazeq1,theta,gamma)
State_Matrix = mazeq1.getValue()

""" Function to always color the oil, bump, start, and green blocks.
 States are in the form of a list of (i,j) coordinates on the state matrix"""
def coloring_blocks(heatmap, oil_states, bump_states, start_state, end_state):
    # Adding red oil blocks
    for i in range(len(oil_states)):
        heatmap.add_patch(Rectangle((oil_states[i][1], oil_states[i][0]), 1, 1,
                                    fill=True, facecolor='red', edgecolor='red', lw=0.25))
    # Adding salmon bump blocks
    for i in range(len(bump_states)):
        heatmap.add_patch(Rectangle((bump_states[i][1], bump_states[i][0]), 1, 1,
                                    fill=True, facecolor='lightsalmon', edgecolor='lightsalmon', lw=0.25))
    # Adding start block (Blue)
    heatmap.add_patch(Rectangle((start_state[1], start_state[0]), 1, 1,
                                fill=True, facecolor='lightblue', edgecolor='lightblue', lw=0.25))

    # Adding end block (Green)
    heatmap.add_patch(Rectangle((end_state[1], end_state[0]), 1, 1,
                                fill=True, facecolor='lightgreen', edgecolor='lightgreen', lw=0.25))

for k in wallPosn:
    State_Matrix[k] = np.NAN

plt.subplots(figsize=(20,15))    
heatmap = sns.heatmap(State_Matrix, fmt=".2f", annot=State_Matrix,linewidths=0.25, linecolor='black',
                      cbar= False, cmap= 'rocket_r')
heatmap.set_facecolor('black') # Color for the NaN cells in the state matrix
coloring_blocks(heatmap, oil_states=oilPosn, bump_states=bumpPosn, \
                start_state=startPosn,end_state=goalPosn)
plt.savefig('PI-Value-c.png')


plt.subplots(figsize=(20,15))    
heatmap = sns.heatmap(State_Matrix, fmt='',annot=np.array(mazeq1.getPolicy()),linewidths=0.25, linecolor='black',
                      cbar= False, cmap= 'rocket_r')
heatmap.set_facecolor('black') # Color for the NaN cells in the state matrix
coloring_blocks(heatmap, oil_states=oilPosn, bump_states=bumpPosn, \
                start_state=startPosn,end_state=goalPosn)
plt.savefig('PI-Policy-c.png')