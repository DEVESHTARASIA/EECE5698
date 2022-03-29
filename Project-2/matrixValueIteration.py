import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import copy

def valueIteration(classobject,gamma,theta):
    numActions = len(classobject.getActions())
    condition = True
    while condition:
        vk = copy.deepcopy(classobject.getValues())
        for k in range(numActions):
            