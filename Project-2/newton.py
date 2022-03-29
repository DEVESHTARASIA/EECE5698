from scipy import optimize
import math

def f(x1,x2):
    exp(x1+3*x2-0.1) + exp(x1-3*x2-0.1) + exp(-x1-0.1)

root = optimize.newton(f,[0,0])