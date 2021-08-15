import numpy as np

def forward(args):
    (x,) = args
    y = (x>0)*x
    return y

def backward(args):
    (y, dy) = args
    dx = (y>0)*dy
    return dx
