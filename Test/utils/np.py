import numpy as np

def numpy_from_file(filename):
    return np.genfromtxt(filename, delimiter=' ')

def numpy_from_str(txt):
    return np.array([[float(e) for e in row.strip(" ").split(" ")] for row in txt.split("\n")])