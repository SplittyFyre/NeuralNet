import random as rand
import numpy as np


fout = open("trainme", "w")

fout.write("0.2 0.2\n")
fout.write("2 4 1\n");

for i in range(0, 4000):
    fout.write("1.0 1.0\n")
    fout.write("0.0\n")
    fout.write("1.0 0.0\n")
    fout.write("1.0\n")
    fout.write("0.0 0.0\n")
    fout.write("0.0\n")
    fout.write("0.0 1.0\n")
    fout.write("1.0\n")
