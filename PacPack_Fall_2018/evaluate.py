"""
The file to run a complete strategy and
count the winning rate and average steps
to evaluate.
"""

import os
import random

if __name__ == '__main__':
    times = 0
    steps = 0
    for i in range(0, 1000):
        num = str(random.randint(1, 10000))
        r = os.popen('python capture.py -l RANDOM' + num)
        infos = str(r.readlines()).split('\\n')
        print(infos[-2])
        times += 1
        if infos[-2].split()[-1] != 'moves':
            print(1200)
            steps += 1200
        else:
            thisStep = int(infos[-2].split()[-2])
            print(thisStep)
            steps += thisStep
    print(float(steps / times))
