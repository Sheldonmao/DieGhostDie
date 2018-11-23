"""
The file to call capture.py several times for training
reinforcement learning agent.
"""

import os
import random

if __name__ == '__main__':
    turns = 0
    while turns < 500:
        turns += 1
        if turns < 100:
            epsilon = 0.2 - turns * 0.0015
            with open('epsilon.txt', "w") as f:
                f.write(str(epsilon))
        num = str(random.randint(0, 10000))
        os.system('python capture.py -q -l RANDOM' + num)