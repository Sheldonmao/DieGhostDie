"""
The file to call capture.py several times for training
reinforcement learning agent.
"""

import os
import random

if __name__ == '__main__':
    while True:
        num = str(random.randint(0, 10000))
        os.system('python capture.py -l RANDOM' + num)