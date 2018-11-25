"""
The file to run a complete strategy and
count the winning rate and average steps
to evaluate.
"""

import os
import random

if __name__ == '__main__':
    while True:
        num = str(random.randint(1, 10000))
        os.system('python capture.py -l RANDOM' + num)
