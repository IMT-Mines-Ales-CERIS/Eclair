import numpy as np
import random
import time
import os
from multiprocessing import Pool, TimeoutError
from Utils import Utils

def f(x):
    time.sleep(0.75 + random.random())
    return x**x

def fp(x,y):
    time.sleep(0.75 + random.random())
    return x+y, x*y

if __name__ == '__main__':
    x = np.array([
        [1, 2, 3, 6],
        [4, 5, 6, 15],
        [7, 8, 9, 24],
        [10, 11, 12, 33],
        [10, 20, 30, 60],
        [40, 50, 60, 150]
    ])
    print(Utils.Kfold(x))

    # print()

    # start = time.time()
    # with Pool(processes=4) as pool:
    #     async_results = [pool.apply_async(
    #         f,
    #         args=(
    #            k,
    #         ))
    #         for k in range(num)
    #     ]
    #     results = [res.get() for res in async_results]
    # print(f'Time : {time.time() - start}', flush=True)