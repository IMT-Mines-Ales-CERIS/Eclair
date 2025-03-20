from multiprocessing import Pool, TimeoutError
import random
import time
import os

def f(x):
    time.sleep(0.75 + random.random())
    return x**x

def fp(x,y):
    time.sleep(0.75 + random.random())
    return x+y, x*y

if __name__ == '__main__':
    num = 10
    # start = time.time()

    # res = []
    # for i in range(num):
    #     res.append(f(i))
    # print(res)

    # print(f'Time : {time.time() - start}', flush=True)


    start = time.time()
    # # start 4 worker processes
    with Pool() as pool:

        # print "[0, 1, 4,..., 81]"
        # res = pool.map(f, range(num))
        # multi arg
        res = pool.starmap(fp, [(i, i+1) for i in range(num)])
        print(res)
        
    print(f'Time : {time.time() - start}', flush=True)

    start = time.time()
    # # start 4 worker processes
    with Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
        res = pool.imap_unordered(f, range(num))
        print(list(res))
        
    print(f'Time : {time.time() - start}', flush=True)

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