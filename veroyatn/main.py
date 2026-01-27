import numpy as np
import pandas as pd 
import random

def calculate_wins():
    ans = 0
    for i in range(1000):
        n1, n2, n3 = [0, 0, 0]
        f = False
        for j in range(9):
            n3 = n2
            n2 = n1
            n1 = random.randint(0, 1)
            # print(n1)
            if(n1 == 1 and n3 == 1):
                ans -= 1
                f = True
                break
        if(f):
            ans += 4
    return ans
            


        

def main():
    ans = calculate_wins()
    print(ans)

if __name__ == "__main__":
    main()