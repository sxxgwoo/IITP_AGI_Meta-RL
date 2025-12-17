import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    hey = torch.tensor([[1,2,3,4,5],[2,3,4,5,6]])
    print(hey.sum(-1))    
    yeap = torch.tensor([1,3])
    print((hey.sum(-1)*yeap).sum(-1))
    
    with open("HI.json","wb") as f:
        pickle.dump([1,2,3,4],f)
        
    with open("HI.json","rb") as f:
        yeah = pickle.load(f)
        print(yeah)
        print(type(yeah))
        