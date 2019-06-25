# import data_preprocessing as dp
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessing import bag_of_words,count_matrix,data1,count
import numpy as np
import pandas as pd


def return_vec(x,k):
    temp = []
    for i in x:
        temp+=(bag_of_words[i].strip().split())
    temp = count.transform([" ".join(list(set(temp)))])
    store = cosine_similarity(temp, count_matrix)
    
    print("Items bought:")
    for i in x:
        # print(data1[i][1],'\t\t\t',data1[i][2],'\t\t\t',data1[i][3])
        print(data1[i][1])
    
    print()
    
    recomm = np.argsort(-store[0])
    
    print("Recommended Items:")
    for i in range(len(recomm)):
        if not k:
            break
        if(recomm[i] in x) or (store[0][recomm[i]]==0):
            continue
        # print(data1[recomm[i]][1],'\t\t\t',data1[recomm[i]][2],'\t\t\t',data1[recomm[i]][3])
        print(data1[recomm[i]][1])
        k-=1

bought = [8898,2345,13000,12]
return_vec(bought,10)
print('-----------------------------------------------------')
bought = [596,568,23,34,569,4577,7899,12345,4565]
return_vec(bought,20)