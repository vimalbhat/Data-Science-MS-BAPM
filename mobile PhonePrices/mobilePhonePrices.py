# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:46:04 2019

@author: Vimal Bhat
"""

def getLargest(x,y):
    """
    largest number
    >>>getKargest(5,10)
    10
    """
    return max(x,y)

print(getLargest(5,10))

print(10 if 10>5 else 2)
hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Comma after the last element is optional)
]
hands[0][1:3]

arrivals = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']
name="Ford"
print("fashionably late" if((round(len(arrivals)/2) <arrivals.index(name)+1) and arrivals.index(name)+1 !=len(arrivals) ) else None)

squares=[n*n for n in range(5,10)]
squares     
range(0)

doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
list=[index for index in range(len(doc_list)) if "CASINO" in doc_list[index].upper()]


#Pandas
a=[]
a.

import pandas as pd
data=pd.read_csv("D:/Kaggle/mobile PhonePrices/train.csv")
data.describe()
data.iloc[:,1]
train_data=data[data.blue,data.wifi]
    data.blue.str.contains("1",regex=False)
pd.Series.
data.isnull().any()
data[data.blue==1]

temp=pd.DataFrame({"Apple":[1,2],"Orange":[3,5]},index={"a","c"})

nums=[3,2,4]
i=0
def twoSum( nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)): 
            print("i"+str(i))
            
            for j in range(i+1,len(nums)):
                print("j"+str(j))
                if nums[i] + nums[j] == target:
                    print("inside")
                    print(i,j)
                    return [i,j]
                
twoSum([3,2,4],6)
len([3,2,4])


s="dvdf"
len(s)

max_length=0
arr=[]
i=0
for char in s:
    print(char)
    if(not(s[i] in arr) ):
        arr.append(char)
    else:
       # arr.append(char)
        print("else")
        print(arr)
        max_length=len(arr) if len(arr)> max_length else max_length
        arr=[]
      #  arr.clear()
        arr.append(char)
    i=i+1
max_length=len(arr) if len(arr)> max_length else max_length




dct = {}
max_so_far = curr_max = start = 0
for index, i in enumerate(s):
    if i in dct and dct[i] >= start:
        max_so_far = max(max_so_far, curr_max)
        curr_max = index - dct[i]
        start = dct[i] + 1
    else:
        curr_max += 1
    dct[i] = index
    
    
s="aaabaaaa"
pali=[]

for i,char in enumerate(s):
#    print(str(i) + char)
    j=len(s)-1
    indexes=set([])
    while(i<=j):
#        print(str(i)+"   "+ str(j))
        if(s[i]==s[j]):
#            print("if")
            indexes.add(i)
            indexes.add(j)
            i=i+1
            j=j-1
        else:
#            print("else")
            indexes=set([])
            j=j-1
            
    if(len(indexes) > len(pali)):        
        pali=[]
        for i in sorted(indexes):
            pali.append(s[i])
    print( "".join(pali))