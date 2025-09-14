import numpy as np
import random
import colored
import json 

re = colored.attr('reset')
na = 0    
a = 1 
b = 2  
c = 3   
d = 4
e = 5
f = 6
g = 7
h = 8
i = 9
j = 10
k = 11
l = 12
m = 13
n = 14
o = 15
dic = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o]

n = 5                               #Grid dimension
chain_limit = 2                   #Minimum line length
iter = 400                          #Number of iterations

def baseMatrix(dim):                #Generates the initial state of the grid with horizontal lines of length 'n'
    A = []                          
    for l in range(dim):
        A.append([])                #Adds 'dim' empty entries in the first level of the list
    for i in range(dim):
        for j in range(dim):
            A[i].append([np.array([i+1,j+1]), dic[i]])              #Fills every level 0 list with the line's informatio:
                                                                    #XY coordinates of each point on the line, as well as a tag indicating the line's color                                                                 #The beginning and end of this list are the line's tails
                                                                    #Selecting a list from 'A' is the same as selecting a whole line
    return A



def edgeSwitch(A):                  #Most important code of this generator: this function extends/shrink two lines whose tails share an edge                                    #First, it analyzes every line:
    sw = False
    for i in range(len(A)):                         #For each row (that is, for every line)...
        if sw == True:
            break
        for k1 in range(-1,1):                      #For every tail of the selected line...
            if sw == True:
                break
            p = A[i][k1][0]                         #where 'p' is the position of the selected tail...
            for j in range(len(A)):                 #For all the other lines...
                if sw == True:
                    break
                if j == i:
                    continue
                if len(A[j]) == chain_limit:        #with length greater than 'chain_limit'...
                    continue
                for k2 in range(-1,1):              #For every tail of the second line...
                    if sw == True:
                        break
                    pprime = A[j][k2][0]            #where'pprime' is the position of the seond line's tail 
                    if np.linalg.norm(p-pprime) == 1.0:                 #If the distance between both positions is exactly one, then they share an edge.
                        n1 = np.random.rand()                           #We flip a coin, and if 'n1' is greater than 0.5...
                        if n1 > 0.5:
                            A[j].pop(k2)                                #We remove the second line's tail...
                            if k1 == -1:
                                A[i].append([pprime,A[i][0][1]])        #... and add it to the first one.
                            elif k1 == 0:
                                A[i].insert(0,[pprime,A[i][0][1]])
                            sw = True
                    else:
                        continue                                        #If 'n1' is not greater than 0.5, we simply continue with another (second) line 
    return A


def flowPrinter_puzzle(A):
    color_points=[]
    colors=[]
    for i in range(len(A)):
        k=0
        color=[]

        for j in range(-1,1):
            x = int(A[i][j][0][0] - 1)
            y = int(A[i][j][0][1] - 1)
            if A[i][j][1] not in  color:
                color.append(A[i][j][1])
            color.append([x,y])
            k+=1
        color_points.append(color)
        colors.append(color[0])
            
    return(colors,color_points)


flow = baseMatrix(n)
puzzles=[]

for i in range(200):
    for step in range(0,iter):
        flow = edgeSwitch(flow)
        random.shuffle(flow)

    colors,color_points=flowPrinter_puzzle(flow)
    puzzles.append({"colors":colors,"color_points":color_points})

with open("puzzles.jsonl", "w") as f:
    for line in puzzles:
        f.write(json.dumps(line) + "\n")
    


