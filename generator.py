import numpy as np
import random
import colored

re = colored.attr('reset')
na = 0    
a = "A" 
b = "B"  
c = "C"   
d = "D"
e = "E"
f = "F"
g = "G"
h = "H"
i = "I"
j = "J"
k = "K"
l = "L"
m = "M"
n = "N"
o = "O"
dic = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o]

n = 4                               #Grid dimension
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



def flowPrinter(A):                 #An auxiliary function for printing the puzzle
    C = []                          #We make an empty list...
    for z in range(n):
        C.append([na]*n)            #... and fill it with n' sublists of 'n' entries whose content is a black double space ('na' is black, according to our color definition)
    for i in range(len(A)):
        for j in range(len(A[i])):              
            x = A[i][j][0][0] - 1               #We find the indices of each line element...
            y = A[i][j][0][1] - 1               
            C[x][y] = A[i][j][1]                #... and overwrite that element in 'C' with the appropriate color
    s = ""
    for k in range(n):                          #Finally, we print every line of  'C' as a single string.
        for l in range(n):
            s = s + C[k][l]
        print(s)
        s = ""
def flowPrinter_puzzle(A):
    C = []
    for z in range(n):
        C.append([na]*n)
    for i in range(len(A)):
        for j in range(-1,1):
            x = A[i][j][0][0] - 1
            y = A[i][j][0][1] - 1
            C[x][y] = A[i][j][1]
    grid=[]
    for k in range(n):
        row=[]
        for l in range(n):
            row.append(C[k][l])
        grid.append(row)
    for i in grid:
        print(i)



flow = baseMatrix(n)
for step in range(0,iter):
    flow = edgeSwitch(flow)
    random.shuffle(flow)
flowPrinter(flow)
print('\n \n')
flowPrinter_puzzle(flow)
