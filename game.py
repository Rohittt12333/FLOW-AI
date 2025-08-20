def addDot(color,pos):
    if color not in colors:
        state="invalid"
        
    elif pos[0]<0 or pos[0]>height:
        state="invalid"
        
    elif pos[1]<0 or pos[1]>length:
        state="invalid"
        
    elif grid[pos[0]][pos[1]]==1:
        grid[pos[0]][pos[1]]=color
        if checkConnect(color,dotStart[color],False)==True:
            state="connected"
        else:
            state="valid"

    elif pos in  dotStart.values() or pos in dotEnd.values():
        state="invalid"
            
    elif grid[pos[0]][pos[1]] in [color,color+"1",color+"2"]:
        state="repeat"

    elif grid[pos[0]][pos[1]] in colors:
        grid[pos[0]][pos[1]]=color
        if checkConnect(color,dotStart[color],False)==True:
            state="cutconnected"
        else:
            state="cut"
    print(state)
    return state

        
def checkConnect(color,current,flag,visited=None):
    if visited is None:
        visited = []
        
    if flag==False and current[0]>0 and [current[0]-1,current[1]] not in visited:      #UP
        visited.append(current)
        if grid[current[0]-1][current[1]]==color+"2":
            flag=True
            return flag
        elif grid[current[0]-1][current[1]]==color:
            current=[current[0]-1,current[1]]
            flag=checkConnect(color,current,flag,visited)
        
            
    if flag==False and current[0]<height and [current[0]+1,current[1]] not in visited:     #DOWN
        visited.append(current)
        if grid[current[0]+1][current[1]]==color+"2":
            flag=True
            return flag            
        elif grid[current[0]+1][current[1]]==color:
            current=[current[0]+1,current[1]]
            flag=checkConnect(color,current,flag,visited)

    if flag==False and current[1]>0 and [current[0],current[1]-1] not in  visited:         #LEFT
        visited.append(current)
        if grid[current[0]][current[1]-1]==color+"2":
            flag=True
            return flag
        elif grid[current[0]][current[1]-1]==color:
            current=[current[0],current[1]-1]
            flag=checkConnect(color,current,flag,visited)

                    
    if flag==False and current[1]<length and [current[0],current[1]+1] not in visited:     #RIGHT
        visited.append(current)
        if grid[current[0]][current[1]+1]==color+"2":
            flag=True
            return flag            
        elif grid[current[0]][current[1]+1]==color:
            current=[current[0],current[1]+1]
            flag=checkConnect(color,current,flag,visited)

    return flag

def game():
    global grid,height,length,dotStart,dotEnd,colors
    grid=[["A1"],[1],["A2"]]
    colors=["A"]
    length=len(grid[0])-1
    height=len(grid)-1
    dotStart={"A":[0,0]}
    dotEnd={"A":[0,1]}
    score=0
    complete=False
    turns=0
    while complete==False:
        for a  in  grid:
            print(a)
        color=input("Enter Color:")
        row=int(input("Enter Row:"))
        col=int(input("Enter Column:"))
        pos=[row,col]
        state=addDot(color,pos)
        turns+=1
        
        if  state=="valid":
            score-=0.2
        elif state=="invalid":
            score-=3
        elif state=="repeat":
            score-=2
        elif state=="connected":
            score+=5
        elif state=="cutconnected":
            score+=3

            
        count=0
        for i in  colors:
            if checkConnect(i,dotStart[i],False)==True:
                count+=1
        if count==len(colors):
            complete=True
            score+=15
    print("COMPLETED!!!")
    print("SCORE:",score)
    print("TURNS:",turns)

game()

        
                


        



            

                






















    
