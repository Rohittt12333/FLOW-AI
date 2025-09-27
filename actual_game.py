import numpy as np
import torch
from collections import deque, namedtuple
import torch.nn as nn
import torch.optim as optim
import random
import json
import os
import pickle
import csv


class FLOW:
    def __init__(self,grid_size,max_steps=200):
            self.valid = []
            self.size = grid_size
            self.colors = None
            self.last_action = None
            self.num_colors= grid_size
            self.color_points = None
            self.rows = grid_size
            self.cols = grid_size
            self.area = self.rows * self.cols
            self.action_space = self.area * self.num_colors
            self.max_steps = max_steps
            self.manhattan={}

    def reset(self,puzzle):
        self.color_points = puzzle["color_points"]
        self.colors = puzzle["colors"]
        self.puzzle_no = puzzle["puzzle_no"]
        self.colors.sort()
        self.grid=np.array
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)         
        for i in self.color_points:
            self.grid[i[1][0]][i[1][1]]=i[0]*-1                         #color_points = [[color,[start_x,start_y],[end_x,end_y]]]
        for i in self.color_points:
            self.grid[i[2][0]][i[2][1]]=i[0]*-1
            self.manhattan[i[0]]=abs(i[1][0]-i[2][0])+abs(i[1][1]-i[2][1])      #manhattan={"color":man_distance} 
        #print(self.grid)
        self.valid = self.valid_actions()        
        self.steps = 0
        self.total_connected = 0
        self.solved = 0
        return self.get_observation()


    def step(self, action):
        x, y,color_idx = self.decode_action(action)
        colors = self.colors
        color = colors[color_idx]
        self.steps += 1
        reward = -0.2
        done = False
        info = {}
        count=0
        if self.grid[y,x] < 0:                                  #FIXED POINT OVERWRITE ATTEMPT
            reward += -5.0
            print("OVERWRITE FIXED")
                
        elif self.grid[y, x] == color:                          #REPEATED PLACEMENT
            reward += -1.0
            #print("REPEAT")

        elif action == self.last_action:                        #ACTION REPEAT
            reward += -3.0
            
        elif self.grid[y, x] == 0:                              #VALID
            self.grid[y, x] = color
            reward += 10.0
            flag,i=self.checkConnect(color,[y,x],False)
            if flag==True:
                for i in self.color_points:
                    if i[0]==color:
                        distance = max(abs(i[1][0]-y)+abs(i[1][1]-x),abs(i[2][0]-y)+abs(i[2][1]-x))
                        if self.manhattan[color]>distance:
                            reward += (self.manhattan[color]-distance)*2
                            break
                self.manhattan[color] = distance
            #print("VALID")
            
        else:                                                   #OVERWRITE DOT
            self.grid[y, x] = color
            reward += -1.0
            flag,i=self.checkConnect(color,[y,x],False)
            if flag==True:
                for i in self.color_points:
                    if i[0]==color:
                        distance = max(abs(i[1][0]+i[1][1]-x-y),abs(i[2][0]+i[2][1]-x-y))
                        if self.manhattan[color]>distance:
                            reward += (self.manhattan[color]-distance)*2
                            break
                self.manhattan[color] = distance                
            #print("OVERWRITE DOT")
        #print(y,x,color)
        #print(self.grid)
        self.last_action = action
        connected_cols=0
        total_conn = 0
        for i in  self.color_points:
            flag, path_len = self.checkConnect(i[0],i[1],False)
            connected_cols += path_len
            flag, path_len = self.checkConnect(i[0],i[2],False)
            connected_cols += path_len
            if connected_cols > 10:
                connected_cols = 10 - connected_cols
            total_conn += min(7,connected_cols)
            if flag==True:
                self.manhattan[i[0]]=0
                count+=1

        if count==len(colors):                                          #REWARD FOR FULL SOLUTION
            info['solved'] = True
            print("SOLVED")
            done=True
            reward += 200.0
            
        else:                                                           #REWARD FOR SOLVED COLORS
            reward += (count - self.solved)*5
        reward += (total_conn - self.total_connected)*2                 #REWARD FOR CONNECTED DOTS
        self.total_connected = total_conn
        self.solved = count
        if self.steps >= self.max_steps:
            done = True
            info['solved'] = False

        return self.get_observation(), reward, done, info


    def get_observation(self):
        # Return C x H x W float32; channel c has 1 where color c+1 is present
        abs_grid = np.abs(self.grid)
        obs = np.zeros((self.num_colors, self.rows, self.cols), dtype=np.float32)
        for c in range(self.num_colors):
            obs[c] = (abs_grid == (c + 1)).astype(np.float32)
        return obs


    def checkConnect(self,color,current,flag,visited=None):
        if visited is None:
            i=0
            visited = []
            visited.append(current)
            
        if flag==False and current[0]>0 and [current[0]-1,current[1]] not in visited:      #UP
            if self.grid[current[0]-1][current[1]]==color*-1:
                flag=True
                return flag,len(visited)
            elif self.grid[current[0]-1][current[1]]==color:
                current=[current[0]-1,current[1]]
                visited.append(current)
                flag,i=self.checkConnect(color,current,flag,visited)
                current=[current[0]+1,current[1]]
            
                
        if flag==False and current[0]<(self.rows-1) and [current[0]+1,current[1]] not in visited:     #DOWN
            if self.grid[current[0]+1][current[1]]==color*-1:
                flag=True
                return flag,len(visited)            
            elif self.grid[current[0]+1][current[1]]==color:
                current=[current[0]+1,current[1]]
                visited.append(current)
                flag,i=self.checkConnect(color,current,flag,visited)
                current=[current[0]-1,current[1]]


        if flag==False and current[1]>0 and [current[0],current[1]-1] not in  visited:         #LEFT
            if self.grid[current[0]][current[1]-1]==color*-1:
                flag=True               
                return flag,len(visited)
            elif self.grid[current[0]][current[1]-1]==color:
                current=[current[0],current[1]-1]
                visited.append(current)                
                flag,i=self.checkConnect(color,current,flag,visited)
                current=[current[0],current[1]+1]
                        
        if flag==False and current[1]<(self.cols-1) and [current[0],current[1]+1] not in visited:     #RIGHT
            if self.grid[current[0]][current[1]+1]==color*-1:
                flag=True                
                return flag,len(visited)            
            elif self.grid[current[0]][current[1]+1]==color:
                current=[current[0],current[1]+1]
                visited.append(current)
                flag,i=self.checkConnect(color,current,flag,visited)
                current=[current[0],current[1]-1]
        return flag,len(visited)
                

    def decode_action(self, action):
        color_idx = action % self.num_colors
        pos_idx = action // self.num_colors
        x = pos_idx % self.cols
        y = pos_idx // self.cols
        return x, y, color_idx

    def valid_actions(self):
        valid = []
        for a in range(self.action_space):
            x, y, color = self.decode_action(a)
            val = self.grid[y, x]
            if val >= 0:
                valid.append(a)
        return valid




# ------------- Replay Buffer -------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

def warmup_replay(env, replay, puzzle, n_steps=2000):
    state = env.reset(puzzle)
    for _ in range(n_steps):
        valid = env.valid_actions()
        action = random.choice(valid)
        next_s, r, done, _ = env.step(action)
        replay.push(state, action, r, next_s, float(done))
        state = next_s if not done else env.reset(puzzle)



#//////////////////////////////CNN CODE////////////////////////////////

class CNN(nn.Module):
    def __init__(self, in_channels, action_size, hidden_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, action_size)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = self.pool(h)
        out = self.head(h)
        return out


def train_dqn(
    grid_size=5,
    color_points=None,
    colors=None,
    episodes=100,
    max_steps_per_episode=100,
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    replay_capacity=20000,
    initial_epsilon=1.0,
    min_epsilon=0.1,
    epsilon_decay=0.995,
    target_update_steps=1000,
    device=None,
    start_episodes=0
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = FLOW(grid_size=grid_size, max_steps=max_steps_per_episode)
    in_channels = env.num_colors
    action_size = env.action_space

    policy_net = CNN(in_channels, action_size).to(device)
    target_net = CNN(in_channels, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=replay_capacity)

    total_steps = 0
    puzzles = []
    with open("puzzles.jsonl", "r") as f:
        for line in f:
            puzzles.append(json.loads(line))
    puzzle = random.choice(puzzles)
    warmup_replay(env, replay, puzzle, n_steps=2000)
    start_episode, epsilon, replay = load_checkpoint(policy_net, target_net, optimizer, replay)
    losses=[]
    q_means=[]
    q_maxs=[]

    for ep in range(start_episode, episodes + 1):
      
        if ep>2500 and ep%50 == 0:
            puzzle = random.choice(puzzles)

        elif ep>1000 and ep%100 == 0:
            puzzle = random.choice(puzzles)
      
        elif ep%250 == 0:
            puzzle = random.choice(puzzles)
        state = env.reset(puzzle)  # C x H x W
        ep_reward = 0.0

        for t in range(max_steps_per_episode):
            total_steps += 1
            # select action (epsilon-greedy)
            if random.random() < epsilon:
                #print("RANDOM")
                action = random.choice(env.valid)

            else:
                st = torch.from_numpy(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q = policy_net(st).squeeze(0)
                    mask = torch.full_like(q, -float("inf"))  # start with -inf everywhere
                    mask[env.valid] = q[env.valid]                    # keep only valid actions

                    action = int(torch.argmax(mask).item())
                    q_mean = q.mean().item()
                    q_max = q.max().item()
                    q_means.append(q_mean)
                    q_maxs.append(q_max)



            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            replay.push(state, action, reward, next_state, float(done))
            state = next_state

            # learning step
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                states = np.stack(batch.state)       # B x C x H x W
                next_states = np.stack(batch.next_state)
                actions = np.array(batch.action, dtype=np.int64)
                rewards = np.array(batch.reward, dtype=np.float32)
                dones = np.array(batch.done, dtype=np.float32)

                states_t = torch.from_numpy(states).to(device)
                next_states_t = torch.from_numpy(next_states).to(device)
                actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(device)
                rewards_t = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
                dones_t = torch.from_numpy(dones).float().unsqueeze(1).to(device)

                # Q(s,a)
                q_values = policy_net(states_t).gather(1, actions_t)

                # target: r + gamma * max_a' Q_target(s', a') * (1-done)
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(1)[0].unsqueeze(1)
                    target_q = rewards_t + gamma * max_next_q * (1.0 - dones_t)

                loss = nn.SmoothL1Loss()(q_values, target_q)
                loss_value = loss.item()
                losses.append(loss_value)   # keep a list or deque of recent losses
               

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            # update target network periodically
            if total_steps % target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)


        if ep % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            avg_q_mean = np.mean(q_means[-100:]) if q_means else 0.0
            avg_q_max = np.mean(q_maxs[-100:]) if q_maxs else 0.0
            with open("log.csv",mode="a",newline="", encoding="utf-8") as logs:
                writer=csv.writer(logs)
                writer.writerow([avg_loss,avg_q_mean,avg_q_max])

        if ep % 500 == 0:
            save_checkpoint(policy_net, target_net, optimizer, replay, ep, epsilon)
        print(f"Ep {ep:4d} | steps {total_steps:6d} | ep_reward {ep_reward:6.1f} | eps {epsilon:.3f} | Puzzle_No {env.puzzle_no:4d}")
    return policy_net,env


# ------------- Save / Load helpers -------------
def save_checkpoint(policy_net, optimizer, episode, filename="checkpoint.pth"):
    path = os.path.join("checkpoints", filename)
    checkpoint = {
        'episode': episode,
        'policy_state': policy_net.state_dict(),
        'target_state': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epsilon': epsilon,
        'replay_buffer': replay.buffer
        }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"✅ Saved checkpoint at episode {episode}")

def load_checkpoint(policy_net, optimizer, filename="checkpoint.pth"):
    path = os.path.join("checkpoints", filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        policy_net.load_state_dict(checkpoint['policy_state'])
        target_net.load_state_dict(checkpoint['target_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epsilon = checkpoint.get('epsilon',1.0)
        replay.buffer = checkpoint['replay_buffer']
        start_episode = checkpoint['episode'] + 1
        print(f"🔄 Resuming from episode {start_episode}")
        return start_episode,epsilon
    return 1,1.0

def save_checkpoint(policy_net, target_net, optimizer, replay, episode, epsilon, filename="checkpoint.pth"):
    os.makedirs("checkpoints", exist_ok=True)
    path = os.path.join("checkpoints", filename)
    checkpoint = {
        'episode': episode,
        'policy_state': policy_net.state_dict(),
        'target_state': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epsilon': epsilon,
        'replay_buffer': replay.buffer 
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"✅ Saved checkpoint at episode {episode}")


def load_checkpoint(policy_net, target_net, optimizer, replay, filename="checkpoint.pth"):
    path = os.path.join("checkpoints", filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        policy_net.load_state_dict(checkpoint['policy_state'])
        target_net.load_state_dict(checkpoint['target_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epsilon = checkpoint.get('epsilon', 1.0)
        replay.buffer = checkpoint['replay_buffer']
        start_episode = checkpoint['episode'] + 1
        print(f"🔄 Resuming from episode {start_episode}")
        return start_episode, epsilon, replay
    return 1, 1.0, replay

# ------------- Example usage -------------
if __name__ == "__main__":
    # small example color pairs for 5x5:
    policy, env = train_dqn(grid_size=5, episodes=5000, max_steps_per_episode=100)
    # test policy after training
    obs = env.reset()
    done = False
    while not done:
        s = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            act = int(torch.argmax(policy(s)).item())
        obs, r, done, info = env.step(act)
        if done:
            print("Done:", info)
            break

    

