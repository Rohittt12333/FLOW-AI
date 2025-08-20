import numpy as np
import torch
from collections import deque, namedtuple
import torch.nn as nn
import torch.optim as optim
class FLOW:
    def __init__(self,grid_size,grid_colors,grid_rows,grid_cols,color_points,max_steps=200):
            self.size = grid_size
            self.colors = grid_colors
            self.color_points = color_points
            self.rows = grid_rows
            self.cols = grid_cols
            self.area = self.grid_rows * self.grid_cols
            self.action_space = self.area * len(self.colors)
            self.max_steps = max_steps
            self.reset()

    def reset(self):
        self.grid=np.array
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int8)         
        for i in self.color_points:
            self.grid[i[1][0]][i[1][1]]=i[0]*-1            #color_points = [[color,[start_x,start_y],[end_x,end_y]]]
        for i in self.color_points:
            self.grid[i[2][0]][i[2][1]]=i[0]*-1

        print(self.grid)
        self.steps = 0
        self.solved=0
        return self.get_observation()


    def step(self, action):
        x, y,color = self.decode_action(action)
        self.steps += 1
        reward = 0.0
        done = False
        info = {}
        if self.grid[y,x]==(color*-1):                                  #FIXED POINT OVERWRITE ATTEMPT
            reward = -5.0
                
        if self.grid[y, x] == color:                                    #REPEATED PLACEMENT
            reward = -0.2
            
        elif self.grid[y, x] == 0:                                      #VALID
            self.grid[y, x] = color
            reward = 1.0
            
        else:                                                           #OVERWRITE DOT
            self.grid[y, x] = color
            reward = -3.0

        for i in  colors:
            if self.checkConnect(i,dotStart[i],False)==True:
                count+=1
                
        if count==len(colors):                                          #REWARD FOR FULL SOLUTION
            info['solved'] = True
            done=True
            reward += 50.0
            
        else:                                                           #REWARD FOR SOLVED COLORS
            reward += (count - self.solved)*5
            
        self.solved = count

        if self.steps >= self.max_steps:
            done = True
            info['solved'] = False

        return self.get_observation(), reward, done, info


    def get_observation(self):
        # Return C x H x W float32; channel c has 1 where color c+1 is present
        abs_grid = np.abs(self.grid)
        obs = np.zeros((self.num_colors, self.grid_rows, self.grid_cols), dtype=np.float32)
        for c in range(self.num_colors):
            obs[c] = (abs_grid == (c + 1)).astype(np.float32)
        return obs


    def checkConnect(self,color,current,flag,visited=None):
        if visited is None:
            visited = []
            
        if flag==False and current[0]>0 and [current[0]-1,current[1]] not in visited:      #UP
            visited.append(current)
            if grid[current[0]-1][current[1]]==color*-1:
                flag=True
                return flag
            elif grid[current[0]-1][current[1]]==color:
                current=[current[0]-1,current[1]]
                flag=checkConnect(color,current,flag,visited)
            
                
        if flag==False and current[0]<(self.rows-1) and [current[0]+1,current[1]] not in visited:     #DOWN
            visited.append(current)
            if grid[current[0]+1][current[1]]==color*-1:
                flag=True
                return flag            
            elif grid[current[0]+1][current[1]]==color:
                current=[current[0]+1,current[1]]
                flag=checkConnect(color,current,flag,visited)

        if flag==False and current[1]>0 and [current[0],current[1]-1] not in  visited:         #LEFT
            visited.append(current)
            if grid[current[0]][current[1]-1]==color*-1:
                flag=True
                return flag
            elif grid[current[0]][current[1]-1]==color:
                current=[current[0],current[1]-1]
                flag=checkConnect(color,current,flag,visited)

                        
        if flag==False and current[1]<(self.cols-1) and [current[0],current[1]+1] not in visited:     #RIGHT
            visited.append(current)
            if grid[current[0]][current[1]+1]==color*-1:
                flag=True
                return flag            
            elif grid[current[0]][current[1]+1]==color:
                current=[current[0],current[1]+1]
                flag=checkConnect(color,current,flag,visited)

        return flag



    def decode_action(self, action):
        color_idx = action % self.num_colors
        pos_idx = action // self.num_colors
        x = pos_idx % self.grid_w
        y = pos_idx // self.grid_w
        return x, y, color_idx



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



#//////////////////////////////CNN CODE////////////////////////////////

class CNNDQN(nn.Module):
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


def train_cnn_dqn(
    grid_size=6,
    color_pairs=None,
    episodes=200,
    max_steps_per_episode=200,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    replay_capacity=20000,
    initial_epsilon=1.0,
    min_epsilon=0.05,
    epsilon_decay=0.995,
    target_update_steps=1000,
    device=None,
    save_path="cnn_dqn_checkpoint.pth"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = PuzzleEnv(grid_size=grid_size, color_pairs=color_pairs, max_steps=max_steps_per_episode)
    state_size = env.area  # flattened grid only
    action_size = env.action_space

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=replay_capacity)

    epsilon = initial_epsilon
    total_steps = 0

    for ep in range(1, episodes+1):
        state = env.reset()
        ep_reward = 0.0

        for t in range(max_steps_per_episode):
            total_steps += 1
            st = torch.from_numpy(state).float().unsqueeze(0).to(device)

            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    qvals = policy_net(st)
                    action = int(torch.argmax(qvals, dim=1).item())

            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            replay.push(state, action, reward, next_state, float(done))
            state = next_state

            # Learn
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
                actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
                dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)

                # Current Q-values
                q_values = policy_net(states).gather(1, actions)

                # Next Q-values from target
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + gamma * next_q_values * (1.0 - dones)

                loss = nn.MSELoss()(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            # Update target network periodically
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep:4d} | steps {total_steps:6d} | ep_reward {ep_reward:6.1f} | epsilon {epsilon:.3f}")

        # Save checkpoint occasionally
        if ep % 100 == 0:
            torch.save({
                'episode': ep,
                'policy_state_dict': policy_net.state_dict(),
                'target_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }, save_path)

    # final save
    torch.save({
        'episode': episodes,
        'policy_state_dict': policy_net.state_dict(),
        'target_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
    }, save_path)
    print("Training finished. Model saved to", save_path)
    return policy_net, env

# ------------- Save / Load helpers -------------
def save_model(net, path):
    torch.save(net.state_dict(), path)

def load_model(net_class, path, *args, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net_class(*args).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net

# ------------- Example usage -------------
if __name__ == "__main__":
    # small example color pairs for 5x5:
    cp = [(0,0,4,4), (4,0,0,4)]  # two colors
    policy, env = train_dqn(grid_size=5, color_pairs=cp, episodes=300, max_steps_per_episode=100)
    # test policy after training
    obs = env.reset()
    env.render()
    done = False
    while not done:
        s = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            act = int(torch.argmax(policy(s)).item())
        obs, r, done, info = env.step(act)
        env.render()
        if done:
            print("Done:", info)
            break

    

