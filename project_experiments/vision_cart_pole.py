import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
from tqdm import tqdm
import vdp
import time
import os, shutil
from random import randint
folder = 'project_exp_figs'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Wandb
#import wandb
#wandb.init(project="Cartpole Vision RL")

############ HYPERPARAMETERS ###############
BATCH_SIZE = 128 # original = 128
#env_batch_size = 256
env_epochs = 30#30
env_loss_ratio = 0.5 # recon_loss = img_loss + env_loss_ratio*reward_loss
kl_ratio = 0.05 # loss = recon_loss + kl_ratio+kl_dif_loss
env_lr = 0.001
env_decay = 0.99
GAMMA = 0.999 # original = 0.999
EPS_START = 0.9 # original = 0.9
EPS_END = 0.01 # original = 0.05
EPS_DECAY = 3000 # original = 200
TARGET_UPDATE = 50 # original = 10
MEMORY_SIZE = 50000 # original = 10000
END_SCORE = 200 # 200 for Cartpole-v0
TRAINING_STOP = 142 # threshold for training stop
N_EPISODES = 50000 # total episodes to be run
LAST_EPISODES_NUM = 20 # number of episodes for stopping training
FRAMES = 2 # state is the number of last frames: the more frames, 
# the more the state is detailed (still Markovian)
RESIZE_PIXELS = 60 # Downsample image to this number of pixels

# ---- CONVOLUTIONAL NEURAL NETWORK ----
HIDDEN_LAYER_1 = 64
HIDDEN_LAYER_2 = 64 
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5 # original = 5
STRIDE = 2 # original = 2
latent_dims = 15
action_latents = 10
deconv_ch2 = 128
deconv_ch1 = 64
# --------------------------------------

GRAYSCALE = True # False is RGB
LOAD_MODEL = False # If we want to load the model, Default= False
USE_CUDA = True # If we want to use GPU (powerful one needed!)
############################################

graph_name = 'cartpole_vision'
device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

# Settings for GRAYSCALE / RGB
if GRAYSCALE == 0:
    resize = T.Compose([T.ToPILImage(), 
                    T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                    T.ToTensor()])
    
    nn_inputs = 3*FRAMES  # number of channels for the nn
else:
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                    T.Grayscale(),
                    T.ToTensor()])
    nn_inputs =  FRAMES # number of channels for the nn

                    
stop_training = False 

env = gym.make("CartPole-v0", render_mode='rgb_array').unwrapped 
print(env.render_mode)

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#plt.ion()

# If gpu is to be used

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
env_transition = namedtuple('Env_Transition',
                        ('state', 'action', 'next_state', 'reward', 'state_vars'))

# Memory for Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None) # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = Transition(*args)  
        self.position = (self.position + 1) % self.capacity # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) 

    def __len__(self): 
        return len(self.memory)
    
class EnvMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None) # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = env_transition(*args)  
        self.position = (self.position + 1) % self.capacity # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) 

    def __len__(self): 
        return len(self.memory)

# Build CNN
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE) 
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = KERNEL_SIZE, stride = STRIDE):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
class D_AutoEncoder(nn.Module):

    def __init__(self, h, w, latent_dim, actions, action_latents):#""", state_vars):"""
        super(D_AutoEncoder, self).__init__()
        self.conv1 = vdp.Conv2d(nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE, input_flag=True) 
        self.bn1 = vdp.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = vdp.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = vdp.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = vdp.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn3 = vdp.BatchNorm2d(HIDDEN_LAYER_3)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = KERNEL_SIZE, stride = STRIDE):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.convh = convh
        self.convw = convw
        nn.Dropout()
        self.latentize = vdp.Linear(linear_input_size, latent_dim)
        #self.latent_mu = vdp.Linear(linear_input_size, latent_dim)
        #self.latent_sigma = vdp.Linear(linear_input_size, latent_dim)
        self.action_encode = vdp.Linear(actions, action_latents, input_flag=True)
        self.un_latentize = vdp.Linear(latent_dim + action_latents, linear_input_size)
        self.tconv3 = vdp.ConvTranspose2d(HIDDEN_LAYER_3, deconv_ch2, kernel_size=KERNEL_SIZE, stride=STRIDE, output_padding=[1,0])
        self.tconv2 = vdp.ConvTranspose2d(deconv_ch2, deconv_ch1, kernel_size=KERNEL_SIZE, stride=STRIDE, output_padding=[1,1])
        self.tconv1 = vdp.ConvTranspose2d(deconv_ch1, nn_inputs, kernel_size=KERNEL_SIZE, stride=STRIDE, output_padding=[1,0])
        self.reward_decode = vdp.Linear(linear_input_size, n_env_vars)
        self.gelu = vdp.GELU()
        self.sigmoid = vdp.Sigmoid()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def encode(self, x):
        #print(x.shape)
        mu, sigma = self.gelu(*self.bn1(*self.conv1(x)))
        mu, sigma = self.gelu(*self.bn2(*self.conv2(mu, sigma)))
        mu, sigma = self.gelu(*self.bn3(*self.conv3(mu, sigma)))
        #mu = self.latent_mu(x.view(x.size(0), -1))
        #log_sigma = self.latent_sigma(x.view(x.size(0), -1))
        mu, sigma = self.latentize(mu.view(x.size(0), -1), sigma.view(x.size(0), -1))
        return mu, sigma
    
    def reparemetrize(self, mu, sigma):
        epsilon = torch.randn_like(sigma).to(device)
        return mu + epsilon * sigma


    def decode(self, mu, sigma):
        mu, sigma = self.gelu(*self.un_latentize(mu, sigma))
        r_mu, r_sigma = self.reward_decode(mu, sigma)
        mu, sigma = self.gelu(*self.tconv3(mu.view(mu.size(0), -1, self.convh, self.convw), 
                                   sigma.view(sigma.size(0), -1, self.convh, self.convw)))
        mu, sigma = self.gelu(*self.tconv2(mu, sigma))
        mu, sigma = self.tconv1(mu, sigma)
        stuff = self.sigmoid(mu, sigma)
        mu, sigma = stuff
        
        return mu, sigma, r_mu.view(-1, n_env_vars), r_sigma.view(-1, n_env_vars)
    
    def forward(self, x, a):
        mu, sigma = self.encode(x)
        #z = self.reparemetrize(mu, torch.exp(0.5*log_sigma))
        #z = mu
        a = torch.cat((a, torch.ones_like(a)-a), dim=1).float()
        a_mu, a_sigma = self.gelu(*self.action_encode(a))
        scr_mu, scr_sigma, sv_mu, sv_sigma = self.decode(torch.cat((mu, a_mu), dim=1), torch.cat((sigma, a_sigma), dim=1))
        return scr_mu, scr_sigma, sv_mu, sv_sigma
    
    def get_kl(self):
        return sum(vdp.gather_kl(self))
    
    
# Cart location for centering image crop
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

# Cropping, downsampling (and Grayscaling) image
def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

env.reset()
"""plt.figure()
if GRAYSCALE == 0:
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
else:
    plt.imshow(get_screen().cpu().squeeze(0).permute(
        1, 2, 0).numpy().squeeze(), cmap='gray')
plt.title('Example extracted screen')"""
#plt.ioff()

env.close()

eps_threshold = 0.9 # original = 0.9

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
print("Screen height: ", screen_height," | Width: ", screen_width)

# Get number of actions from gym action space
n_actions = env.action_space.n
n_env_vars = 4


steps_done = 0

# Action selection , if stop training == True, only exploitation
def select_action(state, stop_training):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
# one for the both data types network
def bd_select_action(state, stop_training):
    global bd_steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * bd_steps_done / EPS_DECAY)
    bd_steps_done += 1
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            
            return bd_policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
# one for the both data types network
def batch_select_action(state, stop_training):
    acts = torch.randint(0,n_actions, (state.shape[0],1)).to(device)
    if stop_training:
        acts = bd_policy_net(state).max(1)[1].view(-1, 1)
    else:
        global bd_steps_done
        sample = torch.rand(state.shape[0])
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * bd_steps_done / EPS_DECAY)
        # print('Epsilon = ', eps_threshold, end='\n')
        ids = torch.where(sample > eps_threshold)

        acts[ids] = bd_policy_net(state[ids]).max(1)[1].view(-1, 1)
    
    return acts

# single and batched using env net to guide exploration
def g_select_action(state, stop_training):
    global g_steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * g_steps_done / EPS_DECAY)
    g_steps_done += 1
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            
            act = g_policy_net(state).max(1)[1].view(1, 1)
            #print(f"exploit {act}")
            return act
    else:
        possible_actions = torch.arange(n_actions).view(-1,1).to(device)
        _, scr_sigma, _, sv_sigma = env_net(state.repeat(n_actions,1,1,1), possible_actions)
        tot_sig = scr_sigma.mean((1,2,3)) + sv_sigma.mean(1) # dimensions that aren't the batch id
        #print(f"explore {possible_actions[torch.argmax(tot_sig)]}")
        return possible_actions[torch.argmax(tot_sig)].view(1,1)


    

def g_batch_select_action(state, stop_training):
    acts = torch.randint(0,n_actions, (state.shape[0],1)).to(device)
    if stop_training:
        acts = g_policy_net(state).max(1)[1].view(-1, 1)
    else:
        global g_steps_done
        sample = torch.rand(state.shape[0])
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * g_steps_done / EPS_DECAY)
        # print('Epsilon = ', eps_threshold, end='\n')
        ids = torch.where(sample > eps_threshold)

        acts[ids] = g_policy_net(state[ids]).max(1)[1].view(-1, 1)
    
    return acts

# Training 
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # torch.cat concatenates tensor sequence
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).type(torch.FloatTensor).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #plt.figure(2)

    #wandb.log({'Loss:': loss})

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    #if (i_episode%50==0):
    #    print(f"regular score: {episode_durations[-1]}")

def optimize_model_with_fake_data(fake_screens=None, fake_actions=None, fake_next_screens=None, fake_rewards=None):
    if (len(bd_memory) < BATCH_SIZE):
        return

    if  not (fake_screens is None):
        #print("used_fake_data")
        
        if fake_screens.size(0) > BATCH_SIZE:
            perm = torch.randperm(fake_screens.size(0))
            idx = perm[:BATCH_SIZE]
            fake_next_states = fake_next_screens[idx]
            fake_state_batch = fake_screens[idx]
            fake_action_batch = fake_actions[idx]
            fake_reward_batch = fake_rewards[idx]
        else:
            fake_next_states = fake_next_screens
            fake_state_batch = fake_screens
            fake_action_batch = fake_actions
            fake_reward_batch = fake_rewards

        
        fake_state_action_values = bd_policy_net(fake_state_batch).gather(1, fake_action_batch)

        #fake_next_state_values = torch.zeros(BATCH_SIZE, device=device)
        fake_next_state_values = bd_target_net(fake_next_states).max(1)[0].detach()
        # Compute the expected Q values
        #print(fake_next_state_values.shape, fake_next_states.shape, fake_reward_batch.shape)
        fake_expected_state_action_values = (fake_next_state_values * GAMMA) + fake_reward_batch

        # Compute Huber loss
        bd_loss = F.smooth_l1_loss(fake_state_action_values, fake_expected_state_action_values.unsqueeze(1))
        # Optimize the model
        bd_optim.zero_grad()
        bd_loss.backward()
        for param in bd_policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        bd_optim.step()

    if (len(bd_memory) >= BATCH_SIZE):
        transitions = bd_memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        # torch.cat concatenates tensor sequence
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).type(torch.FloatTensor).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = bd_policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = bd_target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        bd_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        bd_optim.zero_grad()
        bd_loss.backward()
        for param in bd_policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        bd_optim.step()

    if (i_episode%50==0):
        print(f"bd score: {bd_episode_durations[-1]}")

def optimize_guided_model(fake_screens=None, fake_actions=None, fake_next_screens=None, fake_rewards=None):
    if (len(g_memory) < BATCH_SIZE):
        return

    if  not (fake_screens is None):
        #print("used_fake_data")
        
        if fake_screens.size(0) > BATCH_SIZE:
            perm = torch.randperm(fake_screens.size(0))
            idx = perm[:BATCH_SIZE]
            fake_next_states = fake_next_screens[idx]
            fake_state_batch = fake_screens[idx]
            fake_action_batch = fake_actions[idx]
            fake_reward_batch = fake_rewards[idx]
        else:
            fake_next_states = fake_next_screens
            fake_state_batch = fake_screens
            fake_action_batch = fake_actions
            fake_reward_batch = fake_rewards

        
        fake_state_action_values = g_policy_net(fake_state_batch).gather(1, fake_action_batch)

        #fake_next_state_values = torch.zeros(BATCH_SIZE, device=device)
        fake_next_state_values = g_target_net(fake_next_states).max(1)[0].detach()
        # Compute the expected Q values
        #print(fake_next_state_values.shape, fake_next_states.shape, fake_reward_batch.shape)
        fake_expected_state_action_values = (fake_next_state_values * GAMMA) + fake_reward_batch

        # Compute Huber loss
        g_loss = F.smooth_l1_loss(fake_state_action_values, fake_expected_state_action_values.unsqueeze(1))
        # Optimize the model
        g_optim.zero_grad()
        g_loss.backward()
        for param in g_policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        g_optim.step()

    if (len(g_memory) >= BATCH_SIZE):
        transitions = g_memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        # torch.cat concatenates tensor sequence
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).type(torch.FloatTensor).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = g_policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = g_target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        g_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        g_optim.zero_grad()
        g_loss.backward()
        for param in g_policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        g_optim.step()

    if (i_episode%50==0):
        print(f"guided score: {g_episode_durations[-1]}")


def optimize_env_model(guided=False):
    if len(env_memory) < BATCH_SIZE:
        return
    
    env_net.train()

    env_optim.defaults['lr'] = env_lr
    #s2 = time.time()
    for _ in range(env_epochs):
        transitions = env_memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = env_transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        # torch.cat concatenates tensor sequence
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_variable_batch = torch.cat(batch.state_vars)
        
        predicted_next_states, scr_sigma, predicted_vars, sv_sigma = env_net(state_batch[non_final_mask], \
                                                                       action_batch[non_final_mask])
        

        
        #loss = F.smooth_l1_loss(predicted_next_states, non_final_next_states)
        #loss += env_loss_ratio*F.smooth_l1_loss(predicted_vars, state_variable_batch[non_final_mask])
        loss = -torch.distributions.normal.Normal(predicted_next_states, 
                                vdp.softplus(scr_sigma)).log_prob(non_final_next_states).sum()
        #print(loss.item())
        loss += -env_loss_ratio*torch.distributions.normal.Normal(predicted_vars, 
                                vdp.softplus(sv_sigma)).log_prob(state_variable_batch[non_final_mask]).sum()
        #print(loss.item())
        loss += kl_ratio*env_net.get_kl()
        #print("a",loss.item())
        
        #loss -= kl_ratio*0.5 * torch.sum(1+ log_sigma - mu.pow(2) - log_sigma.exp())
        #print("a",loss.item())
        
        env_optim.zero_grad()
        loss.backward()
        env_optim.step()
        env_optim.defaults['lr'] *= env_decay
        torch.cuda.empty_cache()
    #print(f"actual training: {time.time() - s2}")

    #print(torch.max(non_final_next_states), torch.min(non_final_next_states))
    if (i_episode%25==0):
        pic_id = randint(0, predicted_next_states.shape[0]-1)
        x = predicted_vars[pic_id,0].item()
        theta = predicted_vars[pic_id,2].item()
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        rw = r1 + r2

        tot_sigma = scr_sigma[pic_id].mean() + sv_sigma[pic_id].mean()
        k_s = torch.floor(torch.clip(7-40*tot_sigma, 0, 4)).item()

        fig, ax = plt.subplots(2,2)
        if GRAYSCALE == 0:
            ax[0,0].imshow(predicted_next_states[pic_id,0].cpu().numpy(),
                    interpolation='none')
            ax[0,0].set_title("pred0")
            ax[1,0].imshow(non_final_next_states[pic_id,0].cpu().numpy(),
                    interpolation='none')
            ax[1,0].set_title("real0")
            ax[0,1].imshow(predicted_next_states[pic_id,1].cpu().numpy(),
                    interpolation='none')
            ax[0,1].set_title("pred1")
            ax[1,1].imshow(non_final_next_states[pic_id,1].cpu().numpy(),
                    interpolation='none')
            ax[1,1].set_title("real1")
        else:
            ax[0,0].imshow(predicted_next_states[pic_id,0].detach().cpu().numpy(), 
                    cmap='gray')
            ax[0,0].set_title("pred0")
            ax[1,0].imshow(non_final_next_states[pic_id,0].detach().cpu().numpy(), 
                    cmap='gray')
            ax[1,0].set_title("real0")
            ax[0,1].imshow(predicted_next_states[pic_id,1].detach().cpu().numpy(), 
                    cmap='gray')
            ax[0,1].set_title("pred1")
            ax[1,1].imshow(non_final_next_states[pic_id,1].detach().cpu().numpy(), 
                    cmap='gray')
            ax[1,1].set_title("real1")
        fig.suptitle(
            f"""Rewards: real:{reward_batch[non_final_mask][pic_id].round(decimals=4)}, fake: {round(rw,4)}
            \nvariance: {tot_sigma}, k_star: {k_s}""")
        fig.savefig(f"project_exp_figs/screen{j}_{i_episode}.png")
        plt.close(fig)



    env_net.eval()
    transitions = env_memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = env_transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # torch.cat concatenates tensor sequence
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_variable_batch = torch.cat(batch.state_vars)

    predicted_next_states, scr_sigma, predicted_vars, sv_sigma = env_net(state_batch[non_final_mask], \
                                                                    action_batch[non_final_mask])

    tot_sigma = scr_sigma.mean(dim=(1,2,3)) + sv_sigma.mean(dim=1)
    k_s = torch.floor(torch.clip(7-40*tot_sigma, 0, 4))

    fake_screens = None
    fake_actions = None
    fake_next_screens = None
    fake_rewards = None
    while torch.any(k_s > 0):
        #print("made_fake_data")
        ids = torch.where(k_s > 0)
        k_s -= 1

        if fake_screens is None:
            fake_screens = state_batch[non_final_mask][ids].detach()
            fake_actions = action_batch[non_final_mask][ids].detach()
            fake_next_screens = predicted_next_states[ids].detach()
            
            # Reward modification for better stability
            x = predicted_vars[ids][:,0].detach()
            #print(x.shape)
            theta = predicted_vars[ids][:,2].detach()
            r1 = (env.x_threshold - torch.abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - torch.abs(theta)) / env.theta_threshold_radians - 0.5
            fake_rewards = r1 + r2
            #reward = torch.tensor([reward], device=device)
            """if t >= END_SCORE-1:
                reward = reward + 20
                done = 1
            else: 
                if done:
                    reward = reward - 20 """
            
        else:
            fake_screens = torch.cat([fake_screens, predicted_states[ids].detach()])
            fake_actions = torch.cat([fake_actions, fake_action[ids].detach()])
            fake_next_screens = torch.cat([fake_next_screens, predicted_next_states[ids].detach()])
            
            # Reward modification for better stability
            x = predicted_vars[ids][:,0].detach()
            theta = predicted_vars[ids][:,2].detach()
            r1 = (env.x_threshold - torch.abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - torch.abs(theta)) / env.theta_threshold_radians - 0.5
            fake_rewards = torch.cat([fake_rewards,r1 + r2])
            

        predicted_states = predicted_next_states
        if guided:
            fake_action = g_batch_select_action(predicted_states, False)
        else:
            fake_action = batch_select_action(predicted_states, False)
        predicted_next_states, _, predicted_vars, _ = env_net(predicted_states, fake_action)

    #if not (fake_screens is None):
    #    print(fake_screens.shape, fake_actions.shape, fake_next_screens.shape, fake_rewards.shape)

    return (fake_screens, fake_actions, fake_next_screens, fake_rewards)
    # Optimize the model

episodes_trajectories = []
episodes_after_stop = 100

runs = 3


folder = 'project_episode_durations'
last_run = 0
filenames = os.listdir(folder)

for a in range(runs):
    if f"regular{a}.csv" in filenames:
        last_run = a+1
        file_path = os.path.join(folder, f"regular{a}.csv")
        episodes_trajectories.append(np.loadtxt(file_path, delimiter=',').tolist())


    

# MAIN LOOP
stop_training = False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~real~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for j in range(last_run, runs):
    print(j)
    mean_last = deque([0] * LAST_EPISODES_NUM, LAST_EPISODES_NUM)
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(MEMORY_SIZE)
    
    count_final = 0
    
    steps_done = 0
    episode_durations = []

    for i_episode in tqdm(range(N_EPISODES)):
        env.reset()
        init_screen = get_screen()
        screens = deque([init_screen] * FRAMES, FRAMES)
        state = torch.cat(list(screens), dim=1)
        
        #s1 = time.time()
        for t in count():        
            # Select and perform an action
            action = select_action(state, stop_training)
            state_variables, _, done, _, _ = env.step(action.item())

            # Observe new state
            screens.append(get_screen())
            next_state = torch.cat(list(screens), dim=1) if not done else None

            # Reward modification for better stability
            x, x_dot, theta, theta_dot = state_variables
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            reward = torch.tensor([reward], device=device)
            if t >= END_SCORE-1:
                reward = reward + 20
                done = 1
            else: 
                if done:
                    reward = reward - 20 

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the target network)
            if done:
                episode_durations.append(t + 1)
                mean_last.append(t + 1)
                mean = 0
                #wandb.log({'Episode duration': t+1 , 'Episode number': i_episode})
                for i in range(LAST_EPISODES_NUM):
                    mean = mean_last[i] + mean
                mean = mean/LAST_EPISODES_NUM
                if mean < TRAINING_STOP and stop_training == False:
                    #print(f"\nregular exploration: {time.time()-s1}")
                    #s1 = time.time()
                    torch.cuda.empty_cache()
                    optimize_model()
                    #print(f"\nregular training: {time.time()-s1}")
                    
                else:
                    stop_training = True
                    #print("regular boy did it")
                break



        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            #bd_target_net.load_state_dict(bd_policy_net.state_dict())
        if stop_training == True:
            count_final += 1
            
        #if bd_stop_training == True:
        #    bd_count_final += 1
        
        if stop_training:# and bd_stop_training:
            if (count_final >= 100):# and (bd_count_final >= 100):
                break

    print('Complete')
    stop_training = False
    #bd_stop_training = False
    episodes_trajectories.append(episode_durations)
    episode_durations = np.array(episode_durations)
    np.savetxt(f"project_episode_durations/regular{j}.csv", episode_durations, delimiter=',')
    #bd_episodes_trajectories.append(bd_episode_durations)

# Cherry picking best runs
best = []
for et in episodes_trajectories:
    best.append(et)

maximum = 0
for i in range(len(best)):
    maximum = max(len(best[i]), maximum)
    
# Fill the episodes to make them the same length
for i in range(len(best)):
    length = len(best[i])
    for j in range(maximum - len(best[i])):
        best[i].append(best[i][j+length-100])
    best[i] = np.asarray(best[i])
    
best = np.asarray(best)


# To numpy
score_mean = np.zeros(maximum)
score_std = np.zeros(maximum)
last100_mean = np.zeros(maximum)
print(best[:, max(0, -99):1].mean())
for i in range(maximum):
    score_mean[i]  = best[:, i].mean()
    score_std[i] = best[:, i].std()
    last100_mean[i] = best[:, max(0, i-50):min(maximum, i+50)].mean()
print(len(last100_mean))

t = np.arange(0, maximum, 1)
# from scipy.interpolate import make_interp_spline # make smooth version
# interpol = make_interp_spline(t, score_mean, k=3)  # type: BSpline

fig, ax = plt.subplots(3, figsize=(8, 6))
ax[0].fill_between(t, np.maximum(score_mean - score_std, 0),
                np.minimum(score_mean + score_std, END_SCORE), color='b', alpha=0.2, label='Real Data Variance')
# ax.legend(loc='upper right')
"""ax[0].set_xlabel('Episode')
ax[0].set_ylabel('Score')"""
# ax.set_title('Inverted Pendulum Training Plot from Pixels')
ax[0].plot(t, score_mean, color='b', label='Real Data Only')
#ax.plot(t, last100_mean, color='purple', linestyle='dotted', label='Smoothed mean')
ax[0].legend()

try:
    del(t, last100_mean, best, score_mean, score_std, stop_training, episodes_trajectories, episode_durations, memory, 
        target_net, policy_net, optimizer, mean_last)
except:
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~both data types~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
bd_episodes_trajectories = []


last_run = 0

for a in range(runs):
    if f"bd{a}.csv" in filenames:
        last_run = a+1
        file_path = os.path.join(folder, f"bd{a}.csv")
        bd_episodes_trajectories.append(np.loadtxt(file_path, delimiter=',').tolist())

bd_stop_training = False
for j in range(last_run,runs):
    print("bd",j)
    env_net = D_AutoEncoder(screen_height, screen_width, latent_dims, n_actions, action_latents).to(device)
    bd_mean_last = deque([0] * LAST_EPISODES_NUM, LAST_EPISODES_NUM)
    bd_policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    bd_target_net = DQN(screen_height, screen_width, n_actions).to(device)
    bd_target_net.load_state_dict(bd_policy_net.state_dict())
    bd_target_net.eval()

    env_memory = EnvMemory(MEMORY_SIZE)
    env_optim = optim.Adam(env_net.parameters(), env_lr)
    
    bd_optim = optim.RMSprop(bd_policy_net.parameters())
    bd_memory = ReplayMemory(MEMORY_SIZE)
    bd_count_final = 0
    
    bd_steps_done = 0
    bd_episode_durations = []
    
    for i_episode in tqdm(range(N_EPISODES)):
        if not bd_stop_training:
            env.reset()
            init_screen = get_screen()
            screens = deque([init_screen] * FRAMES, FRAMES)
            state = torch.cat(list(screens), dim=1)

            fake_screens = deque([init_screen] * FRAMES, FRAMES)
            fake_state = torch.cat(list(fake_screens), dim=1)
            
            #s1 = time.time()
            for t in count():
                # Select and perform an action
                action = bd_select_action(state, bd_stop_training)
                state_variables, _, done, _, _ = env.step(action.item())

                # Observe new state
                screens.append(get_screen())
                next_state = torch.cat(list(screens), dim=1) if not done else None

                # Reward modification for better stability
                x, x_dot, theta, theta_dot = state_variables
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                reward = r1 + r2
                reward = torch.tensor([reward], device=device)
                if t >= END_SCORE-1:
                    reward = reward + 20
                    done = 1
                else: 
                    if done:
                        reward = reward - 20 

                # Store the transition in memory
                bd_memory.push(state, action, next_state, reward)
                state_variables=torch.tensor(state_variables, device=device).view(-1,4)
                env_memory.push(state, action, next_state, reward, state_variables)
                

                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the target network)
                if done:
                    bd_episode_durations.append(t + 1)
                    bd_mean_last.append(t + 1)
                    mean = 0
                    #wandb.log({'Episode duration': t+1 , 'Episode number': i_episode})
                    for i in range(LAST_EPISODES_NUM):
                        mean = bd_mean_last[i] + mean
                    mean = mean/LAST_EPISODES_NUM
                    if mean < TRAINING_STOP and bd_stop_training == False:
                        #print(f"\nbd exploration: {time.time()-s1}")
                        #s1 = time.time()
                        torch.cuda.empty_cache()
                        fake_data = optimize_env_model()
                        #print(f"\nenv training: {time.time()-s1}")
                        #s1 = time.time()
                        if not (fake_data is None):
                            optimize_model_with_fake_data(fake_data[0], fake_data[1], 
                                                          fake_data[2], fake_data[3])
                        else:
                            optimize_model_with_fake_data()
                        #print(f"\nbd training: {time.time()-s1}")
                    else:
                        bd_stop_training = True
                        #print("bd boy did it")
                    break



        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            #target_net.load_state_dict(policy_net.state_dict())
            bd_target_net.load_state_dict(bd_policy_net.state_dict())
        #if stop_training == True:
        #    count_final += 1
            
        if bd_stop_training == True:
            bd_count_final += 1
        
        if bd_stop_training:
            if (bd_count_final >= 100):
                break

    print('Complete')
    #stop_training = False
    bd_stop_training = False
    #episodes_trajectories.append(episode_durations)
    bd_episodes_trajectories.append(bd_episode_durations)
    bd_episode_durations = np.array(bd_episode_durations)
    np.savetxt(f"project_episode_durations/bd{j}.csv", bd_episode_durations, delimiter=',')







# again for the both data model
best = []
for et in bd_episodes_trajectories:
    best.append(et)

maximum = 0
for i in range(len(best)):
    maximum = max(len(best[i]), maximum)
    
# Fill the episodes to make them the same length
for i in range(len(best)):
    length = len(best[i])
    for j in range(maximum - len(best[i])):
        best[i].append(best[i][j+length-100])
    best[i] = np.asarray(best[i])
    
best = np.asarray(best)


# To numpy
bd_score_mean = np.zeros(maximum)
bd_score_std = np.zeros(maximum)
bd_last100_mean = np.zeros(maximum)
print(best[:, max(0, -99):1].mean())
for i in range(maximum):
    bd_score_mean[i]  = best[:, i].mean()
    bd_score_std[i] = best[:, i].std()
    bd_last100_mean[i] = best[:, max(0, i-50):min(maximum, i+50)].mean()
print(len(bd_last100_mean))
t = np.arange(0, maximum, 1)



ax[1].fill_between(t, np.maximum(bd_score_mean - bd_score_std, 0),
                np.minimum(bd_score_mean + bd_score_std, END_SCORE), color='g', alpha=0.2, label='Real and Fake Variance')

ax[1].plot(t, bd_score_mean, color='g', label='Real and Fake')
ax[1].legend()




try:
    del(t, bd_last100_mean, best, bd_score_mean, bd_score_std, bd_stop_training, bd_episodes_trajectories, bd_episode_durations, bd_memory, 
        bd_target_net, bd_policy_net, bd_optim, bd_mean_last, env_memory)
except:
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~myyyyyyyyyyyyy wayyyyyyyyyyyyyyy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
g_episodes_trajectories = []
last_run = 0
for a in range(runs):
    if f"g{a}.csv" in filenames:
        last_run = a+1
        file_path = os.path.join(folder, f"g{a}.csv")
        g_episodes_trajectories.append(np.loadtxt(file_path, delimiter=',').tolist())

g_stop_training = False
for j in range(last_run,runs):
    print("g",j)
    env_net = D_AutoEncoder(screen_height, screen_width, latent_dims, n_actions, action_latents).to(device)
    g_mean_last = deque([0] * LAST_EPISODES_NUM, LAST_EPISODES_NUM)
    g_policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    g_target_net = DQN(screen_height, screen_width, n_actions).to(device)
    g_target_net.load_state_dict(g_policy_net.state_dict())
    g_target_net.eval()

    env_memory = EnvMemory(MEMORY_SIZE)
    env_optim = optim.Adam(env_net.parameters(), env_lr)
    
    g_optim = optim.RMSprop(g_policy_net.parameters())
    g_memory = ReplayMemory(MEMORY_SIZE)
    g_count_final = 0
    
    g_steps_done = 0
    g_episode_durations = []
    
    for i_episode in tqdm(range(N_EPISODES)):
        if not g_stop_training:
            env.reset()
            init_screen = get_screen()
            screens = deque([init_screen] * FRAMES, FRAMES)
            state = torch.cat(list(screens), dim=1)

            fake_screens = deque([init_screen] * FRAMES, FRAMES)
            fake_state = torch.cat(list(fake_screens), dim=1)
            
            #s1 = time.time()
            for t in count():
                # Select and perform an action
                action = g_select_action(state, g_stop_training)
                state_variables, _, done, _, _ = env.step(action.item())

                # Observe new state
                screens.append(get_screen())
                next_state = torch.cat(list(screens), dim=1) if not done else None

                # Reward modification for better stability
                x, x_dot, theta, theta_dot = state_variables
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                reward = r1 + r2
                reward = torch.tensor([reward], device=device)
                if t >= END_SCORE-1:
                    reward = reward + 20
                    done = 1
                else: 
                    if done:
                        reward = reward - 20 

                # Store the transition in memory
                g_memory.push(state, action, next_state, reward)
                state_variables=torch.tensor(state_variables, device=device).view(-1,4)
                env_memory.push(state, action, next_state, reward, state_variables)
                

                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the target network)
                if done:
                    g_episode_durations.append(t + 1)
                    g_mean_last.append(t + 1)
                    mean = 0
                    #wandb.log({'Episode duration': t+1 , 'Episode number': i_episode})
                    for i in range(LAST_EPISODES_NUM):
                        mean = g_mean_last[i] + mean
                    mean = mean/LAST_EPISODES_NUM
                    if mean < TRAINING_STOP and g_stop_training == False:
                        #print(f"\nbd exploration: {time.time()-s1}")
                        #s1 = time.time()
                        torch.cuda.empty_cache()
                        fake_data = optimize_env_model(guided=True)
                        #print(f"\nenv training: {time.time()-s1}")
                        #s1 = time.time()
                        if not (fake_data is None):
                            optimize_guided_model(fake_data[0], fake_data[1], 
                                                          fake_data[2], fake_data[3])
                        else:
                            optimize_guided_model()
                        #print(f"\nbd training: {time.time()-s1}")
                    else:
                        g_stop_training = True
                        #print("bd boy did it")
                    break



        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            #target_net.load_state_dict(policy_net.state_dict())
            g_target_net.load_state_dict(g_policy_net.state_dict())
        #if stop_training == True:
        #    count_final += 1
            
        if g_stop_training == True:
            g_count_final += 1
            if (g_count_final >= 100):
                break

    print('Complete')
    #stop_training = False
    g_stop_training = False
    #episodes_trajectories.append(episode_durations)
    g_episodes_trajectories.append(g_episode_durations)
    g_episode_durations = np.array(g_episode_durations)
    np.savetxt(f"project_episode_durations/g{j}.csv", g_episode_durations, delimiter=',')






# again for the both data model
best = []
for et in g_episodes_trajectories:
    best.append(et)
    

maximum = 0
for i in range(len(best)):
    maximum = max(len(best[i]), maximum)
    
# Fill the episodes to make them the same length
for i in range(len(best)):
    length = len(best[i])
    for j in range(maximum - len(best[i])):
        best[i].append(best[i][j+length-100])
    best[i] = np.asarray(best[i])
    
best = np.asarray(best)


# To numpy
g_score_mean = np.zeros(maximum)
g_score_std = np.zeros(maximum)
g_last100_mean = np.zeros(maximum)
print(best[:, max(0, -99):1].mean())
for i in range(maximum):
    g_score_mean[i]  = best[:, i].mean()
    g_score_std[i] = best[:, i].std()
    g_last100_mean[i] = best[:, max(0, i-50):min(maximum, i+50)].mean()
print(len(g_last100_mean))
t = np.arange(0, maximum, 1)


#fig, ax = plt.subplots(figsize=(16, 8))


ax[2].fill_between(t, np.maximum(g_score_mean - g_score_std, 0),
                np.minimum(g_score_mean + g_score_std, END_SCORE), color='r', alpha=0.2, label='Proposed Variance')

ax[2].plot(t, g_score_mean, color='r', label='Proposed')



ax[2].legend()
fig.text(0.5, 0.04, 'Number of Games', ha='center')
fig.text(0.04, 0.5, 'Number of Steps Survived', va='center', rotation='vertical')
fig.savefig('score_all.png')



