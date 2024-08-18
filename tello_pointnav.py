from resnet_policy import *
from ae.apply_recon import *
from utils import *
from create_agent import NetPolicy

from djitellopy import Tello
import time
import pygame
import numpy as np
import os

import torch
import cv2

import wandb
import matplotlib.pyplot as plt
torch.manual_seed(42)
np.random.seed(42)

# Variables
ckpt_path =  "/home/maytusp/Projects/drone/pointnav/checkpoints/ckpt_20000.pth"
q1=np.array([-0.00125363, 0.00186184, -0.00012733, -0.00461001]) # For unnorm action (dataset specific)
q3=np.array([0.0022604, 0.01293997, 0.00013227, 0.0045612 ])

x_target, y_target = 0,3 # World frame
ep_name = f"dua/disco_light/disco_light_ep40_x{x_target}y{y_target}"
use_dua = True
use_aae = False

# AAE Variables
aae_backbone  = "seresnext50"
adapt_encoder = True
aae_state_dict_path = "checkpoints/aae_ckpt_deploy/ae_cnn_bn_drone_small_gibson_large.pt"

if use_dua and use_aae:
    raise TypeError("Shouldn't use both DUA and TTA-Nav")

# Initialise Agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = NetPolicy(use_dua=use_dua,device=device)
agent.load_state_dict(torch.load(ckpt_path))
agent.eval()
        
# adaptive autoencoder (CLIP-ResNet50 as Encoder)
if use_aae:
    
    aae = apply_ae(device=device,
                backbone=aae_backbone,
                adapt_encoder=adapt_encoder,
                state_dict_path=aae_state_dict_path)
    print("AAE LOADED")

# Initialise Tello
tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()
print(f"battery = {tello.get_battery()}")
tello.takeoff()
time.sleep(3)
tello.go_xyz_speed(0, 0, -30, 30)

# Set constants
freq = 15
total_time = 20 # in seconds
x_global, y_global = 0,0 # World frame of current pos (init as origin)
pred_actions = []
goals = []
obs_array = []



saved_dir = f"logs/{ep_name}"
os.makedirs(saved_dir, exist_ok=True)
video_path = os.path.join(saved_dir, "observation.mp4")
result = cv2.VideoWriter(video_path,  
                         cv2.VideoWriter_fourcc(*'MP4V'), 
                         5, (256,256))

for i in range(20): # wait until camera is initialised
    frame_read = tello.get_frame_read()
    frame = frame_read.frame # (W,H,3)
    time.sleep(1/freq)
try:
    for step in range(total_time*freq):
        curr_yaw = -tello.get_yaw() # the Tello PY defines x as forward and y as left, but yaw is defined as clockwise = +
        curr_yaw = curr_yaw * (np.pi / 180)
        # get observations: image and goal
        # get image
        frame_read = tello.get_frame_read()
        frame = frame_read.frame # (W,H,3)
        frame = resize_and_crop(frame, min_size=256, mode="middle") # (W,H,3) to (256,256,3)
        
        if use_aae:
            frame = aae.recon(frame)

        
        result.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        obs_array.append(frame)
        frame = np.expand_dims(frame, axis=0) # From (256,256,3) to (1,256,256,3)

        # get goal
        # Transform goal from global to drone coordinate
        dx_goal_global = x_target - x_global
        dy_goal_global = y_target - y_global
        dx_goal_drone = dx_goal_global*np.cos(curr_yaw) +  dy_goal_global*np.sin(curr_yaw)
        dy_goal_drone = -dx_goal_global*np.sin(curr_yaw) + dy_goal_global*np.cos(curr_yaw)
        point_goal = [dx_goal_drone, dy_goal_drone]
        
        # print(f"t={step}: curr_yaw = {curr_yaw*(180/np.pi)}, goal: {point_goal}, battery = {tello.get_battery()}")

        # process obs
        observations = dict()
        observations["rgb_image"] = torch.tensor(frame).to(device)
        observations["point_goal"] = torch.tensor(np.expand_dims(np.array(point_goal), axis=0
        ), dtype=torch.float32).to(device)

        norm_action = agent.get_action(observations)
        
        # get and process action
        action = unnormalize_action(norm_action,
                                    q1,
                                    q3,
                                    device)
        print("unnorm action: forward, turn right left", action[1], action[3])
        print("yaw", curr_yaw)
        action = list(action.detach().cpu().numpy())
        pred_actions.append(action)
        goals.append(point_goal)
        action[0] = 0
        dx, dy, dz, dyaw = action[0], action[1]*3, action[2], action[3]*3.2

        dx_global = dx*np.cos(curr_yaw) - dy*np.sin(curr_yaw)
        dy_global = dx*np.sin(curr_yaw) + dy*np.cos(curr_yaw)

        vx, vy, vz, vyaw = dx * freq, dy * freq, dz * freq, dyaw * freq
        
        # vx, vy, vz, vyaw = dx_global * freq, dy_global * freq, dz * freq, 0 * freq # freq
        vx_cm, vy_cm, vz_cm, vyaw = int(vx*0), int(vy*100), int(vz*0), int(vyaw*100) # m to cm and calibrate yaw from radian to Tello control command

        # control define vx as left right
        # yaw from the model and yaw in tello is differently defined
        tello.send_rc_control(vx_cm,vy_cm,0, -vyaw)
        
        # Odometry (imperfect) TODO Check this again
        x_global += dx_global
        y_global += dy_global

        # set control frequency
        time.sleep(1/freq)
finally:
    tello.land()
    result.release()
    print("VIDEO SAVED")

    # Wandb Log
    img_strip = np.concatenate(np.array(obs_array[::3]), axis=1)
    ACTION_DIM_LABELS = ['forward backward', 'turn right left']
    GOALS = ['distance head', 'distance side']
    figure_layout = [
        ['image'] * len(ACTION_DIM_LABELS),
        ACTION_DIM_LABELS,
        GOALS
    ]
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    # plot actions
    pred_actions = np.array(pred_actions).squeeze()
    pred_actions = pred_actions[:, [1,3]]
    goals = np.array(goals).squeeze()
    for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        try:
            axs[action_label].plot(pred_actions[:, action_dim])
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel('Time in one episode')
        except:
            pass
    for dim, label in enumerate(GOALS):
        try:
            axs[label].plot(goals[:, dim])
            axs[label].set_title(label)
            axs[label].set_xlabel('Time in one episode')
        except:
            pass

    axs['image'].imshow(img_strip)
    axs['image'].set_xlabel('Time in one episode (subsampled)')
    plt.legend()

    plt.savefig(os.path.join(saved_dir, "plot.png"))