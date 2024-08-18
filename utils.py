import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
def pad_collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.
    Args:
        batch (list of np.array): List of numpy arrays of shape (T, H, W, 3).
    Returns:
        Tensor: Padded batch of shape (B, T_max, H, W, 3).
        Tensor: Original lengths of each sequence in the batch.
    """
    # Find the maximum sequence length in the batch

    T_max = max(sample['observation']['rgb_image'].shape[0] for sample in batch)
    H, W, C = batch[0]['observation']['rgb_image'].shape[1:]  # assuming all images have the same H, W, C
    action_dim = batch[0]['action'].shape[1]
    goal_dim = batch[0]['observation']['point_goal'].shape[1]


    padded_rgb_image = np.zeros((len(batch), T_max, H, W, C), dtype=np.float32)
    padded_action = np.zeros((len(batch), T_max, action_dim), dtype=np.float32)
    padded_goal = np.zeros((len(batch), T_max, goal_dim), dtype=np.float32)
    padded_is_terminal = np.array([[True]*T_max]*len(batch))

    lengths = []

    for i, sample in enumerate(batch):
        T = sample['observation']['rgb_image'].shape[0]
        padded_rgb_image[i, :T, :, :, :] = sample['observation']['rgb_image']
        padded_action[i, :T, :] = sample['action']
        padded_goal[i, :T, :] = sample['observation']['point_goal']
        padded_is_terminal[i, :T] = sample['is_terminal']
        lengths.append(T)

    return {'observation': {'rgb_image':torch.tensor(padded_rgb_image), 'point_goal':torch.tensor(padded_goal)},
            'action' : torch.tensor(padded_action),
            'is_terminal' : torch.tensor(padded_is_terminal),
            'lengths': torch.tensor(lengths),
            'max_length':T_max
        }
class TFDSWrapper(Dataset):
    def __init__(self, tfds_dataset, q1=None, q3=None):
        self.q1 = q1
        self.q3 = q3
        
        self.tfds_dataset = tfds_dataset
        self.flattened_data = self.flatten_rlds(tfds_dataset)


    def __len__(self):
        return len(self.tfds_dataset)

    def __getitem__(self, idx):
        sample = self.flattened_data[idx]
        observations  = torch.tensor(sample['images'], dtype=torch.float32)
        actions = sample['actions']
        goals = sample['goals']
        is_terminals = sample['is_terminals']
        actions = (actions - self.q1) / (self.q1 - self.q3)
        return  {'observation' : {'rgb_image':observations, 'point_goal':goals},
                'action' : actions,
                'is_terminal' : is_terminals,
                }

    def flatten_rlds(self, rlds_dataset):
        episodic_data = []
        count = 0
        all_actions = [] # For calculating norm
        for episode in iter(rlds_dataset):

            steps = list(episode['steps'])
            images = np.array([cv2.resize(np.array(step['observation']['image']), (256, 256)) for step in steps])
            actions = np.array([step['action'].numpy() for step in steps])
            goals = np.array([step['goal'].numpy() for step in steps])
            is_terminals = np.array([step['is_terminal'].numpy() for step in steps])
            states = np.array([step['observation']['state'].numpy() for step in steps])
            episodic_data.append({'images':images, 'actions':actions, 'goals':goals, 'is_terminals':is_terminals, 'states':states})
            count+=1
            for step in steps:
                all_actions.append(list(step['action'].numpy()))
        all_actions = np.array(all_actions)
        if self.q1 is None or self.q3 is None:
            print("q1 and/or q3 are not given. Calculating q1 and q3")
            self.q1 = np.percentile(all_actions, 25, axis=0)
            self.q3 = np.percentile(all_actions, 75, axis=0)
        return episodic_data

def unnormalize_action(norm_actions, q1, q3, device):
    #TODO if isinstance(q1, np.array):
    q1 = torch.tensor(q1).to(device)
    q3 = torch.tensor(q3).to(device)
    unnorm_actions = norm_actions*(q1 - q3) + q1
    return unnorm_actions


def resize_and_crop(frame, min_size=224, mode="middle"):
    '''
    mode can be ["middle", "crop_left", "crop_right"]
    frame has the size of (H, W, 3)
    '''
    # Calculate the new dimensions maintaining the aspect ratio
    if frame.shape[2] == 3:
        height, width, channel = frame.shape
    else:
        channel, height, width = frame.shape

    if width < height:
        new_width = min_size
        new_height = int(height * (min_size / width))
    else:
        new_height = min_size
        new_width = int(width * (min_size / height))
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    crop_size = min_size
    if mode == "middle": # Crop from the centre of the
        center_x, center_y = new_width // 2, new_height // 2
        if frame.shape[2] == 3:
            cropped_frame = resized_frame[
                center_y - crop_size // 2 : center_y + crop_size // 2,
                center_x - crop_size // 2 : center_x + crop_size // 2
            ]
        else:
            cropped_frame = resized_frame[
                :,
                center_y - crop_size // 2 : center_y + crop_size // 2,
                center_x - crop_size // 2 : center_x + crop_size // 2
            ]
    elif mode == "crop_left": # Crop from the most left
        if frame.shape[2] == 3:
            cropped_frame = resized_frame[
                (new_height - crop_size) // 2 : (new_height + crop_size) // 2,
                0 : crop_size
            ]
        else:
            cropped_frame = resized_frame[
                :,
                (new_height - crop_size) // 2 : (new_height + crop_size) // 2,
                0 : crop_size
            ]
    elif mode == "crop_right": # Crop to the most right
        if frame.shape[2] == 3:
            cropped_frame = resized_frame[
                (new_height - crop_size) // 2 : (new_height + crop_size) // 2,
                new_width - crop_size : new_width
            ]
        else:
            cropped_frame = resized_frame[
                :,
                (new_height - crop_size) // 2 : (new_height + crop_size) // 2,
                new_width - crop_size : new_width
            ]
    else:
        raise ValueError("Invalid mode. Supported modes are: 'middle', 'crop_left', 'crop_right'")
    
    return cropped_frame