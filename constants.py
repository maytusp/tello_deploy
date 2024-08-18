import gym
from gym import spaces
import numpy as np

observation_space = spaces.Dict({
    'rgb_image': spaces.Box(
        low=0,
        high=255,
        shape=(256, 256, 3),
        dtype=np.uint8
    ),
    'point_goal': spaces.Box(
        low=0,
        high=255,
        shape=(2,),
        dtype=np.uint8
    ),
})
action_space = spaces.Box(
    low=np.array([-1.0, -1.0, -1.0, -np.pi]),  # Minimum values for x, y, z, and angle
    high=np.array([1.0, 1.0, 1.0, np.pi]),     # Maximum values for x, y, z, and angle
    dtype=np.float32
)

if __name__ == "__main__":
    print(action_space.shape[0])