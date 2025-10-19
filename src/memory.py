import numpy as np


class Memory():
    def __init__(self, capacity, image_shape):
        '''Initialises a memory to store observations and their associated rewards.'''

        # Initialize memory capacity
        self.capacity = capacity
        self.image_shape = image_shape
        # Initialize the memory for images aka observations in the frequency domain
        self.image = np.zeros((capacity, *image_shape), dtype=np.float32)
        # Initialize the memory for rewards associated with each observation
        self.reward = np.zeros(capacity, dtype=np.float32)

    def getReward(self):
        return self.reward

    def getRewardAt(self, pointer):
        return self.reward[pointer]

    def getAverageReward(self):
        return sum(self.reward) / len(self.reward) if self.reward else 0

    def getImageAt(self, pointer):
        return self.image[pointer]

    def getAllImages(self):
        return self.image

    def getCapacity(self):
        return len(self.image)

    def update(self, image, reward):
        '''Updates the memory at the current write pointer with the new image and reward.'''
        self.image = image
        self.reward = reward

    def updateAt(self, pointer, image, reward):
        '''Updates the memory at the specified pointer with the new image and reward.'''
        self.image[pointer] = image
        self.reward[pointer] = reward