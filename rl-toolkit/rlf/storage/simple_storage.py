import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from rlf.storage import BaseStorage


class SimpleStorage(BaseStorage):
    def __init__(self, num_steps, num_processes, obs_space, action_space):
        super().__init__()
        """
        A simplified storage class that only stores observations, actions, and next observations.
        """
        self.num_steps = num_steps
        self.num_processes = num_processes

        # Initialize storage for observations, actions, and next observations
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_space)
        self.actions = torch.zeros(num_steps, num_processes, action_space)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.next_obs = torch.zeros(num_steps, num_processes, *obs_space)

        self.top = 0
        self.step = 0

    def insert(self, obs, action, next_obs):
        """
        Insert data into the storage.
        """
        self.obs[self.top].copy_(obs)
        self.actions[self.top].copy_(action)
        self.next_obs[self.top].copy_(next_obs)
        self.top = (self.top + 1) % self.num_steps
        if self.step < self.num_steps:
            self.step += 1

    def get_generator(self, num_mini_batch=None, mini_batch_size=None):
        """
        Generate mini-batches of data for training.
        """
        num_steps, num_processes = self.step, self.num_processes
        batch_size = num_steps * num_processes

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "The total batch size ({}) must be greater than or equal to the number of mini-batches ({})."
                .format(batch_size, num_mini_batch)
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            next_obs_batch = self.next_obs.view(-1, *self.next_obs.size()[2:])[indices]

            yield {
                "state": obs_batch,
                "action": actions_batch,
                "next_state": next_obs_batch,
            }

    def get_obs(self, step):
        """
        Get the observation at a specific step.
        """
        return self.obs[step]

    def get_action(self, step):
        """
        Get the action at a specific step.
        """
        return self.actions[step]

    def get_next_obs(self, step):
        """
        Get the next observation at a specific step.
        """
        return self.next_obs[step]

    def num_steps_can_sample(self):
        num_steps, num_processes = self.step, self.num_processes
        return num_steps * num_processes

    def to(self, device):
        """
        Move all stored tensors to the specified device.
        """
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.next_obs = self.next_obs.to(device)

    def reset(self):
        """
        Reset the storage for a new rollout.
        """
        self.top = 0
        self.step = 0
        self.obs.zero_()
        self.actions.zero_()
        self.next_obs.zero_()


if __name__ == '__main__':
    import time
    # Example usage
    obs_space = tuple([20])  # Example observation space
    action_space = 4  # Example action space

    storage = SimpleStorage(num_steps=128, num_processes=32, obs_space=obs_space, action_space=action_space)
    obs = torch.randn(3, 20)
    action = torch.randn(3,4)
    print(obs)
    print(action)
    for o,a in zip(obs,action):
        storage.insert(o, a, o)
        print(storage.num_steps_can_sample())

