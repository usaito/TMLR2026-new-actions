from dataclasses import dataclass
from scipy.special import comb

import random
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import torch
import conf


class CoeffLoggedData():
    def __init__(self, dim_context, 
                 dim_feature, 
                 num_id_per_each_feature, 
                 s, 
                 random_state):
        random_ = check_random_state(random_state)

        self.dim_context = dim_context # d_x
        self.dim_feature = dim_feature # d_f
        self.num_id_per_each_feature = num_id_per_each_feature # |F_j|
        self.num_actions_all = self.num_id_per_each_feature ** self.dim_feature # |A| := |F_j| ** d_f

        self.T_0_x_f = random_.uniform(low = 0, high = 1, size = (self.dim_context, self.dim_feature, self.num_id_per_each_feature))
    
    
        # Define the coefficients for the noise expected reward
        self.eta_f = random_.normal(loc = 0, scale = conf.eta_f_var, size = (self.num_actions_all))

        self.interaction_coef_s= random_.uniform(low = -1, high = 1 , size =(self.dim_context, self.num_id_per_each_feature**s))
        self.interaction_coef_d= random_.uniform(low = -1, high = 1, size =(self.dim_context, self.num_actions_all))


# function to create the matrix of the one hot array of the action feature
# typical shape: (|A_all|, d, |F_l|)
def generate_tensor(dim_feature, num_id_per_each_feature):
    # Generate all possible combinations of length N using elements from [0, M-1]
    combinations = np.indices([num_id_per_each_feature] * dim_feature).reshape(dim_feature, -1).T
    
    # Initialize the tensor
    tensor = np.zeros((num_id_per_each_feature ** dim_feature, dim_feature, num_id_per_each_feature), dtype=int)
    
    # Fill the tensor with one-hot vectors
    for idx, combo in enumerate(combinations):
        for j, val in enumerate(combo):
            tensor[idx, j, val] = 1
    
    return tensor


def sample_set_of_actions(dim_feature, num_id_per_each_feature, num_actions_all, num_actions, random_state, s):
    random_ = check_random_state(random_state)

    # Create A_base_is: Sample actions to maintain the independent support 
    count = 0
    for i in range(dim_feature):
        count += num_id_per_each_feature ** i
    A_base_is = np.arange(num_id_per_each_feature) * count

    # Create A_base_pcs: Sample actions to maintain the Partial Combination Support
    B_1_s = num_id_per_each_feature ** (dim_feature - s) * np.arange(0, num_id_per_each_feature ** s)
    B_1_s_minus = np.arange(0, num_id_per_each_feature) * num_id_per_each_feature ** (dim_feature - s) * (num_id_per_each_feature ** s - 1) / (num_id_per_each_feature - 1)
    mask = np.isin(B_1_s, B_1_s_minus, invert=True)
    B_1_s = B_1_s[mask]
    A_base_pcs = B_1_s

    # Take the union of the A_base_is and A_base_pcs
    A_base = np.concatenate((A_base_is, A_base_pcs))
    A_base.sort()

    # # Sample the other actions to construct the set of actions A
    A_random_candidate = np.arange(num_actions_all)
    mask = np.isin(A_random_candidate, A_base, invert = True)
    A_random_candidate = A_random_candidate[mask]

    if num_actions > len(A_base):
        A_random = random_.choice(A_random_candidate, num_actions - len(A_base), replace=False)
        set_of_actions = np.concatenate((A_base, A_random))
    elif num_actions == len(A_base):
        set_of_actions = A_base
    else:
        raise ValueError(f"num_actions = {num_actions}. len(A_base_is) = num_id_per_each_feature = {len(A_base_is)}. len(A_base_pcs) = {len(A_base_pcs)}. But, num_actions > len(A_base_is) + len(A_base_pcs) should hold")
    set_of_actions.sort()
    return set_of_actions
    

def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions

def calc_policy_value(pi: np.ndarray, q_x_a: np.ndarray, random_state: int = 12345) -> np.ndarray:
    policy_value = (pi * q_x_a).sum(axis=1).mean()

    return policy_value



def softmax(x: np.ndarray) -> np.ndarray:
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator



def gen_eps_greedy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    eps: float = 0.0,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    base_pol = np.zeros_like(expected_reward)
    if is_optimal:
        a = np.argmax(expected_reward, axis=1)
    else:
        a = np.argmin(expected_reward, axis=1)
    base_pol[
        np.arange(expected_reward.shape[0]),
        a,
    ] = 1
    pol = (1.0 - eps) * base_pol
    pol += eps / expected_reward.shape[1]

    return pol[:, :]


def fix_seed(seed=12345):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)

@dataclass
class RegBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.context.shape[0] == self.action.shape[0] == self.reward.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class GradientBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]
    



@dataclass
class LCPIDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    pscore_f: np.ndarray
    pscore_f_up_to_s: np.ndarray
    q_hat: np.ndarray
    pi_0: np.ndarray
    pi_0_f: np.ndarray
    Gamma_matrix: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.pscore_f.shape[0]
            == self.pscore_f_up_to_s.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
            == self.pi_0_f.shape[0]
            == self.Gamma_matrix.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.pscore_f[index],
            self.pscore_f_up_to_s[index],
            self.q_hat[index],
            self.pi_0[index],
            self.pi_0_f[index],
            self.Gamma_matrix[index],
        )

    def __len__(self):
        return self.context.shape[0]