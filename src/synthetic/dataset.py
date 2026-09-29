import numpy as np
from sklearn.utils import check_random_state
from scipy.special import comb
import itertools
from itertools import combinations

from utils import sample_action_fast, softmax, generate_tensor
import conf

# Create nCk combinations only used in one of the reward functions
def generate_combination_matrix(n, k):
    # Generate all nCk combinations
    combinations = list(itertools.combinations(range(n), k))
    
    # Initialize the matrix
    matrix = np.zeros((len(combinations), n), dtype=int)
    
    # transform the combinations into the matrix
    for i, comb in enumerate(combinations):
        matrix[i, comb] = 1
    
    return matrix


# Create the simple latent reward for each action feature: q_l(x_i, f_l) \forall i \in [n], l \in [d]
# Only the user-item interaction effect
def create_q_l_x_f_l_simple(x, # sampled context: x_i i \in [n]
                    num_actions_all, # number of all actions: |A_all|
                    dim_feature, # dimension of the action feature: d
                    one_hot_mat_f, # the matrix of the one hot array of each action feature: shape of one_hot_mat_f: (|A_all|, d, |F_l|)
                    coef_dataset, # object which saves the coefficient for generating the logged data such as alpha, M, etc
                    ):
    dim_context = x.shape[1] # dimension of the context

    # Create the coefficient for the user-interaction effect by already sampled hyperparameters
    tensor_d_x_num_a_dim_f = np.zeros((dim_context, num_actions_all, dim_feature))
    for i in range(dim_context):
        tensor_d_x_num_a_dim_f[i] = (one_hot_mat_f * coef_dataset.T_0_x_f[i]).sum(axis = 2)
    
    # user-item interaction effect
    q_l_x_f_l = np.einsum('nm,mlp->nlp', x, tensor_d_x_num_a_dim_f)
    
    return q_l_x_f_l





def create_expected_reward_action(x, # sampled context: x_i i \in [n]
                           num_actions_all, # number of all actions: |A_all|
                           dim_feature, # dimension of the action feature: d
                           one_hot_mat_f, # the matrix of the one hot array of each action feature: shape of one_hot_mat_f: (|A_all|, d, |F_l|)
                           coef_dataset, # object which saves the coefficient for generating the logged data such as alpha, M, etc
                           gamma, # parameter which can adjust how much the reward condition is satisfied (if gamma = 1, 100% satisfied)
                           eta,
                           s, # the dimension up to which the PCS is satisfied
                           ):

    # Create latent reward for each dimension of the action feature: q_l(x_i, f_l)
    if conf.flag_q_l_x_f_l_simple == True:
        q_l_x_f_l = create_q_l_x_f_l_simple(x, num_actions_all, dim_feature, one_hot_mat_f, coef_dataset)


    # Linear Effect from 1~d dimensions
    # \sum_{l=s+1}^d q_l(x, f_l) / (d - s)
    if dim_feature == s:
        q_linear_sp1_d = np.zeros((q_l_x_f_l.shape[0], q_l_x_f_l.shape[1]))
    else:
        q_linear_sp1_d = q_l_x_f_l[:, :, s:].sum(axis = 2) / (dim_feature - s) 
    q_linear_sp1_s = q_l_x_f_l[:, :, :s].sum(axis = 2) / (s) 
    q_linear_sp1_full = q_l_x_f_l.sum(axis = 2) / (dim_feature) 
    q_x_s = x @ coef_dataset.interaction_coef_s 

    q_1_s =  q_linear_sp1_s

    one_hot_mat_fs = generate_tensor(dim_feature = s, num_id_per_each_feature = conf.num_id_per_each_feature)

    indices_list = []

    for i in range(num_actions_all):
        match = np.all(one_hot_mat_fs == one_hot_mat_f[i, :s, :], axis=(1, 2))
        indices = np.where(match)[0]
        indices_list.append(indices)

    indices_array = np.array(indices_list)  

    for j in range(x.shape[0]):
        q_1_s[j, :] += eta * q_x_s[j, indices_array].squeeze()

    q_1_d =  q_linear_sp1_full + eta * x @ coef_dataset.interaction_coef_d 
    
    q_x_a = (q_linear_sp1_d + q_1_s) + gamma * q_1_d

    return q_x_a 




def generate_synthetic_data(
    random_state: int, 
    num_data: int, 
    dim_context: int, 
    num_actions_all: int, 
    num_id_per_each_feature: int, 
    dim_feature: int, 
    coef_dataset: callable, 
    set_of_actions: np.ndarray, 
    beta: float, 
    gamma: float, 
    eta: float, 
    s: int, 
    lambda_: float, 
) -> dict:

    random_ = check_random_state(random_state)

    ############# CONTEXT X #############
    # Sample context from a uniform distribution
    # x \sim Unif ([0, 1])
    if conf.flag_uniform_context == True:
        x = random_.uniform(low = 0, high = 1, size=(num_data, dim_context))
    else:
        x = random_.normal(loc = 1, scale = 1, size=(num_data, dim_context))

    ############# EXPECTED REWARD q(x, a) #############
    ### Define q(x, a)
    num_actions = len(set_of_actions) # number of action considered in a logging policy: |A|
    one_hot_a_all = np.eye(num_actions_all) # shape of one_hot_a_all: (|A_all|, |A_all|)
    one_hot_a = np.eye(num_actions) # shape of one_hot_a: (|A|, |A|)
    one_hot_mat_f = generate_tensor(dim_feature = dim_feature, num_id_per_each_feature = num_id_per_each_feature) # shape of one_hot_mat_f: (|A_all|, d, |F_l|)


    if conf.reward_type == "action":
        q_x_a = create_expected_reward_action(x, num_actions_all, dim_feature, one_hot_mat_f, coef_dataset, gamma, eta, s)


    ############# LOGGING POLICY \pi_0(a|x) #############
    # Extract the q(x, a) for actions included in the set of actions A
    q_x_a_logging = q_x_a[:, set_of_actions]

    # Add noise to q(x, a) by \eta_f to create noised expected reward
    noised_q_x_a_logging = q_x_a_logging + coef_dataset.eta_f[set_of_actions]

    # Create the logging policy in A by apply a softmax function to the noise q(x, a)
    pi_0 = softmax(beta * noised_q_x_a_logging)

    # Create the logging policy in A_all
    pi_0_A_all = np.zeros_like(q_x_a)
    pi_0_A_all[:, set_of_actions] = pi_0
    pi_0_f = np.einsum('nm,mlp->nlp', pi_0_A_all, one_hot_mat_f)

    ############# ACTION A #############
    # sample actions for each data based on the logging policy
    # each action is from A considered in the logging policy 
    a = sample_action_fast(pi_0, random_state=random_state)

    # a_in_A_all is the index of the action in the whole action set A_all
    a_in_A_all = set_of_actions[a]

    ############# PROPENSITY SCORE \pi_0(f_{i, l|x_i) #############
    # propensity scores in terms of each dimension of the action feature
    pscore_f = pi_0_f * one_hot_mat_f[a_in_A_all]
    pscore_f = pscore_f.sum(axis = 2)

    ############# PROPENSITY SCORE 1:S \pi_0(f_{i, 1:s}|x_i) #############
    # propensity scores in terms of 1:s dimensions of the action feature
    taken_feature_tensor = one_hot_mat_f[a_in_A_all]
    taken_feature_up_to_s = taken_feature_tensor[:, :s, :].reshape(num_data, s * num_id_per_each_feature)
    one_hot_mat_f_up_to_s = one_hot_mat_f[:, :s, :].reshape(num_actions_all, s * num_id_per_each_feature)
    
    indicator_mat_up_to_s = np.all(taken_feature_up_to_s[:, None, :] == one_hot_mat_f_up_to_s[None, :, :], axis=2)

    pscore_f_up_to_s = (pi_0_A_all * indicator_mat_up_to_s).sum(axis = 1)

    ############# REWARD R #############
    # Sample reward from a normal distribution
    # r \sim N(q(x, a), sigma^2=1)
    idx = np.arange(num_data)
    q_x_a_factual = q_x_a_logging[idx, a] 
    r = random_.normal(q_x_a_factual, 0.5) 


    # create Gamma matrix for LCPI-PG
    action_indicator_all = get_action_indicator_all(num_id_per_each_feature, s, dim_feature, one_hot_mat_f)
    action_indicator_all = np.expand_dims(action_indicator_all, 0)
    action_indicator_all_T = action_indicator_all.transpose(0, 1, 3, 2)

    action_indicator_cov_mat = action_indicator_all @ action_indicator_all_T

    cov_mat = action_indicator_cov_mat


    action_indicator_all = action_indicator_all.astype(np.float16)
    cov_mat = cov_mat.astype(np.float16)
    pi_0_A_all = pi_0_A_all.astype(np.float16)

    # Set batch size
    batch_size = 64  # Adjust as needed

    # Calculate the number of batches for a_in_A_all
    num_batches = (a_in_A_all.shape[0] + batch_size - 1) // batch_size

    # Compute Gamma_matrix for each batch
    Gamma_matrices = []

    for batch_idx in range(num_batches):
        # Define the range of the current batch
        start = batch_idx * batch_size
        end = min(start + batch_size, a_in_A_all.shape[0])

        # Extract data for the current batch
        a_in_A_batch = a_in_A_all[start:end]
        pi_0_A_batch = pi_0_A_all[start:end]

        # Expand dimensions as needed
        pi_0_A_all_exp = np.expand_dims(pi_0_A_batch, axis=(2, 3))
        a_in_A_all_expanded = np.expand_dims(a_in_A_batch, axis=(1, 2, 3))

        # Retrieve indices from action_indicator_all
        action_selected = np.take_along_axis(action_indicator_all, a_in_A_all_expanded, axis=1).squeeze(1)

        # Calculate Gamma for the current batch
        Gamma_batch = np.sum(pi_0_A_all_exp * cov_mat, axis=1)
        Gamma_batch = Gamma_batch.astype(np.float32)

        # Compute the pseudoinverse matrix
        Gamma_inv_batch = np.linalg.pinv(Gamma_batch)
        Gamma_matrix_batch = Gamma_inv_batch @ action_selected
        Gamma_matrices.append(Gamma_matrix_batch)

    # Concatenate all batches
    Gamma_matrix = np.concatenate(Gamma_matrices, axis=0)




    # Create dataset for the OPL methods that can consider action feature
    dataset_all = dict(
        num_data = num_data,
        num_actions_all = num_actions_all,
        num_id_per_each_feature = num_id_per_each_feature, 
        dim_feature = dim_feature, 
        s = s, 
        x = x,
        a = a_in_A_all, 
        r = r,
        pi_0 = pi_0_A_all, 
        pi_0_f = pi_0_f, 
        pscore = pi_0[idx, a],
        pscore_f = pscore_f, 
        pscore_f_up_to_s = pscore_f_up_to_s, 
        q_x_a = q_x_a,
        one_hot_a = one_hot_a, 
        one_hot_a_all = one_hot_a_all, 
        one_hot_mat_f = one_hot_mat_f, 
        Gamma_matrix = Gamma_matrix,
        action_indicator_all_T = action_indicator_all_T
    )

    # Create dataset for the OPL methods that only consider the exsting actions
    dataset_logging = dict(
        num_data = num_data,
        num_action = set_of_actions.shape[0],
        num_id_per_each_feature = num_id_per_each_feature, 
        dim_feature = dim_feature, 
        s = s, 
        x = x,
        a = a, 
        r = r,
        pi_0 = pi_0, 
        pi_0_f = pi_0_f, 
        pscore = pi_0[idx, a],
        pscore_f = pscore_f, 
        pscore_f_up_to_s = pscore_f_up_to_s, 
        q_x_a = q_x_a_logging,
        one_hot_a = one_hot_a, 
        one_hot_a_all = one_hot_a_all, 
        one_hot_mat_f = one_hot_mat_f, 
    )
    return dataset_all, dataset_logging


def get_action_indicator_all(num_id_per_each_feature, s, dim_feature, one_hot_mat_f):


    one_hot_combination = np.eye(num_id_per_each_feature ** s)

    one_hot_combination_ = np.repeat(one_hot_combination, num_id_per_each_feature ** (dim_feature - s), axis=0)
    
    action_indicator_all = np.concatenate(
        [one_hot_mat_f.reshape(num_id_per_each_feature ** dim_feature, -1), one_hot_combination_],
        axis=1
    )

    return np.expand_dims(action_indicator_all, axis=2)
