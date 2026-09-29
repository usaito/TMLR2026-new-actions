import numpy as np
from sklearn.utils import check_random_state

from utils import sample_action_fast, softmax
import conf
import pandas as pd


def generate_real_data(
    all_data: pd.DataFrame,
    user_data: pd.DataFrame,
    action_data: pd.DataFrame,
    random_state: int, 
    num_data: int, 
    num_id_per_each_feature: int, 
    dim_feature: int, 
    set_of_actions: np.ndarray, 
    beta: float, 
    s: int, 
) -> dict:

    random_ = check_random_state(random_state)
    idx = random_.choice(all_data["user_id"].unique(), size=num_data, replace=True)

    x = user_data.set_index("user_id").loc[idx].to_numpy()

    num_actions_all = action_data["video_id"].nunique()
    num_actions = len(set_of_actions)

    one_hot_a_all = np.eye(num_actions_all) # shape of one_hot_a_all: (|A_all|, |A_all|)
    one_hot_a = np.eye(num_actions) # shape of one_hot_a: (|A|, |A|)

    q_x_a = all_data.set_index("user_id").loc[idx]["watch_ratio"].to_numpy().reshape(num_data, -1)
    q_x_a_logging = q_x_a[:, set_of_actions]

    pi_0 = softmax(beta * q_x_a_logging)

    # Create the logging policy in A_all
    pi_0_A_all = np.zeros_like(q_x_a)
    pi_0_A_all[:, set_of_actions] = pi_0  

    a = sample_action_fast(pi_0, random_state=random_state)

    # a_in_A_all is the index of the action in the whole action set A_all
    a_in_A_all = set_of_actions[a]

    idx = np.arange(num_data)
    q_x_a_factual = q_x_a_logging[idx, a] 
    r = random_.normal(q_x_a_factual, 0.5) 
   
    if conf.action_feature == "video_duration":
        action_feature_list = action_data[["feat", "first_level_category_id", "second_level_category_id", "video_duration"]].to_numpy()
        num_id_per_each_feature_list = action_data.nunique()[["feat", "first_level_category_id", "second_level_category_id", "video_duration"]].to_numpy()
    elif conf.action_feature == "third_level_category_id":
        action_feature_list = action_data[["feat", "first_level_category_id", "second_level_category_id", "third_level_category_id"]].to_numpy()
        num_id_per_each_feature_list = action_data.nunique()[["feat", "first_level_category_id", "second_level_category_id", "third_level_category_id"]].to_numpy()
    elif conf.action_feature =="second_level_category_id":
        action_feature_list = action_data[["feat", "first_level_category_id", "third_level_category_id","second_level_category_id"]].to_numpy()
        num_id_per_each_feature_list = action_data.nunique()[["feat", "first_level_category_id", "third_level_category_id", "second_level_category_id"]].to_numpy()
    elif conf.action_feature == "feat":
        action_feature_list = action_data[["first_level_category_id", "second_level_category_id", "third_level_category_id", "feat"]].to_numpy()
        num_id_per_each_feature_list = action_data.nunique()[["first_level_category_id", "second_level_category_id", "third_level_category_id","feat"]].to_numpy()
    elif conf.action_feature == "first_level_category_id":
        action_feature_list = action_data[["feat", "third_level_category_id", "second_level_category_id", "first_level_category_id"]].to_numpy()
        num_id_per_each_feature_list = action_data.nunique()[["feat", "third_level_category_id", "second_level_category_id", "first_level_category_id"]].to_numpy()
    elif conf.action_feature == "author_id":
        action_feature_list = action_data[["feat", "first_level_category_id", "second_level_category_id", "third_level_category_id", "author_id"]].to_numpy()
        num_id_per_each_feature_list = action_data.nunique()[["feat", "first_level_category_id", "second_level_category_id", "third_level_category_id", "author_id"]].to_numpy()


    if num_id_per_each_feature_list.shape[0] == conf.dim_feature:
        pass
    else:
        raise 


    if conf.early_generate_data == True:
        one_hot_features = [
            np.eye(num_id_per_each_feature_list[i])[action_feature_list[:, i]] 
            for i in range(dim_feature)
        ]
        action_indicator_all = np.concatenate(one_hot_features, axis=1)
        action_indicator_all = np.expand_dims(action_indicator_all, 0)
        Gamma_matrix = None
        action_indicator_all_T = None
    else:
        action_indicator_all, zero_columns = get_action_indicator_all(dim_feature, s, action_feature_list, num_id_per_each_feature_list)


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

            # Calculate Gamma_matrix for the current batch
            Gamma_matrix_batch = Gamma_inv_batch @ action_selected

            # Store the batch result in the list
            Gamma_matrices.append(Gamma_matrix_batch)

        # Concatenate all batches
        Gamma_matrix = np.concatenate(Gamma_matrices, axis=0)

    
    # Create dataset for the OPL methods that can consider action feature
    dataset_all = dict(
        num_data = num_data,
        num_actions_all = num_actions_all,
        num_id_per_each_feature_list = num_id_per_each_feature_list, 
        dim_feature = dim_feature, 
        s = s, 
        x = x,
        a = a_in_A_all, 
        r = r,
        pi_0 = pi_0_A_all, 
        pscore = pi_0[idx, a],
        q_x_a = q_x_a,
        one_hot_a = one_hot_a, 
        one_hot_a_all = one_hot_a_all, 
        Gamma_matrix = Gamma_matrix,
        action_feature_all = action_indicator_all.reshape(num_actions_all, -1)[:, :num_id_per_each_feature_list.sum()],
        action_indicator_all_T = action_indicator_all_T
    )

    # Create dataset for the OPL methods that only consider the exsting actions
    dataset_logging = dict(
        num_data = num_data,
        num_action = num_actions,
        num_id_per_each_feature_list = num_id_per_each_feature_list, 
        dim_feature = dim_feature, 
        s = s, 
        x = x,
        a = a, 
        r = r,
        pi_0 = pi_0, 
        pscore = pi_0[idx, a],
        q_x_a = q_x_a_logging,
        one_hot_a = one_hot_a, 
        one_hot_a_all = one_hot_a_all, 
    )
    return dataset_all, dataset_logging



def get_indices_from_id_list(id_list, max_id_list):

    list = []
    for id in id_list:
        index = 0
        multiplier = 1
        for id_val, max_id in zip(reversed(id), reversed(max_id_list)):
            index += id_val * multiplier
            multiplier *= (max_id)
        list.append(index)

    return np.array(list)


def get_action_indicator_all(d, s, action_feature_list, num_id_per_each_feature_list):

    # one-hot encoding
    one_hot_features = [
        np.eye(num_id_per_each_feature_list[i])[action_feature_list[:, i]] 
        for i in range(d)
    ]
    one_hot_features_combined = np.concatenate(one_hot_features, axis=1)

    # combination_id
    combination_ids = get_indices_from_id_list(action_feature_list[:, :s], num_id_per_each_feature_list[:s])

    # one_hot_combination
    n_combination = np.prod(num_id_per_each_feature_list[:s])
    one_hot_combinations = np.eye(n_combination)[combination_ids]

    action_indicator_all = np.hstack([one_hot_features_combined, one_hot_combinations])

    # Identify columns where all values are 0.
    zero_columns = np.where(np.all(action_indicator_all == 0, axis=0))[0]

    # Remove columns where all values are 0.
    action_indicator_all_cleaned = np.delete(action_indicator_all, zero_columns, axis=1)


    return np.expand_dims(action_indicator_all_cleaned, axis=2), zero_columns
