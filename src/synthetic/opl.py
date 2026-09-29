from sklearn.utils import check_random_state
from policylearners import (
    RegBased, 
    RegBasedActionFeature,
    GradientBasedPolicyLearner,
    LCPI,
    PONA,
)
import pandas as pd
import conf
from utils import gen_eps_greedy
from visualization import show_loss, show_value, show_bias, show_variance
import numpy as np
from sklearn.model_selection import KFold


# Calculate the following four evaluation metrics for a given learned policy, only for the methods which can select new actions
def evaluate_policy(pi, 
                    label, 
                    result_data, 
                    dataset_test_all, 
                    set_of_actions = None, 
                    set_of_new_actions = None, 
                    estimated_policy_value = None, 
                    ):

    V = (dataset_test_all["q_x_a"] * pi).sum(1).mean()

    if label  in ["Logging", "IPS-PG (a)", "DR-PG (a)", "RegBased (a)"]:
        existing_action_ratio = 1
        new_action_ratio = 0
        existing_V = V
        new_V = 0
        per_existing_V = V
        per_new_V = 0

    else:
        vec_existing_action_ratio = pi[:, set_of_actions].sum(axis = 1)
        vec_new_action_ratio = 1 - vec_existing_action_ratio

        vec_existing_V = (dataset_test_all["q_x_a"] * pi)[:, set_of_actions].sum(axis = 1)
        vec_new_V = (dataset_test_all["q_x_a"] * pi)[:, set_of_new_actions].sum(axis = 1)

        existing_action_ratio = vec_existing_action_ratio.mean()
        new_action_ratio = vec_new_action_ratio.mean()
        existing_V = vec_existing_V.mean()
        new_V = vec_new_V.mean()
        per_existing_V = (vec_existing_V / vec_existing_action_ratio).mean()
        per_new_V = (vec_new_V / vec_new_action_ratio).mean()
    


    new_row = pd.DataFrame([{
        "method": label,
        "V": V,
        "existing_action_ratio": existing_action_ratio,
        "new_action_ratio": new_action_ratio,
        "existing_V": existing_V,
        "new_V": new_V,
        "per_existing_V": per_existing_V,
        "per_new_V": per_new_V,
        "estimated_V": estimated_policy_value,
    }])

    result_data = pd.concat([result_data, new_row], axis = 0)

    return result_data



def run_opl(
    dataset_train_all, 
    dataset_test_all, 
    dataset_train_logging, 
    dataset_test_logging, 
    set_of_actions, 
    num_actions_all, 
    num_actions, 
    kappa,
    random_state, 
):
        
    num_test = dataset_test_all["num_data"]
    num_train = dataset_train_all["num_data"]
    dim_x = dataset_train_all["x"].shape[1]
    dim_feature = dataset_test_all["dim_feature"]
    s = dataset_test_all["s"]

    result_data = pd.DataFrame([], columns = ["method", "V", "existing_action_ratio", "new_action_ratio", "existing_V", "new_V", "per_existing_V", "per_new_V", "estimated_V"])


    set_of_all_actions = np.arange(num_actions_all) # the set of all actions: A_all
    mask = np.isin(set_of_all_actions, set_of_actions, invert=True)
    set_of_new_actions = set_of_all_actions[mask] # the set of new actions: A_new
    
    random_ = check_random_state(random_state)


    if "Logging" in conf.learner_list:
        result_data = evaluate_policy(dataset_test_all["pi_0"], "Logging", result_data, dataset_test_all)

    # Optimal
    if "Optimal" in conf.learner_list:
        pi_best = gen_eps_greedy(expected_reward=dataset_test_all["q_x_a"], is_optimal=True, eps=0)
        result_data = evaluate_policy(pi_best, "Optimal", result_data, dataset_test_all, set_of_actions, set_of_new_actions)

    # Random
    if "Random" in conf.learner_list:
        pi_random = np.ones((num_test, num_actions_all)) / num_actions_all
        result_data = evaluate_policy(pi_random, "Random", result_data, dataset_test_all, set_of_actions, set_of_new_actions)



    ####################### END Optimal Policy #######################

    reg_action, reg_feature, ips_action, dr_action, lcpi, pona = None, None, None, None, None, None,


    ####################### START Regression-based method based on action a #######################
    if "RegBased (a)" in conf.learner_list:
        reg_action = RegBased(dim_x=dim_x, 
                            num_actions=num_actions, 
                            max_iter=conf.num_epochs_reg, 
                            random_state=random_state, 
                            solver = conf.solver_reg, 
                            learning_rate_init = conf.learning_rate_init_reg, 
                            batch_size = conf.batch_size_reg,
                            beta = conf.beta_reg,
                            flag_show_curve = conf.flag_show_curve,
                            )
        reg_action.fit(dataset_train_logging, dataset_test_logging)
        pi_reg_action = reg_action.predict(dataset_test_logging)
        result_data = evaluate_policy(pi_reg_action, 
                            "RegBased (a)", 
                            result_data,
                            dataset_test_logging, 
                            )

    ####################### END Regression-based method based on action a #######################



    ####################### START IPS-PG (a) based on action a #######################
    if "IPS-PG (a)" in conf.learner_list:
        ips_action = GradientBasedPolicyLearner(dim_x=dim_x, 
        num_actions=num_actions, 
        max_iter=conf.num_epochs, 
        random_state=random_state, 
        solver = conf.solver, 
        learning_rate_init = conf.learning_rate_init, 
        batch_size = conf.batch_size,
        flag_show_curve = conf.flag_show_curve,
        flag_f_model = False, 
        )

        ips_action.fit(dataset_train_logging, dataset_test_logging)
        pi_ips_action = ips_action.predict(dataset_test_logging)

        result_data = evaluate_policy(pi_ips_action, 
                            "IPS-PG (a)", 
                            result_data,
                            dataset_test_logging, 
                            )

    ####################### END IPS-PG (a) based on action a #######################


    ####################### START DR-PG (a) based on action a #######################
    if "DR-PG (a)" in conf.learner_list:
        dr_action = GradientBasedPolicyLearner(dim_x=dim_x, 
        num_actions=num_actions, 
        max_iter=conf.num_epochs, 
        max_iter_reg=conf.num_epochs_reg, 
        random_state=random_state, 
        solver = conf.solver, 
        solver_reg = conf.solver_reg, 
        learning_rate_init = conf.learning_rate_init, 
        learning_rate_init_reg = conf.learning_rate_init_reg, 
        batch_size = conf.batch_size,
        batch_size_reg = conf.batch_size_reg,
        flag_show_curve = conf.flag_show_curve,
        dr=True,
        beta_reg=conf.beta_reg,
        flag_f_model= conf.flag_f_model,
        )

        dr_action.fit(dataset_train_logging, dataset_test_logging, dataset_train_logging["q_x_a"] + random_.normal(loc = 0, scale = conf.dr_noise, size =(dataset_train_logging["num_data"], num_actions)))
        pi_dr_action = dr_action.predict(dataset_test_logging)

        result_data = evaluate_policy(pi_dr_action, 
                            "DR-PG (a)", 
                            result_data,
                            dataset_test_logging, 
                            )
        
    ####################### END DR-PG (a) based on action a #######################




    ####################### START Regression-based method based on action feature f #######################
    if "RegBased (f)" in conf.learner_list:
        reg_feature = RegBasedActionFeature(dim_x=dim_x, 
                                    num_actions=num_actions_all, 
                                    num_id_per_each_feature = conf.num_id_per_each_feature, 
                                    dim_feature = conf.dim_feature, 
                                    max_iter=conf.num_epochs_reg, 
                                    random_state=random_state, 
                                    solver = conf.solver_reg, 
                                    learning_rate_init = conf.learning_rate_init_reg, 
                                    batch_size = conf.batch_size_reg,
                                    beta = conf.beta_reg,
                                    flag_show_curve = conf.flag_show_curve,
                                    )
        
        reg_feature.fit(dataset_train_all, dataset_test_all)
        pi_reg_feature = reg_feature.predict(dataset_test_all)

        # Calculate the four evaluation metrics 
        result_data = evaluate_policy(pi = pi_reg_feature, 
                        label = "RegBased (f)", 
                        result_data = result_data,
                        dataset_test_all = dataset_test_all, 
                        set_of_actions = set_of_actions, 
                    set_of_new_actions = set_of_new_actions)
    ####################### END Regression-based method based on action feature f #######################



    ####################### START LCPI method #######################
    if "LCPI-PG" in conf.learner_list:
        lcpi = LCPI(
                    dim_x=dim_x, 
                    dim_feature=conf.dim_feature, 
                    num_id_per_each_feature=conf.num_id_per_each_feature, 
                    s = s, 
                    one_hot_mat_f = dataset_train_all["one_hot_mat_f"], 
                    num_actions=num_actions_all, 
                    max_iter=conf.num_epochs, 
                    random_state=random_state, 
                    solver = conf.solver, 
                    learning_rate_init = conf.learning_rate_init, 
                    batch_size = conf.batch_size,
                    flag_show_curve = conf.flag_show_curve,
                )

        lcpi.fit(dataset_train_all, dataset_test_all)
        pi_lcpi = lcpi.predict(dataset_test_all)

        

        # Calculate the four evaluation metrics 
        result_data = evaluate_policy(pi = pi_lcpi, 
                        label = "LCPI-PG", 
                        result_data = result_data,
                        dataset_test_all = dataset_test_all, 
                        set_of_actions = set_of_actions, 
                        set_of_new_actions = set_of_new_actions)
    ####################### END LCPI method #######################




    ####################### START PONA #######################
    
    if "PONA" in conf.learner_list:
        estimated_policy_value_list, new_action_ratio_list = [], []
        

        valid_keys = ['x', 'a', 'r', 'pi_0', 'pi_0_f', 'pscore', 'pscore_f', 'pscore_f_up_to_s', 'q_x_a', "Gamma_matrix"]
        constant_keys = ['one_hot_mat_f', 'one_hot_a', 'one_hot_a_all', 'dim_feature', 'num_id_per_each_feature', 's', 'num_actions_all', 'num_data', "action_indicator_all_T"]

        num_folds = 3
        kf = KFold(n_splits=num_folds)

        fold_dataset = []
        for _, valid_index in kf.split(dataset_train_all['x']):
            fold_data = {key: dataset_train_all[key][valid_index] if (key in valid_keys) else dataset_train_all[key] for key in dataset_train_all}
            fold_dataset.append(fold_data)



        for i in range(num_folds):
            print(f"[Fold {i+1}/{num_folds}]")
            
            tuning_data_valid = fold_dataset[i]
            
            tuning_data_train = {key: np.concatenate([fold_dataset[j][key] for j in range(num_folds) if j != i]) for key in dataset_train_all if key in valid_keys }
    
            tuning_data_train.update({key: dataset_train_all[key] for key  in constant_keys})

            estimated_policy_value_fold = []
            new_action_ratio_fold = []
            tuning_data_valid["num_data"] = tuning_data_valid["x"].shape[0]
            num_valid = tuning_data_valid["num_data"]
            tuning_data_train["num_data"] = tuning_data_train["x"].shape[0]


            for candidate_kappa in conf.tuning_kappa_list:
                pona = PONA(
                            dim_x=dim_x, 
                            dim_feature=conf.dim_feature, 
                            num_id_per_each_feature=conf.num_id_per_each_feature, 
                            s = s, 
                            kappa = candidate_kappa,
                            one_hot_mat_f = tuning_data_train["one_hot_mat_f"], 
                            num_actions=num_actions_all, 
                            max_iter=conf.num_epochs, 
                            max_iter_reg=conf.num_epochs_reg, 
                            random_state=random_state, 
                            solver = conf.solver, 
                            solver_reg = conf.solver_reg, 
                            learning_rate_init = conf.learning_rate_init, 
                            learning_rate_init_reg = conf.learning_rate_init_reg, 
                            batch_size = conf.batch_size,
                            batch_size_reg = conf.batch_size_reg,
                            flag_show_curve = conf.flag_show_curve,
                            dr = True,
                        )

                pona.fit(tuning_data_train, tuning_data_valid)

                pi_pona_valid = pona.predict(tuning_data_valid)

                pi_pona_valid_new = pi_pona_valid.copy()
                pi_pona_valid_new[:, set_of_actions] = np.zeros((num_valid, num_actions))
                
                
                estimated_policy_value = off_policy_evaluation(
                    dataset=tuning_data_valid, 
                    evaluate_pi=pi_pona_valid, 
                    estimator="LCPI", 
                    )
            

                policy_value_mean = estimated_policy_value.mean()
                new_action_ratio = pi_pona_valid_new.sum(axis=1).mean()

                estimated_policy_value_fold.append(policy_value_mean)
                new_action_ratio_fold.append(new_action_ratio)

                
            estimated_policy_value_list.append((estimated_policy_value_fold))
            new_action_ratio_list.append((new_action_ratio_fold))
        
        estimated_policy_value_list = np.mean(estimated_policy_value_list, axis=0)
        new_action_ratio_list = np.mean(new_action_ratio_list, axis=0)
        
        best_index = np.argmax(estimated_policy_value_list)




        pona = PONA(
                    dim_x=dim_x, 
                    dim_feature=conf.dim_feature, 
                    num_id_per_each_feature=conf.num_id_per_each_feature, 
                    s = s, 
                    kappa = conf.tuning_kappa_list[best_index],
                    one_hot_mat_f = dataset_train_all["one_hot_mat_f"], 
                    num_actions=num_actions_all, 
                    max_iter=conf.num_epochs, 
                    max_iter_reg=conf.num_epochs_reg, 
                    random_state=random_state, 
                    solver = conf.solver, 
                    solver_reg = conf.solver_reg, 
                    learning_rate_init = conf.learning_rate_init, 
                    learning_rate_init_reg = conf.learning_rate_init_reg, 
                    batch_size = conf.batch_size,
                    batch_size_reg = conf.batch_size_reg,                    
                    flag_show_curve = conf.flag_show_curve,
                    dr = True,
                )

        pona.fit(dataset_train_all, dataset_test_all)
        pona_pi = pona.predict(dataset_test_all)

        

        # Calculate the four evaluation metrics 
        result_data = evaluate_policy(pi = pona_pi, 
                        label = "PONA", 
                        result_data = result_data,
                        dataset_test_all = dataset_test_all, 
                        set_of_actions = set_of_actions, 
                        set_of_new_actions = set_of_new_actions)




    learned_dict = {"RegBased (a)":reg_action, "RegBased (f)":reg_feature, "IPS-PG (a)":ips_action, "DR-PG (a)":dr_action, "PONA":pona, "LCPI-PG":lcpi}
    
    # Visualize the learning curve if needed
    if conf.flag_show_curve == True:
        show_loss(conf.learner_list, learned_dict, conf.color_dict)
        show_value(conf.learner_list, learned_dict, conf.color_dict)

        if conf.flag_show_bias_variance == True:
            show_bias(conf.learner_list, learned_dict, conf.color_dict)
            show_variance(conf.learner_list, learned_dict, conf.color_dict)
    # Absolute Policy Value
    # Relative Policy Value compared to the logging policy
    # Relative Policy Value compared to PI-PG
    # Percentage of New Actions
    # Expected Reward by New Actions versus Expected Reward by Existing Actions
    return result_data


def off_policy_evaluation(
    dataset,
    evaluate_pi,
    estimator="LCPI",
    action_indicator_all=None,
    action_indicator_all_T=None,
    action_indicator_cov_mat=None,
):
    if estimator == "IPS":
        a = dataset["a"].astype('int8')
        idx = np.arange(a.shape[0], dtype='int8')
        iw = evaluate_pi[idx, a] / dataset["pscore"]
        
        estimated_policy_value = iw * dataset["r"]

    if estimator == "LCPI":

        a = dataset["a"].astype('int8')
        idx = np.arange(a.shape[0], dtype='int8')

        evaluate_pi = np.expand_dims(np.expand_dims(evaluate_pi, 2), 3)
        
        Gamma_matrix = dataset["Gamma_matrix"]
        action_indicator_all_T = dataset["action_indicator_all_T"]

        # Calculate importance weights
        iw = np.sum(evaluate_pi * action_indicator_all_T, axis=1) @ Gamma_matrix
        estimated_policy_value = iw.reshape(-1) * dataset["r"]

    return estimated_policy_value

