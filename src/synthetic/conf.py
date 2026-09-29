dim_context = 5
dim_feature = 5
num_id_per_each_feature = 3
num_actions_all = num_id_per_each_feature ** dim_feature

num_actions = int(num_actions_all*0.8)
num_new_actions_percentage_list = [10, 15, 20, 25, 30]

eta_f_var = 0
beta = 0.05
beta_list = [-0.2, -0.1, 0, 0.1, 0.2]
gamma = 0.5
gamma_list = [0, 0.25, 0.5, 0.75, 1.0]
eta = 0.5

s = 3
s_list = [1, 2, 3, 4]

kappa = 0.3
kappa_list = [0.2, 0.4, 0.6, 0.8]
tuning_kappa_list = [0, 0.25, 0.5, 0.75, 1.0]
rho_U = 0.1
rho_U_list = [0, 0.05, 0.1, 0.15, 0.2]
rho_L = 0.1
rho_L_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

num_train = 2000
num_test = 2000
num_trains_list = [500, 1000, 2000, 4000]
num_seeds = 200

reward_type = "action"

dr_noise = 0

lambda_ = 0
lambda_list = [-1, -0.5, 0, 0.5, 1]


flag_uniform_context = False
flag_q_l_x_f_l_simple = True


num_epochs_reg = 20
learning_rate_init_reg = 5e-3
beta_reg = 100
solver_reg = "adam"
batch_size_reg = 16

num_epochs = 20
learning_rate_init = 2e-4
solver = "adam"
batch_size = 32

random_state = 12345

flag_f_model = True
flag_show_curve = False
flag_show_bias_variance = False
flag_calc_optimal = False


learner_list = ["Logging", "Random", "RegBased (a)", "RegBased (f)", "IPS-PG (a)", "DR-PG (a)","LCPI-PG", "PONA"]
show_learner_list = ["Logging",  "RegBased (a)", "RegBased (f)", "IPS-PG (a)",  "DR-PG (a)","LCPI-PG", "PONA"]

show_relative = "Random"
show_log = "result"

flag_show_execution_time = True
fig_width = 17
fig_height = 7
fig_width_two = 17
fig_height_two = 7
fig_width_three = 19
fig_height_three = 5
fig_width_three_experiment_proposed_w_varying_s = 33
fig_height_three_experiment_proposed_w_varying_s = 8
linewidth = 7
marker = "o"
markersize = 15
fontsize_ylabel = 25
fontsize_ylabel_two = 30
fontsize_ylabel_three = 20
labelsize_yticks = 25
labelsize_yticks_three = 18
fontsize_xlabel = 30
fontsize_xlabel_two = 25
fontsize_xlabel_three = 18
labelsize_xticks = 30
labelsize_xticks_three = 18
fontsize_legend = 25
fontsize_legend_three = 20
width_show_loss_value = 9
height_show_loss_value = 4


color_dict = {"Logging": "tab:gray", "Optimal": "tab:pink", "Random": "tab:brown", "RegBased (a)": "tab:red", "RegBased (f)": "tab:orange", "PolicyBased (IPS)": "tab:blue", "PolicyBased (DR)": "tab:purple", "IPS-PG (a)": "tab:blue", "DR-PG (a)": "tab:purple", "LCPI-PG": "tab:pink", "PONA": "green", "PONA ($\rho_U$)": "goldenrod", "PONA ($\rho_L$)": "goldenrod", "PONA ($\rho_L=-\infty$)": "green", "PONA ($\rho_U=\infty$)": "green", "PONA (varying $\rho_L$)": "goldenrod", "PONA (varying $\rho_U$)": "goldenrod",  "PONA (rho_U)": "goldenrod", "PONA (rho_L)": "goldenrod", "PONA (rho_L=-infty)": "green", "PONA (rho_U=infty)": "green", "PONA (varying rho_L)": "goldenrod", "PONA (varying rho_U)": "goldenrod", "PONA (kappa=1.0)": "tab:pink", "PONA ($\kappa=1.0$)": "tab:pink", r"PONA ($\rho_U$)": "goldenrod", r"PONA ($\rho_L$)": "goldenrod", r"PONA ($\rho_L=-\infty$)": "green", r"PONA ($\rho_U=\infty$)": "green", r"PONA (varying $\rho_L$)": "goldenrod", r"PONA (varying $\rho_U$)": "goldenrod", }

