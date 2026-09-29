from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import softmax, RegBasedPolicyDataset, GradientBasedPolicyDataset, LCPIDataset, fix_seed, calc_policy_value



@dataclass
class RegBased:
    dim_x: int # d_x
    num_actions: int # |A|
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    beta: float = 100
    learning_rate_init: float = 0.01
    flag_show_curve: bool = True
    alpha: float = 1e-6 # weight decay used in optimizer
    log_eps: float = 1e-10 # this variable is not used in RegBasedPolicyLearner
    var_reg_param: float = 0.01
    solver: str = "adagrad"
    max_iter: int = 50 # number of epochs
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        fix_seed(seed=self.random_state)
        layer_list = []
        input_size = self.dim_x + self.num_actions

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, 1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        one_hot_a = dataset["one_hot_a"] 

        # Instantiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(x, a, r)

        # start policy training
        q_x_a_train, q_x_a_test = dataset["q_x_a"], dataset_test["q_x_a"]
        
        for _ in tqdm(range(self.max_iter), desc=f"RegBased (a) learning"):
            loss_epoch = []

            # Start training mode
            self.nn_model.train()
            for x, a, r in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                a_context = torch.from_numpy(one_hot_a[a]).float()
                input = torch.cat((x, a_context), dim=1)
                # Calculate the estimated expected reward (batch size * |A|)
                q_hat = self.nn_model(input)

                # Calculate the loss
                loss = ((r - q_hat) ** 2).mean()

                loss += self.var_reg_param * torch.var(q_hat)
 
                # Calculate the gradient using backward propagation
                loss.backward()
                # Update the parameters using the calculated gradient 
                optimizer.step()
                loss_epoch.append(loss.item())
                


            if self.flag_show_curve:
                # Calculate the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calculate the true value of the trained policy
                self.train_value.append(calc_policy_value(pi_train, q_x_a_train, random_state=self.random_state))
                self.train_loss.append(np.mean(loss_epoch))               

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
    ) -> tuple:
        dataset = RegBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:
        # Set the evaluation mode
        self.nn_model.eval()

        x = dataset_test["x"]
        one_hot_a = dataset_test["one_hot_a"]

        num_data = x.shape[0]

        x_expanded = x.repeat(self.num_actions, axis = 0)
        one_hot_a_expanded = np.tile(one_hot_a, (num_data, 1))

        input = np.hstack((x_expanded, one_hot_a_expanded))
        input = torch.from_numpy(input).float()
        q_hat = self.nn_model(input).detach().numpy()
        q_hat = q_hat.reshape(num_data, self.num_actions)

        return softmax(self.beta * q_hat)

    # This method is not used in RegBasedPolicyLearner
    def predict_q(self, dataset_test: np.ndarray) -> np.ndarray:
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()

        return self.nn_model(x).detach().numpy()



@dataclass
class RegBasedActionFeature:
    dim_x: int # d_x
    num_actions: int # |A|
    num_id_per_each_feature: int # |F_l|
    dim_feature: int # d_f
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    beta: float = 100
    learning_rate_init: float = 0.01
    flag_show_curve: bool = True
    alpha: float = 1e-6 # weight decay used in optimizer
    var_reg_param: float = 0.01
    log_eps: float = 1e-10 # this variable is not used in RegBasedPolicyLearner
    solver: str = "adagrad"
    max_iter: int = 50 # number of epochs
    n_iter_no_change: int = 10
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        fix_seed(seed=self.random_state)
        layer_list = []
        input_size = self.dim_x + self.num_id_per_each_feature * self.dim_feature
        self.num_actions_all = self.num_id_per_each_feature ** self.dim_feature

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, 1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        one_hot_mat_f = dataset["one_hot_mat_f"] 

        # Instantiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(x, a, r)

        # start policy training
        q_x_a_train, q_x_a_test = dataset["q_x_a"], dataset_test["q_x_a"]
        for _ in tqdm(range(self.max_iter), desc=f"RegBased (f) learning"):

            loss_epoch = []
            n_not_improving_training = 0
            previous_training_loss = None

            # Start training mode
            self.nn_model.train()
            for x, a, r in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                batch_size = x.shape[0]
                f_context = torch.from_numpy(one_hot_mat_f[a].reshape(batch_size, self.num_id_per_each_feature * self.dim_feature)).float()
                input = torch.cat((x, f_context), dim=1)
                # Calculate the estimated expected reward (batch size * |A|)
                q_hat = self.nn_model(input)
                # Calculate the loss
                loss = ((r - q_hat) ** 2).mean()
                loss += self.var_reg_param * torch.var(q_hat)
                # Calculate the gradient using backward propagation
                loss.backward()
                # Update the parameters using the calculated gradient 
                optimizer.step()
                loss_epoch.append(loss.item())
            

            if self.flag_show_curve:
                # Calculate the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calculate the true value of the trained policy
                self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

                # Calculate the trained policy using the test data
                pi_test = self.predict(dataset_test)
                # Calculate the true value of the test policy
                self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

                self.train_loss.append(np.mean(loss_epoch))

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
    ) -> tuple:
        dataset = RegBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:
        # Set the evaluation mode
        self.nn_model.eval()

        x = dataset_test["x"]
        one_hot_mat_f = dataset_test["one_hot_mat_f"]
        one_hot_mat_f = one_hot_mat_f.reshape(self.num_actions_all, self.num_id_per_each_feature * self.dim_feature)

        num_data = x.shape[0]

        x_expanded = x.repeat(self.num_actions_all, axis = 0)
        one_hot_f_expanded = np.tile(one_hot_mat_f, (num_data, 1))

        input = np.hstack((x_expanded, one_hot_f_expanded))
        input = torch.from_numpy(input).float()
        q_hat = self.nn_model(input).detach().numpy()
        q_hat = q_hat.reshape(num_data, self.num_actions_all)

        return softmax(self.beta * q_hat)

    # This method is not used in RegBasedPolicyLearner
    def predict_q(self, dataset_test: np.ndarray) -> np.ndarray:
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()

        return self.nn_model(x).detach().numpy()




@dataclass
class GradientBasedPolicyLearner:
    dim_x: int # d_x
    num_actions: int # |A|
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    batch_size_reg: int = 32
    learning_rate_init: float = 0.01
    learning_rate_init_reg: float = 0.01
    beta_reg: int = 100
    flag_show_curve: bool = True
    alpha: float = 1e-6 # weight decay used in optimizer
    imit_reg: float = 0.0 #### this is the different param from RegBased
    var_reg_param: float = 0.01
    log_eps: float = 1e-10 # to calculate the log the input of the log should not be zero so engineering trick
    solver: str = "adagrad"
    solver_reg: str = "adagrad"
    max_iter: int = 50 # number of epochs
    max_iter_reg: int = 50 # number of epochs
    random_state: int = 12345
    dr: bool = False
    flag_f_model: bool =True

    def __post_init__(self) -> None:
        """Initialize class."""
        fix_seed(seed=self.random_state)
        layer_list = []
        f_layer_list = []

        input_size = self.dim_x
        f_input_size = self.dim_x + self.num_actions

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1))) 
        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        fix_seed(seed=self.random_state)
        for i, h in enumerate(self.hidden_layer_size):
            f_layer_list.append(("l{}".format(i), nn.Linear(f_input_size, h)))
            f_layer_list.append(("a{}".format(i), activation_layer()))
            f_input_size = h
        f_layer_list.append(("output", nn.Linear(f_input_size, 1)))
        self.f_model = nn.Sequential(OrderedDict(f_layer_list))

        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.test_value = []
        self.train_bias= []
        self.train_variance = []


    def fit(self, dataset: dict, dataset_test: dict, q_hat: np.ndarray = None) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        pscore, pi_0 = dataset["pscore"], dataset["pi_0"] 
        one_hot_a = dataset["one_hot_a"] 
        ####### this is also the difference from Regression based Policy Learner ######
        if q_hat is None:
            q_hat = np.zeros((r.shape[0], self.num_actions))

        # Instantiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")


        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore,
            q_hat,
            pi_0,
        )

        # start policy training
        q_x_a_train, q_x_a_test = dataset["q_x_a"], dataset_test["q_x_a"]
        if self.dr == True:
            name = "DR"
            if self.flag_f_model:
                reg_training_data_loader = self.reg_create_train_data_for_opl(
                    x,
                    a,
                    r,
                )
                self.train_reward_model(one_hot_a, reg_training_data_loader)
        
        else:
            name = "IPS"

        for _ in tqdm(range(self.max_iter), desc=f"{name}-PG (a) learning"):

            loss_epoch=[]

            # Start training mode
            self.nn_model.train()
            for x, a, r, p, q_hat_, pi_0_ in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                # Calculate the estimated trained policy at time _ (batch size * |A|)
                pi = self.nn_model(x)
                if self.flag_f_model:
                    a_context = torch.from_numpy(one_hot_a).float()
                    a_context_ = a_context.unsqueeze(0).repeat(x.shape[0], 1,  1)
                    x_ = x.unsqueeze(1).repeat(1, self.num_actions, 1)

                    input = torch.cat((x_, a_context_), dim=2)

                    q_hat_ = self.f_model(input).squeeze(2).detach()
                    
                # Calculate the loss
                policy_grad_arr = -self._estimate_policy_gradient(
                    a=a,
                    r=r,
                    pscore=p,
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                )
                loss = policy_grad_arr.mean()
                loss += self.var_reg_param * torch.var(policy_grad_arr)
                # Calculate the gradient using backward propagation
                loss.backward()
                # Update the parameters using the calculated gradient 
                optimizer.step()

            if self.flag_show_curve:    
                # Calculate the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calculate the true value of the trained policy
                self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

                # Calculate the trained policy using the test data
                pi_test = self.predict(dataset_test)
                # Calculate the true value of the test policy
                self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

                self.train_loss.append(np.mean(loss_epoch))


                # bias
                self.train_bias.append(((q_x_a_train * pi_train * np.log(pi_train)).sum(1).mean() + policy_grad_arr.mean().item())**2)
                # variance
                self.train_variance.append(((policy_grad_arr**2).mean() - policy_grad_arr.mean()**2).detach().numpy())


    def train_reward_model(
        self,
        one_hot_a,
        training_data_loader: torch.utils.data.dataloader,
    ) -> None:

        # Instantiate the solver (adagrad or adam)
        if self.solver_reg == "adagrad":
            optimizer = optim.Adagrad(
                self.f_model.parameters(),
                lr=self.learning_rate_init_reg,
                weight_decay=self.alpha,
            )
        elif self.solver_reg == "adam":
            optimizer = optim.AdamW(
                self.f_model.parameters(),
                lr=self.learning_rate_init_reg,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        # start policy training
        for _ in tqdm(range(self.max_iter_reg), desc=f"q_hat learning"):

            # Start training mode
            self.f_model.train()
            for x, a, r in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                a_context = torch.from_numpy(one_hot_a[a]).float()
                input = torch.cat((x, a_context), dim=1)
                # Calculate the estimated expected reward (batch size * |A|)
                q_hat = self.f_model(input)
                # Calculate the loss
            
                loss = ((r - q_hat) ** 2).mean()

                # Calculate the gradient using backward propagation
                loss.backward()
                # Update the parameters using the calculated gradient 
                optimizer.step()
         
    
    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        pi_0: np.ndarray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def reg_create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
    ) -> tuple:
        dataset = RegBasedPolicyDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_reg,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)
        iw = current_pi[idx, a] / pscore

        if self.dr != True:
            estimated_policy_grad_arr = iw * r * log_prob[idx, a]

        else:

            q_hat_factual = q_hat[idx, a]
            estimated_policy_grad_arr = iw * (r - q_hat_factual) * log_prob[idx, a]
            estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)


        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()




@dataclass
class LCPI:
    dim_x: int # d_x
    num_actions: int # |A|
    dim_feature: int # d_f
    num_id_per_each_feature: int # |F_j|
    s: int # s
    one_hot_mat_f: np.ndarray
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    learning_rate_init: float = 0.01
    flag_show_curve: bool = True
    alpha: float = 1e-6 # weight decay used in optimizer
    imit_reg: float = 0.0 #### this is the different param from RegBased
    var_reg_param: float = 0.01
    log_eps: float = 1e-10 # to calculate the log the input of the log should not be zero so engineering trick
    solver: str = "adagrad"
    max_iter: int = 50 # number of epochs
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        fix_seed(seed=self.random_state)
        layer_list = []
        input_size = self.dim_x
        self.one_hot_mat_f = torch.Tensor(self.one_hot_mat_f)

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.train_bias = []
        self.train_variance = []
        self.test_value = []


    def fit(self, dataset: dict, dataset_test: dict, q_hat: np.ndarray = None) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        pscore, pi_0 = dataset["pscore"], dataset["pi_0"]
        pscore_f, pi_0_f = dataset["pscore_f"], dataset["pi_0_f"]
        pscore_f_up_to_s = dataset["pscore_f_up_to_s"]
        action_indicator_all_T = torch.from_numpy(dataset["action_indicator_all_T"]).float()


        # Calculate importance weights
        Gamma_matrix = dataset["Gamma_matrix"]
        
        # if \hat{q}(x, a) is not provided, then create zero vector
        if q_hat is None:
            q_hat = np.zeros((r.shape[0], self.num_actions))

        # Instantiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")


        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore,
            pscore_f, 
            pscore_f_up_to_s, 
            q_hat,
            pi_0,
            pi_0_f,
            Gamma_matrix,
        )


        # start policy training
        q_x_a_train, q_x_a_test = dataset["q_x_a"], dataset_test["q_x_a"]
        for _ in tqdm(range(self.max_iter), desc=f"LCPI-PG learning"):
        
            loss_epoch = []
            # Start training mode
            self.nn_model.train()
            for x, a, r, p, p_f, p_f_s, q_hat_, pi_0_, pi_0_f_, Gamma_matrix in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                # Calculate the estimated trained policy at time _ (batch size * |A|)
                pi = self.nn_model(x)
            
                # Calculate the loss
                policy_grad_arr = -self._estimate_policy_gradient(
                    a=a,
                    r=r,
                    pscore=p,
                    pscore_f=p_f, 
                    pscore_f_up_to_s=p_f_s, 
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi_0_f=pi_0_f_, 
                    pi=pi,
                    Gamma_matrix=Gamma_matrix,
                    action_indicator_all_T=action_indicator_all_T
                )
                loss = policy_grad_arr.mean()
    
                # Calculate the gradient using backward propagation
                loss.backward()
                # Update the parameters using the calculated gradient 
                optimizer.step()
                loss_epoch.append(loss.item())

            if self.flag_show_curve:
                
                # Calculate the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calculate the true value of the trained policy
            
                self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

                # Calculate the trained policy using the test data
                pi_test = self.predict(dataset_test)
                # Calculate the true value of the test policy
                self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

                self.train_loss.append(np.mean(loss_epoch))

                # bias
                self.train_bias.append(((q_x_a_train * pi_train * np.log(pi_train)).sum(1).mean() + policy_grad_arr.mean().item())**2)
                # variance
                
                self.train_variance.append(((policy_grad_arr**2).mean() - policy_grad_arr.mean()**2).detach().numpy())

    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        pscore_f: np.ndarray, 
        pscore_f_up_to_s: np.ndarray, 
        q_hat: np.ndarray,
        pi_0: np.ndarray,
        pi_0_f: np.ndarray, 
        Gamma_matrix: np.ndarray, 
    ) -> tuple:
        dataset = LCPIDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(pscore_f).float(),
            torch.from_numpy(pscore_f_up_to_s).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
            torch.from_numpy(pi_0_f).float(),
            torch.from_numpy(Gamma_matrix).float()
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader


    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        pscore_f: torch.Tensor,
        pscore_f_up_to_s: torch.Tensor, 
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
        pi_0_f: torch.Tensor, 
        Gamma_matrix: torch.Tensor, 
        action_indicator_all_T: torch.Tensor, 
    ) -> torch.Tensor:

        pi = pi.unsqueeze(2).unsqueeze(3)
        current_pi = pi.detach()
        
        iw = (current_pi * torch.log(pi + self.log_eps) * action_indicator_all_T).sum(dim=1)@ Gamma_matrix
        estimated_policy_grad_arr =  iw.reshape(-1) * r 

        return estimated_policy_grad_arr



    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()



@dataclass
class PONA:
    dim_x: int # d_x
    num_actions: int # |A|
    dim_feature: int # d_f
    num_id_per_each_feature: int # |F_j|
    s: int # s
    one_hot_mat_f: np.ndarray
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    batch_size_reg: int = 32
    learning_rate_init: float = 0.01
    learning_rate_init_reg: float = 0.01
    beta_reg: int = 100
    flag_show_curve: bool = True
    alpha: float = 1e-6 # weight decay used in optimizer
    kappa: float = 0.5 # percentage of Proposed and IPS
    imit_reg: float = 0.0 #### this is the different param from RegBased
    var_reg_param: float = 0.01
    log_eps: float = 1e-10 # to calculate the log the input of the log should not be zero so engineering trick
    solver: str = "adagrad"
    solver_reg: str = "adagrad"
    max_iter: int = 50 # number of epochs
    max_iter_reg: int = 50 # number of epochs
    random_state: int = 12345
    dr: bool = False

    def __post_init__(self) -> None:
        """Initialize class."""
        fix_seed(seed=self.random_state)
        layer_list = []
        f_layer_list = []
        input_size = self.dim_x
        f_input_size = self.dim_x + self.num_actions

        self.one_hot_mat_f = torch.Tensor(self.one_hot_mat_f)
        self.num_actions_all = self.num_id_per_each_feature ** self.dim_feature

        # Set the activation layer (Tanh, ReLU, or ELU)
        # Default ELU
        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        # Define the model
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))  
        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        fix_seed(seed=self.random_state)
        for i, h in enumerate(self.hidden_layer_size):
            f_layer_list.append(("l{}".format(i), nn.Linear(f_input_size, h)))
            f_layer_list.append(("a{}".format(i), activation_layer()))
            f_input_size = h
        f_layer_list.append(("output", nn.Linear(f_input_size, 1)))
        self.f_model = nn.Sequential(OrderedDict(f_layer_list))


        self.random_ = check_random_state(self.random_state)

        # the length of the vector is the number of epochs
        self.train_loss = []
        self.train_value = []
        self.train_bias = []
        self.train_variance = []
        self.test_value = []


    def fit(self, dataset: dict, dataset_test: dict, q_hat: np.ndarray = None) -> None:
        x, a, r = dataset["x"], dataset["a"], dataset["r"]
        one_hot_mat_f = dataset["one_hot_mat_f"] 
        pscore, pi_0 = dataset["pscore"], dataset["pi_0"] 
        one_hot_a = dataset["one_hot_a_all"]
        ####### this is also the difference from Regression based Policy Learner ######
        pscore_f, pi_0_f = dataset["pscore_f"], dataset["pi_0_f"]
        pscore_f_up_to_s = dataset["pscore_f_up_to_s"]
        Gamma_matrix = dataset["Gamma_matrix"]
        action_indicator_all_T = torch.from_numpy(dataset["action_indicator_all_T"]).float()
        # if \hat{q}(x, a) is not provided, then create zero vector
        if q_hat is None:
            q_hat = np.zeros((r.shape[0], self.num_actions))

        # Instantiate the solver (adagrad or adam)
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")


        # Create the training data loader
        training_data_loader = self._create_train_data_for_opl(
            x,
            a,
            r,
            pscore,
            pscore_f, 
            pscore_f_up_to_s, 
            q_hat,
            pi_0,
            pi_0_f, 
            Gamma_matrix,
        )

        # start policy training
        q_x_a_train, q_x_a_test = dataset["q_x_a"], dataset_test["q_x_a"]
        if self.dr == True:
            name = "DR"
            self.train_reward_model(one_hot_a, training_data_loader)
        
        else:
            name = "IPS"


        for _ in tqdm(range(self.max_iter), desc=f"PONA({name}, kappa={self.kappa}) learning"):
        
            loss_epoch = []
            # Start training mode
            self.nn_model.train()
            for x, a, r, p, p_f, p_f_s, q_hat_, pi_0_, pi_0_f_, Gamma_matrix in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                # Calculate the estimated trained policy at time _ (batch size * |A|)
                pi = self.nn_model(x)

                batch_size = x.shape[0]
                a_context = torch.from_numpy(one_hot_a).float()
                a_context_ = a_context.unsqueeze(0).repeat(x.shape[0], 1,  1)
                x_ = x.unsqueeze(1).repeat(1, self.num_actions, 1)

                input = torch.cat((x_, a_context_), dim=2)

        
                # Calculate the estimated expected reward (batch size * |A|)
                q_hat_ = self.f_model(input).squeeze(2).detach()


                # Calculate the loss
                policy_grad_arr = -self._estimate_policy_gradient(
                    a=a,
                    r=r,
                    pscore=p,
                    pscore_f=p_f, 
                    pscore_f_up_to_s=p_f_s, 
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi_0_f=pi_0_f_, 
                    pi=pi,
                    Gamma_matrix=Gamma_matrix,
                    action_indicator_all_T=action_indicator_all_T
                )
                loss = policy_grad_arr.mean()
                loss += self.var_reg_param * torch.var(policy_grad_arr)
                # Calculate the gradient using backward propagation
                loss.backward()
                # Update the parameters using the calculated gradient 
                optimizer.step()
                loss_epoch.append(loss.item())

            if self.flag_show_curve:
                
                # Calculate the trained policy using the training data
                pi_train = self.predict(dataset)
                # Calculate the true value of the trained policy
            
                self.train_value.append((q_x_a_train * pi_train).sum(1).mean())

                # Calculate the trained policy using the test data
                pi_test = self.predict(dataset_test)
                # Calculate the true value of the test policy
                self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

                self.train_loss.append(np.mean(loss_epoch))

                # bias
                self.train_bias.append(((q_x_a_train * pi_train * np.log(pi_train)).sum(1).mean() + loss.item())**2)
                # variance
                
                self.train_variance.append(((policy_grad_arr**2).mean() - policy_grad_arr.mean()**2).detach().numpy())


    def train_reward_model(
        self,
        one_hot_a,
        training_data_loader: torch.utils.data.dataloader,
    ) -> None:

        # Instantiate the solver (adagrad or adam)
        if self.solver_reg == "adagrad":
            optimizer = optim.Adagrad(
                self.f_model.parameters(),
                lr=self.learning_rate_init_reg,
                weight_decay=self.alpha,
            )
        elif self.solver_reg == "adam":
            optimizer = optim.AdamW(
                self.f_model.parameters(),
                lr=self.learning_rate_init_reg,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        # start policy training
        for _ in tqdm(range(self.max_iter_reg), desc=f"q_hat learning"):

            # Start training mode
            self.f_model.train()
            for x, a, r, p, p_f, p_f_s, q_hat_, pi_0_, pi_0_f_, Gamma_matrix in training_data_loader:
                # Set the gradient to zero
                optimizer.zero_grad()
                batch_size = x.shape[0]
                a_context = torch.from_numpy(one_hot_a[a]).float()
                input = torch.cat((x, a_context), dim=1)
                # Calculate the estimated expected reward (batch size * |A|)
                q_hat = self.f_model(input)
                # Calculate the loss
                loss = ((r - q_hat) ** 2).mean()
    
                # Calculate the gradient using backward propagation
                loss.backward()
                # Update the parameters using the calculated gradient 
                optimizer.step()
    def _create_train_data_for_opl(
        self,
        x: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        pscore: np.ndarray,
        pscore_f: np.ndarray, 
        pscore_f_up_to_s: np.ndarray, 
        q_hat: np.ndarray,
        pi_0: np.ndarray,
        pi_0_f: np.ndarray, 
        Gamma_matrix: np.ndarray, 
    ) -> tuple:
        dataset = LCPIDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(a).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(pscore_f).float(),
            torch.from_numpy(pscore_f_up_to_s).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
            torch.from_numpy(pi_0_f).float(),
            torch.from_numpy(Gamma_matrix).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader


    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        pscore_f: torch.Tensor,
        pscore_f_up_to_s: torch.Tensor, 
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
        pi_0_f: torch.Tensor, 
        Gamma_matrix: torch.Tensor, 
        action_indicator_all_T: torch.Tensor, 
    ) -> torch.Tensor:
        current_pi = pi.detach()
        batch_size = current_pi.shape[0]


        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)
        iw = current_pi[idx, a] / pscore

        if self.dr != True:
            old_action_grad_arr = iw * r * log_prob[idx, a]

        else:
            q_hat_factual = q_hat[idx, a]
            old_action_grad_arr = iw * (r - q_hat_factual) * log_prob[idx, a]
            old_action_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)


        if self.kappa==0:
            estimated_policy_grad_arr = old_action_grad_arr
            return estimated_policy_grad_arr
        
        else:
            pi = pi.unsqueeze(2).unsqueeze(3)
            current_pi = pi.detach()

            iw = (current_pi * torch.log(pi + self.log_eps) * action_indicator_all_T).sum(dim=1)@ Gamma_matrix
            lcpi_grad_arr =  iw.reshape(-1) * r 

            estimated_policy_grad_arr = (self.kappa*lcpi_grad_arr + (1-self.kappa)*old_action_grad_arr) 

            return estimated_policy_grad_arr



    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["x"]).float()
        return self.nn_model(x).detach().numpy()

