from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from pandas.api.types import CategoricalDtype
import conf

def plot_existing_new(
    all_data,
    x, 
    xticks, 
    xticklabels, 
    xlabels, 
    fig_1, 
    fig_2, 
    fig_3, 
    relative=None,
    flag_legend = True, 
):

    if relative==None:
        pass
    else:
        all_data = relative_by_policy(all_data, x, relative)   

    eliminate_policy = ["Logging", "Random", "RegBased (a)", "IPS-PG (a)",  "DR-PG (a)"]
    all_data.loc[all_data["method"].isin(eliminate_policy), "per_new_V"] = None


    all_data = all_data[all_data['method'].isin(conf.show_learner_list)] 

    cat_type = CategoricalDtype(categories=conf.show_learner_list, ordered=True)
    all_data['method'] = all_data['method'].astype(cat_type)
    all_data = all_data.sort_values(by="method")



    methods = all_data["method"].unique()
    colors = [get_color_with_default(method, conf.color_dict) for method in methods]
    palette = dict(zip(methods, colors))

    num_estimators = len(conf.show_learner_list)

    # Create the legend lines and markers
    line_legend_elements = [
        Line2D([0], [0], color=conf.color_dict[est], linewidth=5, marker='o', markerfacecolor=conf.color_dict[est], markersize=10, label=est) for est in conf.show_learner_list
    ]

    plt.style.use('ggplot')

    fig_width_three = conf.fig_width_three
    fig_height_three = conf.fig_height_three
    fig, ax = plt.subplots(1, 3, figsize=(fig_width_three, fig_height_three), tight_layout=True)

    if fig_2 == "existing_V":
        ylabel_second_fig = r"policy value of existing actions"
        ylabel_third_fig = r"policy value of new actions"
        filename = f'{x}_{relative}_existing_new_value.png'
    elif fig_2 == "per_existing_V":
        ylabel_second_fig = r"policy value per existing action"
        ylabel_third_fig = r"policy value per new action"
        filename = f'{x}_{relative}_existing_new_per_value.png'
    elif fig_2 == "existing_action_ratio":
        ylabel_second_fig = r"proportion of existing actions"
        ylabel_third_fig = r"proportion of new actions"
        filename = f'{x}_{relative}_existing_new_action_ratio.png'
        
    sns.lineplot(
        linewidth=conf.linewidth,
        marker=conf.marker,
        markersize=conf.markersize,
        markers=True,
        x=x,
        y=fig_1,
        hue="method",
        ax=ax[0],
        palette=palette,
        legend=False,
        data=all_data,
    )
    # yaxis
    ax[0].set_ylabel(r"overall policy value $V(\pi_{\theta})$", fontsize=conf.fontsize_ylabel_three, labelpad=20)
    ax[0].tick_params(axis="y", labelsize=conf.labelsize_yticks_three)
    ax[0].yaxis.set_label_coords(-0.15, 0.5)

    # xaxis
   
    ax[0].set_xlabel("", fontsize=conf.fontsize_xlabel_three)
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(xticklabels, fontsize=conf.labelsize_xticks_three)
    ax[0].xaxis.set_label_coords(0.5, -0.1)


    sns.lineplot(
        linewidth=conf.linewidth,
        marker=conf.marker,
        markersize=conf.markersize,
        markers=True,
        x=x,
        y=fig_2,
        hue="method",
        ax=ax[1],
        palette=palette,
        legend=False,
        data=all_data,
    )
    # yaxis
    ax[1].set_ylabel(ylabel_second_fig, fontsize=conf.fontsize_ylabel_three, labelpad=20)
    ax[1].tick_params(axis="y", labelsize=conf.labelsize_yticks_three)
    ax[1].yaxis.set_label_coords(-0.15, 0.5)

    # xaxis
    ax[1].set_xlabel("", fontsize=conf.fontsize_xlabel_three)
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xticklabels, fontsize=conf.labelsize_xticks_three)
    ax[1].xaxis.set_label_coords(0.5, -0.1)


    sns.lineplot(
        linewidth=conf.linewidth,
        marker=conf.marker,
        markersize=conf.markersize,
        markers=True,
        x=x,
        y=fig_3,
        hue="method",
        ax=ax[2],
        palette=palette,
        legend=False,
        data=all_data,
    )
    # yaxis
    ax[2].set_ylabel(ylabel_third_fig, fontsize=conf.fontsize_ylabel_three, labelpad=20)
    ax[2].tick_params(axis="y", labelsize=conf.labelsize_yticks_three)
    ax[2].yaxis.set_label_coords(-0.15, 0.5)

    # xaxis
    ax[2].set_xlabel("", fontsize=conf.fontsize_xlabel_three)
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xticklabels, fontsize=conf.labelsize_xticks_three)
    ax[2].xaxis.set_label_coords(0.5, -0.1)
    ax[2].legend(loc='upper right', fontsize=15, title='')


    # Add common xlabel
    fig.text(0.5, -0.1, xlabels, ha='center', fontsize=conf.fontsize_xlabel)

    if flag_legend == True:
        # Plot legend at the top of the figure
        fig.legend(handles=line_legend_elements, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 1.15), 
                ncol=num_estimators, 
                fontsize=conf.fontsize_legend_three)
    

    # Define the directory and filename
    directory = 'pic'  # Ensure the leading slash is appropriate for your system
    full_path = os.path.join(directory, filename)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save the plot to the specified directory
    fig.savefig(full_path, dpi=300, bbox_inches='tight')




def relative_by_policy(data, x, relative):
    logging_values = data[data['method'] == relative].set_index(['seed', x])
    numeric_cols = ['V', 'existing_V', 'new_V', 'per_existing_V', 'per_new_V', 'estimated_V']

    def relative_row(row):
        key = (row['seed'], row[x])        
        logging_row = logging_values.loc[key, numeric_cols]
        return row[numeric_cols] / logging_row.replace(0, float('inf')) 

    relative_data = data.copy()
    relative_data[numeric_cols] = data.apply(relative_row, axis=1)

    return relative_data

def get_color_with_default(method, palette):

    if method in palette:
        return palette[method]
    else:
        return (random.random(), random.random(), random.random())





# Visualize the loss growth as the number of epochs increases
def show_loss(learner_list, learned_dict, color_dict):
    plt.style.use('ggplot')
    plt.figure(figsize=(conf.width_show_loss_value, conf.height_show_loss_value))

    for learner in learner_list:
        if learner in ["Optimal", "Random", "Logging"]:
            pass
        elif learner in ["RegBased (a)", "RegBased (f)"]:
            plt.plot(range(1, conf.num_epochs_reg + 1), learned_dict[learner].train_loss, color=color_dict[learner], linestyle='-',  label = learner)
        else:
            plt.plot(range(1, conf.num_epochs + 1), learned_dict[learner].train_loss, color=color_dict[learner], linestyle='-',  label = learner)
 
    plt.xticks(ticks=range(1, conf.num_epochs + 1))
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

# Visualize the learned policy value growth as the number of epochs increases
def show_value(learner_list, learned_dict, color_dict):
    plt.style.use('ggplot')
    plt.figure(figsize=(conf.width_show_loss_value, conf.height_show_loss_value))


    for learner in learner_list:
        if learner in ["Optimal", "Random", "Logging"]:
            pass
        else:
            plt.plot(learned_dict[learner].train_value, color=color_dict[learner], linestyle='-',  label = (learner+' tr'))
            plt.plot(learned_dict[learner].test_value, color=color_dict[learner], linestyle='--',  label = (learner+' te'))

    plt.xlabel('number of epochs')
    plt.ylabel(r'learned policy value $V(\pi_{\theta})$')
    plt.title(r'Learned Policy Value $V(\pi_{\theta})$')
    plt.legend()
    plt.show()


def show_bias(learner_list, learned_dict, color_dict):
    plt.style.use('ggplot')
    plt.figure(figsize=(conf.width_show_loss_value, conf.height_show_loss_value))

    for learner in learner_list:
        if learner in ["Optimal", "Random", "RegBased (a)", "RegBased (f)"]:
            pass
        else:
            plt.plot(range(1, conf.num_epochs + 1), learned_dict[learner].train_bias, color=color_dict[learner], linestyle='-',  label = learner)

    plt.xticks(ticks=range(1, conf.num_epochs + 1))
    plt.xlabel('number of epochs')
    plt.ylabel('square bias')
    plt.title('Square Bias')
    plt.legend()
    plt.show()


def show_variance(learner_list, learned_dict, color_dict):
    plt.style.use('ggplot')
    plt.figure(figsize=(conf.width_show_loss_value, conf.height_show_loss_value))

    for learner in learner_list:
        if learner in ["Optimal", "Random", "RegBased (a)", "RegBased (f)"]:
            pass
        else:
            plt.plot(range(1, conf.num_epochs + 1), learned_dict[learner].train_variance, color=color_dict[learner], linestyle='-',  label = learner)

    plt.xticks(ticks=range(1, conf.num_epochs + 1))
    plt.xlabel('number of epochs')
    plt.ylabel('variance')
    plt.title('Variance')
    plt.legend()
    plt.show()

