import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

dir = "/home/rliu02/learning/school_work/CSE791_CV/log/MsPacmanNoFrameskip-v4/dqn/0"
tags = [
    "test/returns_stat/mean", "test/lens_stat/mean", "test/n_collected_steps", 
        "train/lens_stat/mean", "train/returns_stat/max", "train/n_collected_steps",
        "info/test_step", "info/best_reward", "info/timing/train_time"
        ]
# tags = ["Train/epoch_reward_max"]

def generate_plot(data_dict, name):
    fig_name = "_".join(name.split("/"))
    plt.clf()
    plt.figure()
    for k in data_dict.keys():
        legend_name = k
        # legend_name = legend_name.split("ab_exact_train_3_4_5_test_6_7_gru_vpg_normalize_w_")[1]
        legend_name = "cnn" if legend_name == "240404-134128" else "vit"
        if legend_name == "":
            legend_name = "default"
        plt.plot(range(len(data_dict[k])), data_dict[k], label=legend_name)
    plt.legend()
    plt.title(fig_name)
    plt.xlabel("iteration")
    # plt.ylabel(fig_name)
    plt.savefig("{}.pdf".format(fig_name))


def plot_data(tag_dict):
    for t in tags:
        generate_plot(tag_dict[t], t)

if __name__ == "__main__":
    f_names = os.listdir(dir)
    steps = None
    tag_dict = {}
    for t in tags:
        data_dict = {}
        for name in f_names:
            file = os.path.join(dir, name)
            event_acc = EventAccumulator(file)
            event_acc.Reload()
            scalars = event_acc.Scalars(t)
            # if steps is None:
            #     steps = [s.step for s in scalars]
            data_dict[name] = [s.value for s in scalars]
        tag_dict[t] = data_dict
    plot_data(tag_dict)
            
                