import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
import warnings
from scipy.stats import ttest_ind
from scipy.stats import rankdata
from tabulate import tabulate

warnings.filterwarnings("ignore")

# Copy these values from experiment, it has to be the same to correctly load files 
clf_names = [
    "HDWE-GNB",
    "HDWE-MLP",
    "HDWE-CART",
    # "HDWE-HDDT",
    "HDWE-KNN",
    # "HDWE-SVC",
]
metric_names = [
    "specificity",
    "recall",
    "precision",
    "f1_score",
    "balanced_accuracy_score",
    "geometric_mean_score_1",
    "geometric_mean_score_2",
]
metric_alias = [
    "Specificity",
    "Recall",
    "Precision",
    "F1",
    "BAC",
    "G-mean1",
    "G-mean2",
]
random_states = [1111, 1234, 1567]
# random_states = [123] # testing
st_stream_weights = [
    [0.01, 0.99], 
    [0.03, 0.97], 
    [0.05, 0.95], 
    [0.1, 0.9], 
    [0.15, 0.85], 
    [0.2, 0.8], 
    [0.25, 0.75]
]
d_stream_weights = [
    (2, 5, 0.99), 
    (2, 5, 0.97), 
    (2, 5, 0.95), 
    (2, 5, 0.9), 
    (2, 5, 0.85), 
    (2, 5, 0.8), 
    (2, 5, 0.75)
]
drifts = ['sudden', 'incremental']
n_chunks = 200-1

sigma = 2 # Parameter to gaussian filter

n_streams = len(drifts)*(len(st_stream_weights)*len(random_states)+len(d_stream_weights)*len(random_states))

plot_data = np.zeros((len(clf_names), n_chunks, len(metric_names)))
mean_scores = np.zeros((len(metric_names), n_streams, len(clf_names)))

# Loading data from files, drawing and saving figures in png and eps format
for drift_id, drift in enumerate(drifts):
    
    # Loop for experiment for stationary imbalanced streams
    for weight_id, weights in enumerate(st_stream_weights):
        for rs_id, random_state in enumerate(random_states):
            s_name = "stat_ir%s_rs%s" % (weights, random_state)
            stream_id = drift_id*len(st_stream_weights)*len(random_states) + weight_id*len(random_states) + rs_id
            for metric_id, metric_name in enumerate(metric_names):
                for clf_id, clf_name in enumerate(clf_names):
                    # Load data from file
                    filename = "results/experiment2/metrics/gen/%s/%s/%s/%s.csv" % (drift, s_name, metric_name, clf_name)
                    plot_data = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    
                    # 1) Save average of scores into mean_scores, 1 stream = 1 avg
                    scores = plot_data.copy()
                    mean_score = np.mean(scores)
                    mean_scores[metric_id, stream_id, clf_id] = mean_score
                    
                #     if sigma > 0:
                #         plot_data = gaussian_filter1d(plot_data, sigma)
                        
                #     plt.plot(range(len(plot_data)), plot_data, label=clf_name)
                    
                # plot_name = "p_gen_%s_s_ir%s_%s_rs%s" % (drift, weights, metric_name, random_state)
                # plotfilename_png = "results/experiment2/plots/gen/%s/%s/%s.png" % (drift, metric_name, plot_name)
                # plotfilename_eps = "results/experiment2/plots/gen/%s/%s/%s.eps" % (drift, metric_name, plot_name)
                
                # if not os.path.exists("results/experiment2/plots/gen/%s/%s/" % (drift, metric_name)):
                #     os.makedirs("results/experiment2/plots/gen/%s/%s/" % (drift, metric_name))
                    
                # plt.legend(framealpha=1)
                # plt.ylabel(metric_alias[metric_id])
                # plt.xlabel("Data chunk")
                # plt.axis([0, n_chunks, 0, 1])
                # plt.gcf().set_size_inches(10, 5) # Get the current figure
                # plt.savefig(plotfilename_png)
                # plt.savefig(plotfilename_eps)
                # # plt.show()
                # plt.clf() # Clear the current figure
                # plt.close() 
                
    # Loop for experiment for dynamically imbalanced streams
    for weight_id, weights in enumerate(d_stream_weights):
        for rs_id, random_state in enumerate(random_states):
            s_name = "d_ir%s_rs%s" % (weights, random_state)
            stream_id = 42 + drift_id*len(st_stream_weights)*len(random_states) + weight_id*len(random_states) + rs_id
            for metric_id, metric_name in enumerate(metric_names):
                for clf_id, clf_name in enumerate(clf_names):
                    # Load data from file
                    filename = "results/experiment2/metrics/gen/%s/%s/%s/%s.csv" % (drift, s_name, metric_name, clf_name)
                    plot_data = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    
                    # 1) Save average of scores into mean_scores, 1 stream = 1 avg
                    scores = plot_data.copy()
                    mean_score = np.mean(scores)
                    mean_scores[metric_id, stream_id, clf_id] = mean_score
                                    
                #     if sigma > 0:
                #         plot_data = gaussian_filter1d(plot_data, sigma)
                        
                #     plt.plot(range(len(plot_data)), plot_data, label=clf_name)
                    
                # plot_name = "p_gen_%s_d_ir%s_%s_rs%s" % (drift, weights, metric_name, random_state)
                # plotfilename_png = "results/experiment2/plots/gen/%s/%s/%s.png" % (drift, metric_name, plot_name)
                # plotfilename_eps = "results/experiment2/plots/gen/%s/%s/%s.eps" % (drift, metric_name, plot_name)
                
                # if not os.path.exists("results/experiment2/plots/gen/%s/%s/" % (drift, metric_name)):
                #     os.makedirs("results/experiment2/plots/gen/%s/%s/" % (drift, metric_name))
                    
                # plt.legend(framealpha=1)
                # plt.ylabel(metric_alias[metric_id])
                # plt.xlabel("Data chunk")
                # plt.axis([0, n_chunks, 0, 1])
                # plt.gcf().set_size_inches(10, 5) # Get the current figure
                # plt.savefig(plotfilename_png)
                # plt.savefig(plotfilename_eps)
                # # plt.show()
                # plt.clf() # Clear the current figure
                # plt.close()
                
# print("\nMean scores:\n", mean_scores)
alfa = 0.05
t_statistics = np.zeros((len(metric_names), len(clf_names), len(clf_names)))
p_values = np.zeros((len(metric_names), len(clf_names), len(clf_names)))
# 2) obliczenie rang, im wyższa ranga, tym lepsza metoda
for metric_id, metric_name in enumerate(metric_names):
    ranks = []
    for ms in mean_scores[metric_id]:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    # print("\nRanks for", metric_name, ": ", ranks, "\n")
    mean_ranks = np.mean(ranks, axis=0)
    # print(metric_name)
    # print("Mean ranks:\n", mean_ranks)
    # print()
    
    # statistic T-student test
    # t_statistic = np.zeros((len(clf_names), len(clf_names)))
    # p_value = np.zeros((len(clf_names), len(clf_names)))
    
    for i in range(len(clf_names)):
        for j in range(len(clf_names)):
            t_statistics[metric_id, i, j], p_values[metric_id, i, j] = ttest_ind(ranks.T[i], ranks.T[j])
    # print("\nt-statistic:\n", t_statistics, "\n\np-value:\n", p_values)   

    headers = clf_names
    names_column = np.expand_dims(np.array(clf_names), axis=1)
    t_statistic_table = np.concatenate((names_column, t_statistics[metric_id]), axis=1)
    t_statistic_table = tabulate(t_statistics[metric_id], headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_values[metric_id]), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("\nt-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
    
    advantage = np.zeros((len(clf_names), len(clf_names)))
    advantage[t_statistics[metric_id] > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)
    
    significance = np.zeros((len(clf_names), len(clf_names)))
    significance[p_values[metric_id] <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    # print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)
    
# Plotting test
treshold = 0.10
for metric_id, metric_name in enumerate(metric_names):
    ranking = {}
    for clf_name in clf_names:
        ranking[clf_name] = {"win": 0, "lose": 0, "tie": 0}
        
    for i, method_1 in enumerate(clf_names):
        for j, method_2 in enumerate(clf_names):
            if p_values[metric_id, i, j] < treshold:
                if t_statistics[metric_id, i, j] > 0:
                    ranking[method_1]["win"] += 1
                else:
                    ranking[method_1]["lose"] += 1
            else:
                ranking[method_1]["tie"] += 1
            
    rank_win = []
    rank_tie = []
    rank_lose = []
    rank_none = []

    for clf_name in clf_names:
        rank_win.append(ranking[clf_name]['win'])
        rank_tie.append(ranking[clf_name]['tie'])
        rank_lose.append(ranking[clf_name]['lose'])
        try:
            rank_none.append(ranking[clf_name]['none'])
        except Exception:
            pass

    rank_win.reverse()
    rank_tie.reverse()
    rank_lose.reverse()
    rank_none.reverse()

    rank_win = np.array(rank_win)
    rank_tie = np.array(rank_tie)
    rank_lose = np.array(rank_lose)
    rank_none = np.array(rank_none)
    ma = clf_names.copy()
    ma.reverse()
    plt.rc('ytick', labelsize=30)

    plt.barh(ma, rank_win, color="green", height=0.9)
    plt.barh(ma, rank_tie, left=rank_win, color="gold", height=0.9)
    plt.barh(ma, rank_lose, left=rank_win+rank_tie, color="crimson", height=0.9)
    try:
        plt.barh(ma, rank_none, left=rank_win+rank_tie+rank_lose, color="black", height=0.9)
    except Exception:
        pass
    # plt.xlim(0, len(self.stream_names))
    plt.axvline(4, 0, 1, linestyle="--", linewidth=3, color="black")
    plt.title(metric_name, fontsize=40)
    plt.show()
    # if not os.path.exists("results/ranking_plots/%s/" % (experiment_name)):
    #     os.makedirs("results/ranking_plots/%s/" % (experiment_name))
    plt.gcf().set_size_inches(5, 5)
    # plt.savefig(fname="results/ranking_plots/%s/%s_%s_hbar" % (experiment_name, self.streams_alias, metric), bbox_inches='tight')
    plt.clf()
    
print(ranking)