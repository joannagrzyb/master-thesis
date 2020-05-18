import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# TODO: uzupełnij poprawnie

# Copy these values from experiment, it has to be the same to correctly load files 
clf_names = [
    "HDWE-GNB",
    "HDWE-MLP",
    "HDWE-CART",
    "HDWE-HDDT",
    "HDWE-KNN",
    "HDWE-SVC",
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

plot_data = np.zeros((len(clf_names), n_chunks, len(metric_names)))

# Loading data from files, drawing and saving figures in png and eps format
for drift in drifts:
    
    # Loop for experiment for stationary imbalanced streams
    for weights in tqdm(st_stream_weights, "Plloting stationary imb. stream"):
        for random_state in random_states:
            s_name = "stat_ir%s_rs%s" % (weights, random_state)
            for metric_id, metric_name in enumerate(metric_names):
                for clf_name in clf_names:
                    # Load data from file
                    filename = "results/experiment3/metrics/gen/%s/%s/%s/%s.csv" % (drift, s_name, metric_name, clf_name)
                    plot_data = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    
                    if sigma > 0:
                        plot_data = gaussian_filter1d(plot_data, sigma)
                        
                    plt.plot(range(len(plot_data)), plot_data, label=clf_name)
                    
                plot_name = "p_gen_%s_s_ir%s_%s_rs%s" % (drift, weights, metric_name, random_state)
                plotfilename_png = "results/experiment3/plots/gen/%s/%s/%s.png" % (drift, metric_name, plot_name)
                plotfilename_eps = "results/experiment3/plots/gen/%s/%s/%s.eps" % (drift, metric_name, plot_name)
                
                if not os.path.exists("results/experiment3/plots/gen/%s/%s/" % (drift, metric_name)):
                    os.makedirs("results/experiment3/plots/gen/%s/%s/" % (drift, metric_name))
                    
                plt.legend(framealpha=1)
                plt.ylabel(metric_alias[metric_id])
                plt.xlabel("Data chunk")
                plt.axis([0, n_chunks, 0, 1])
                plt.gcf().set_size_inches(10, 5) # Get the current figure
                plt.savefig(plotfilename_png)
                plt.savefig(plotfilename_eps)
                # plt.show()
                plt.clf() # Clear the current figure
                plt.close() 
                
    # Loop for experiment for dynamically imbalanced streams
    for weights in tqdm(d_stream_weights, "Plotting dynamically imb. stream"):
        for random_state in random_states:
            s_name = "d_ir%s_rs%s" % (weights, random_state)
            for metric_id, metric_name in enumerate(metric_names):
                for clf_name in clf_names:
                    # Load data from file
                    filename = "results/experiment3/metrics/gen/%s/%s/%s/%s.csv" % (drift, s_name, metric_name, clf_name)
                    plot_data = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    
                    if sigma > 0:
                        plot_data = gaussian_filter1d(plot_data, sigma)
                        
                    plt.plot(range(len(plot_data)), plot_data, label=clf_name)
                    
                plot_name = "p_gen_%s_d_ir%s_%s_rs%s" % (drift, weights, metric_name, random_state)
                plotfilename_png = "results/experiment3/plots/gen/%s/%s/%s.png" % (drift, metric_name, plot_name)
                plotfilename_eps = "results/experiment3/plots/gen/%s/%s/%s.eps" % (drift, metric_name, plot_name)
                
                if not os.path.exists("results/experiment3/plots/gen/%s/%s/" % (drift, metric_name)):
                    os.makedirs("results/experiment3/plots/gen/%s/%s/" % (drift, metric_name))
                    
                plt.legend(framealpha=1)
                plt.ylabel(metric_alias[metric_id])
                plt.xlabel("Data chunk")
                plt.axis([0, n_chunks, 0, 1])
                plt.gcf().set_size_inches(10, 5) # Get the current figure
                plt.savefig(plotfilename_png)
                plt.savefig(plotfilename_eps)
                # plt.show()
                plt.clf() # Clear the current figure
                plt.close()
          
# TODO: Uśrednianie jest potrzebne do testów statystycznych, zrób to później, jak prof. odpisze
# TODO: testy statystyczne - testy parowe 