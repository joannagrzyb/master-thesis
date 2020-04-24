import pandas as pd
import numpy as np
import plotly.graph_objs as go
import chart_studio.plotly
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


class Ploting:
    def __init__(self, directory="./"):
        self.directory = directory

    def plot(self, method_names, stream_name, experiment_name, auto_open=True, metrics=None):
        self.method_names = method_names
        self.stream_name = stream_name
        self.metrics = metrics

        data_trace = None
        if not os.path.exists("results/plots/%s/" % self.stream_name):
            os.makedirs("results/plots/%s/" % self.stream_name)

        for method_name in self.method_names:
            try:
                data = pd.read_csv("results/raw/%s/%s.csv" % (self.stream_name, method_name), header=0, index_col=0)

                if data_trace is None:
                    if self.metrics is not None:
                        data_trace = [[] for column_name in self.metrics]
                        column_names = self.metrics
                    else:
                        data_trace = [[] for column_name in data.columns]
                        column_names = data.columns

                for i, column_name in enumerate(column_names):
                    data_trace[i] += [go.Scatter(
                        x=data.index.values, y=data[column_name].values, name='%s' % method_name,
                        mode='lines',
                        line=dict(width=1),
                        showlegend=True
                    )]
            except(FileNotFoundError):
                continue

        for i, column_name in enumerate(column_names):
            layout = go.Layout(title='%s, data stream - %s' % (column_name, self.stream_name), plot_bgcolor='rgb(230, 230, 230)')
            fig = go.Figure(data=data_trace[i], layout=layout)
            plotly.offline.plot(fig, filename="results/plots/%s/%s.html" % (self.stream_name, column_name), auto_open=auto_open)

    def plot_streams(self, streams, method_name, auto_open=True):

        data_trace = None
        if not os.path.exists("results/plots/%s/" % method_name):
            os.makedirs("results/plots/%s/" % method_name)

        for stream in streams:
            try:
                stream = self.directory + stream
                data = pd.read_csv("results/raw/%s/%s.csv" % (stream, method_name), header=0, index_col=0)

                if data_trace is None:
                    data_trace = [[] for column_name in data.columns]

                for i, column_name in enumerate(data.columns):
                    data_trace[i] += [go.Scatter(
                        x=data.index.values, y=data[column_name].values, name='%s' % stream,
                        mode='lines',
                        line=dict(width=1),
                        showlegend=True
                    )]
            except(FileNotFoundError):
                continue

        for i, column_name in enumerate(data.columns):
            layout = go.Layout(title='%s, method - %s' % (column_name, method_name), plot_bgcolor='rgb(230, 230, 230)')
            fig = go.Figure(data=data_trace[i], layout=layout)
            plotly.offline.plot(fig, filename="results/plots/%s/%s.html" % (method_name, column_name), auto_open=auto_open)

    def plot_streams_matplotlib(self, methods, streams, metrics, experiment_name, gauss=0, methods_alias=None, metrics_alias=None):

        if methods_alias is None:
            methods_alias = methods
        if metrics_alias is None:
            metrics_alias = metrics

        data = {}

        for stream_name in streams:
            for clf_name in methods:
                for metric in metrics:
                    # filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                    # data[stream_name, clf_name, metric] = np.genfromtxt(filename, delimiter=',', dtype=np.int16)
                    try:
                        filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                        data[stream_name, clf_name, metric] = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    except Exception:
                        data[stream_name, clf_name, metric] = None
                        print("Error in loading data", stream_name, clf_name, metric)

        for stream_name in tqdm(streams, "Plotting"):
            for metric, metric_a in zip(metrics, metrics_alias):
                for clf_name, method_a in zip(methods, methods_alias):
                    if data[stream_name, clf_name, metric] is None:
                        continue

                    plot_data = data[stream_name, clf_name, metric]

                    if gauss > 0:
                        plot_data = gaussian_filter1d(plot_data, 5)

                    plt.plot(range(len(plot_data)), plot_data, label=method_a)

                filename = "results/plots/%s/%s/%s.png" % (experiment_name, metric, stream_name)

                stream_name_ = "/".join(stream_name.split("/")[0:-1])
                # print(stream_name)

                if not os.path.exists("results/plots/%s/%s/%s/" % (experiment_name, metric, stream_name_)):
                    os.makedirs("results/plots/%s/%s/%s/" % (experiment_name, metric, stream_name_))

                plt.legend()
                plt.ylabel(metric_a)
                plt.xlabel("Data chunk")
                plt.gcf().set_size_inches(10, 5)
                plt.savefig(filename)
                plt.clf()
                plt.close()