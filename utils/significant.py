from scipy import stats
import pandas as pd
import os
import numpy as np
import chart_studio.plotly
import plotly.graph_objs as go
from datetime import datetime

from scipy.stats import ttest_rel, ttest_ind


class Significant:

    def __init__(self, method_names, stream_names, metrics=None):
        self.method_names = method_names
        self.stream_names = stream_names
        self.dimension = len(self.method_names)
        self.metrics = metrics

        self.date_time = "{:%Y-%m-%d__%H-%M}".format(datetime.now())
        if not os.path.exists("results/significant_tests/%s/" % (self.date_time)):
            os.makedirs("results/significant_tests/%s/" % (self.date_time))

    def test(self, treshold=0.05):
        data = {}
        ranking = {}
        self.iter = 0

        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                try:
                    data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)
                except:
                    print("None is ",method_name, stream_name)
                    data[(method_name, stream_name)] = None

        if self.metrics is None:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for metric in self.metrics:
            stream_means = []
            stream_stds = []

            f = open('results/significant_tests/%s/table_%s.tex' % (self.date_time, metric), 'wt', encoding='utf-8')

            f2 = open('results/significant_tests/%s/table_%s.md' % (self.date_time, metric), 'wt', encoding='utf-8')


            title1 = "dataset | " + " | ".join([mn for mn in self.method_names]) + "\n"
            title2 = "---|" + "|".join(":---:" for i in self.method_names) + "\n"

            f2.write(title1)
            f2.write(title2)

            for stream_name in self.stream_names:
                means = []
                stds = []
                for method_name in self.method_names:
                    means.append(np.mean(data[(method_name, stream_name)][metric]))
                    stds.append(np.std(data[(method_name, stream_name)][metric]))
                best = np.argmax(means)

                significant = np.zeros(len(self.method_names)).astype(int)
                better_than = []

                # significant = []
                for i, method_1 in enumerate(self.method_names):
                    local_best = []
                    for j, method_2 in enumerate(self.method_names):
                        if i != j:
                            d1 = data[(method_1, stream_name)][metric]
                            d2 = data[(method_2, stream_name)][metric]
                            res = ttest_ind(d1, d2)
                            p = res.pvalue
                            if np.sum(d1 - d2) == 0 and j == best:
                                significant[i] = 1
                            if p <= treshold and means[i] > means[j]:
                                local_best.append(j + 1)
                            if p > treshold and j == best:
                                significant[i] = 1

                    better_than.append(local_best)


                z = stream_name.replace("_", "-").split('/')[-1]

                a = "\\emph{%s} & " % z + " & ".join(["%s%.3f" % ("\\bfseries " if significant[i] == 1 else "", score) for i, score in enumerate(means)]) + " \\\\\n"
                f.write(a)

                b = "& "+ " & ".join([",".join("-" if len(group) == 0 else ["%i" % one if idx_group < 5 else "%s" % one for idx_one, one in enumerate(group)]) for idx_group, group in enumerate(better_than)]) + " \\\\\n"
                f.write(b)

                a1 = "*%s* |" % z +  "|".join([" %s%.3f%s " % ("**" if significant[i] == 1 else "", score, "**" if significant[i] == 1 else "") for i, score in enumerate(means)]) + "\n"
                f2.write(a1)

                b1 = "| |"+ " | ".join([",".join("-" if len(group) == 0 else ["%i" % one if idx_group < 5 else "%s" % one for idx_one, one in enumerate(group)]) for idx_group, group in enumerate(better_than)]) + "\n"
                f2.write(b1)

                stream_means.append(means)

            stream_means = pd.DataFrame(stream_means, index=[stream.split('/')[-1] for stream in self.stream_names], columns=self.method_names)
            stream_means.index.name = "stream"

            stream_means.to_csv("results/significant_tests/%s/%s.csv" % (self.date_time, metric))

            f.close()
            f2.close()
