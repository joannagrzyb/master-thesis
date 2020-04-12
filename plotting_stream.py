import strlearn as sl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


n_chunks = 100 # Always declare certain number

# Plotting stream function
cm = LinearSegmentedColormap.from_list("lokomotiv", colors=[(0.3, 0.7, 0.3), (0.7, 0.3, 0.3)])
chunks_plotted = np.linspace(0, n_chunks - 1, 8).astype(int)

def plot_stream(stream, filename="foo", title=""):
    fig, ax = plt.subplots(1, len(chunks_plotted), figsize=(14, 2.5))
    j = 0
    for i in range(n_chunks):
        X, y = stream.get_chunk()
        if i in chunks_plotted:
            ax[j].set_title("Chunk %i" % i)
            ax[j].scatter(X[:, 0], X[:, 1], c=y, cmap=cm, s=10, alpha=0.5)
            ax[j].set_ylim(-4, 4)
            ax[j].set_xlim(-4, 4)
            ax[j].set(aspect="equal")
            j += 1
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig("results/plots/%s.png" % filename)


concept_kwargs = {
    "n_chunks": n_chunks,
    "chunk_size": 250,
    "n_classes": 2,
    # "random_state": 106,
    "n_features": 2,
    "n_drifts": 0,
    "n_informative": 2,
    "n_redundant": 0,
    "n_repeated": 0,
    "weights": [0.1, 0.9],     # stationary imbalanced stream
    # "weights": (1, 5, 0.9),    # dynamically imbalanced stream - do mgr'ki
}
stream = sl.streams.StreamGenerator(**concept_kwargs)

plot_stream(
    # stream, "dynamic-imbalanced-stream", "Data stream with dynamically imbalanced drift"
    stream, "stationary-imbalanced-stream", "Data stream with stationary imbalanced drift"
    )
