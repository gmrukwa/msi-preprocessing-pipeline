import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


def histogram(vals, thresholds):
    nbins = int(np.sqrt(vals.size))
    h_up = np.histogram(vals, bins=nbins)[0].max()
    hist = go.Histogram(x=vals, nbinsx=nbins)
    fig = go.Figure(data=[hist] + [
        go.Scatter(x=[x, x], y=[0, h_up], mode='lines',
                   name='Threshold {0}'.format(idx + 1))
        for idx, x in enumerate(thresholds)
    ])
    return fig


def save_decomposition(vals, thresholds, path):
    fig = histogram(vals, thresholds)
    with path.temporary_path() as tmp_path:
        fig.write_html(tmp_path)
