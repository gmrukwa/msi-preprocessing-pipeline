import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


def histogram(vals, thresholds):
    hist = go.Histogram(x=vals)
    h_down = hist.ybins.start
    h_up = hist.ybins.end
    fig = go.Figure(data=[hist] + [
        go.Scatter(x=[x, x], y=[h_down, h_up], mode='lines',
                   name='Threshold {0}'.format(idx + 1))
        for idx, x in enumerate(thresholds)
    ])
    return fig


def save_decomposition(vals, thresholds, path):
    fig = histogram(vals, thresholds)
    with path.temporary_path() as tmp_path:
        fig.write_html(tmp_path)
