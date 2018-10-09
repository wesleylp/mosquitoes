import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(
    'paper',
    'white',
    'colorblind',
    font_scale=2.2,
    rc={
        'lines.linewidth': 2,
        'figure.figsize': (10.0, 6.0),
        'image.interpolation': 'nearest',
        'image.cmap': 'gray'
    })


def bar_plot(data):
    plt.figure()
    y_pos = np.arange(len(data))
    plt.bar(y_pos, data, align='center', alpha=0.5)

    return plt
