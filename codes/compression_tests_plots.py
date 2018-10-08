import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from compression_tests import bar_plot, compute_filesize

sns.set(
    'paper',
    'white',
    font_scale=2.0,
    rc={
        'lines.linewidth': 2,
        'figure.figsize': (16.0, 12.0),
        'image.interpolation': 'nearest',
        'image.cmap': 'gray'
    })
sns.set_palette('Paired', 12)

data_path = '../data/DJI4_cam/2018-09-05/seq001'
dict_mean = {'imageio{:02d}'.format(d): [] for d in range(0, 11)}
dict_std = {'imageio{:02d}'.format(d): [] for d in range(0, 11)}
dict_filesize = {'imageio{:02d}'.format(d): [] for d in range(0, 11)}

for (dirpath, dirnames, filenames) in os.walk(data_path):

    if len(filenames) == 0:
        continue

    for filename in filenames:

        # if the file is pickle file, then process!
        if filename.lower().endswith(('.data')):

            input_path = os.path.join(dirpath, filename)
            print(input_path)

            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            f.close()

            psnr = data['psnr']

            # plot difference frame by frame
            plt.figure()
            plot = bar_plot(psnr)
            plot.xlabel('frame number')
            plot.ylabel('PSNR [dB]')
            plot.ylim([0, 45])
            plot.axhline(y=np.mean(psnr), c='r', ls='--')
            plot.savefig(os.path.join(dirpath, filename.split('.')[0] + '.pdf'))
            plt.close('all')

            if 'imageio' in filename:
                regex_rule = r'imageio\d{2}'
                compress = re.findall(regex_rule, filename)[0]

                dict_mean[compress].append(np.mean(psnr))
                dict_std[compress].append(np.std(psnr))
                dict_filesize[compress].append(
                    compute_filesize(os.path.join(dirpath,
                                                  filename.split('.')[0] + '.MOV')))

# Transform in a dataframe
df_mean = pd.DataFrame(dict_mean)
df_std = pd.DataFrame(dict_std)

fig, ax = plt.subplots()
# df_mean.plot.bar(yerr=df_std, ax=ax, rot=0)
df_mean.plot.bar(ax=ax, rot=0)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('PSNR [dB]')
ax.set_xticklabels(('Video 1\n 140.6MB', 'Video 2\n 1.8GB', 'Video 3\n 1.8GB', 'Video 4\n 1.8GB'))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
plt.ylim([15, 40])

text_format = '{:.2f}MB'

count = 0
count_comp = 0
count_vid = 0

for i in ax.patches:

    if count_vid == len(dict_filesize['imageio00']):
        count_vid = 0
        count_comp += 1

    video_size = dict_filesize['imageio{:02d}'.format(count_comp)][count_vid]
    text = text_format.format(video_size)
    count_vid += 1

    ax.text(
        i.get_x() + i.get_width() * 0.43,
        1.01 * i.get_height(),
        text,
        fontsize=11,
        color='dimgrey',
        rotation=-45,
        rotation_mode='anchor',
        ha='right',
        va='center')

    count += 1

plot.savefig(os.path.join(data_path, 'comparisson.pdf'))
