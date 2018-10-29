import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import cv2
from utils.img_utils import add_bboxes_on_image

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


# Helper function to show a batch
def show_bboxes_batch(sample_batched, bboxes_batched, normalized=True, nrow=4, padding=2):
    """Show image with bboxes for a batch of samples."""

    frames_batch = sample_batched
    bboxes_batch = bboxes_batched
    batch_size = len(frames_batch)
    # im_size = frames_batch.size(2)

    # grid = utils.make_grid(frames_batch, nrow, padding)
    # plt.imshow(grid.numpy().transpose((1, 2, 0)))

    ncol = np.ceil(batch_size / nrow).astype('int')

    for idx, frame in enumerate(bboxes_batch):
        ax = plt.subplot(ncol, nrow, idx + 1)

        img = frames_batch[idx, ]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = add_bboxes_on_image(img, bboxes_batch[frame])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        # fig.add_subplot(ax)

    plt.suptitle('Batch from dataloader')

    return

    # for j in range(ncol):
    #     for i in range(nrow):
    #         if i + j * nrow >= len(bboxes_batch):
    #             continue

    #         plt.scatter(
    #             bboxes_batch[i + j * nrow, :, 0].numpy() * im_size + i * im_size + padding * i,
    #             bboxes_batch[i + j * nrow, :, 1].numpy() * im_size + j * im_size + padding * j,
    #             s=10,
    #             marker='.',
    #             c='r')
