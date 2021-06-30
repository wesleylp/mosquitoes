import json
import matplotlib.pyplot as plt

experiment_folder = './output/v1_new_val/faster_rcnn_R_50_FPN_3x/mbg_train5_tire'
# model_iter4000_lr0005_wf1_date2020_03_20__05_16_45'


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

iter_train = [x['iteration'] for x in experiment_metrics if 'total_loss' in x]
loss_train = [x['total_loss'] for x in experiment_metrics if 'total_loss' in x]

iter_val = [x['iteration'] for x in experiment_metrics if 'total_val_loss' in x]
loss_val = [x['total_val_loss'] for x in experiment_metrics if 'total_val_loss' in x]

plt.plot(iter_train, loss_train)
plt.plot(iter_val, loss_val)

plt.legend(['train', 'val'], loc='upper right')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.savefig(experiment_folder + "/train_val_loss.png")
plt.show()
