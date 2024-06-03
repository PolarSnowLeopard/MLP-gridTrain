from config import OUTPUT_DIR, PARAMS

from time import localtime, strftime
from os import mkdir
from os.path import exists
import itertools

import matplotlib.pyplot as plt

class Utils:
    def __init__(self) -> None:
        self.label = strftime("%Y%m%d_%H%M%S", localtime())
        self.output_dir = OUTPUT_DIR + "/" + self.label
        self.log_path = self.output_dir + "/log.md"

        if not exists(OUTPUT_DIR):
            mkdir(OUTPUT_DIR)
            
        mkdir(self.output_dir)
        with open(self.log_path, "w", encoding="utf-8") as f:
            pass

    def save_loss_acc_plot(self, loss_train, acc_train, loss_val, acc_val, id):
        # 画 Loss 图
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_train, label='Train Loss', color='blue')  # 蓝色
        plt.plot(loss_val, label='Validation Loss', color='orange')  # 橙色
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 画 Accuracy 图
        plt.subplot(1, 2, 2)
        plt.plot(acc_train, label='Train Accuracy', color='blue')  # 蓝色
        plt.plot(acc_val, label='Validation Accuracy', color='orange')  # 橙色
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(f"{self.output_dir}/{id}.png")
        plt.close('all')

        self.log("![result](%s.png)\n\n" % (id))
    
    def log(self, string):
        print(string)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(string)

    def generate_param_combinations(self):
        keys = ['lr', 'weight_decay', 'early_stop', 'num_epochs', 'dropout1', 'dropout2', 'batch_size', 'hide1_num', 'hide2_num']
        param_id = 1  # 初始化参数ID
        for combination in itertools.product(*PARAMS):
            param_dict = dict(zip(keys, combination))
            param_dict['id'] = param_id  # 添加参数ID
            yield param_dict
            param_id += 1  # 递增参数ID
