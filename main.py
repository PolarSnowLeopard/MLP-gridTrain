from data_preprocessing import load_data
from train import train_model
import os

from utils import Utils

if __name__ == "__main__":
    utils = Utils()

    # params = {
    #     'lr': 0.0003,
    #     'weight_decay': 0.002,
    #     'early_stop': 500,
    #     'num_epochs': 50,
    #     'dropout1': 0.2,
    #     'dropout2': 0.5,
    #     'batch_size': 32,
    #     'hide1_num': 256,
    #     'hide2_num': 64,
    #     'id':2
    # }

    # X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    # model, losses, accuracies, validation_losses, validation_accuracy = train_model(X_train, X_val, y_train, y_val, params, utils)

    val_acc_dict = dict()
    val_los_dict = dict()

    for params in utils.generate_param_combinations():
        X_train, X_val, X_test, y_train, y_val, y_test = load_data()
        try:
            model, losses, accuracies, validation_losses, validation_accuracy = train_model(X_train, X_val, y_train, y_val, params, utils)
            val_acc_dict[params['id']] = max(validation_accuracy)
            val_los_dict[params['id']] = min(validation_losses)
        except Exception as e:
            print(e)
            utils.log(f"<hr>\n\n{e}\n\n")

    utils.save_rank(val_acc_dict, val_los_dict)