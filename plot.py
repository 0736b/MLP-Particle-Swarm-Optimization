import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import numpy as np

def plot_pbest_gbest_folds(model, max_t):
    """plot mae on gbest with 5, 10 predict day

    Args:
        model (str): layers and nodes
        max_t (int): max epoch
    """
    idxs = [int(i) for i in range(max_t)]
    plt.figure(figsize = (20,10))
    for i in range(10):
        fold = str(i+1)
        path_train_log_gbest_5 = 'saved/result_train/' + '5' + 'p/' + model + '/' + 'log_gbest' + fold + '.data'
        path_train_log_gbest_10 = 'saved/result_train/' + '10' + 'p/' + model + '/' + 'log_gbest' + fold + '.data'
        with open(path_train_log_gbest_5, 'rb') as f_gbest5:
            gbests_5 = pickle.load(f_gbest5)
        with open(path_train_log_gbest_10, 'rb') as f_gbest10:
            gbests_10 = pickle.load(f_gbest10)
        plt.subplot(2,5,i+1)
        gbest_5_train = pd.DataFrame(gbests_5, index=idxs, columns=['MAE'])
        gbest_5_train.index.name = 'Epoch'
        gbest_10_train = pd.DataFrame(gbests_10, index=idxs, columns=['MAE'])
        gbest_10_train.index.name = 'Epoch'
        merged_train = pd.concat([gbest_5_train, gbest_10_train], axis=0, keys=['5-day', '10-day']).reset_index()
        merged_train = merged_train.rename(columns={'level_0': '', 'level_1': 'Epoch'})
        sns.lineplot(data=merged_train, x='Epoch', y='MAE', hue='', palette=['b','r'])
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.suptitle('Model ' + model + '\nPredict 5, 10 Days G-Best Converge', fontweight='bold', fontsize=24)
        plt.title('Fold ' + fold + ', G-Best@last epoch\n5-day: ' + str(round(gbests_5[max_t - 1], 4)) + '\n10-day: ' + str(round(gbests_10[max_t - 1], 4)), fontweight='bold')
    plt.subplots_adjust(left=0.04,bottom=0.117,right=0.97,top=0.817,wspace=0.29,hspace=0.51)
    plt.show()
    
def plot_train_valid(model, max_t, predictDays):
    """plot training/validation mae

    Args:
        model (str): layers and nodes
        max_t (int): max epoch
        predictDays (int): day predict 5 or 10
    """
    idxs = []
    train_name = ['Training'] * 10
    valid_name = ['Validation'] * 10
    last_gbest_train = []
    last_gbest_valid = []
    for i in range(10):
        fold = str(i+1)
        path_train_log_gbest = 'saved/result_train/' + str(predictDays) + 'p/' + model + '/' + 'log_gbest' + fold + '.data'
        path_valid_best_mae = 'saved/result_valid/' + str(predictDays) + 'p/' + model + '/' + 'best_mae' + '10' + '.data'
        with open(path_train_log_gbest, 'rb') as f_train_gbest:
            train_gbest = pickle.load(f_train_gbest)
        with open(path_valid_best_mae, 'rb') as f_valid_gbest:
            valid_gbest = pickle.load(f_valid_gbest)
        last_gbest_train.append(train_gbest[max_t - 1])
        last_gbest_valid.append(valid_gbest[i])
        idxs.append(fold)
    plot_gbest_train = pd.DataFrame(last_gbest_train, columns=['MAE'])
    gbest_train_name = pd.DataFrame(train_name, columns=['Training/Validation'])
    plot_gbest_valid = pd.DataFrame(last_gbest_valid, columns=['MAE'])
    gbest_valid_name = pd.DataFrame(valid_name, columns=['Training/Validation'])
    df_idx = pd.DataFrame(idxs, columns=['Fold'])
    merge_train = pd.concat([df_idx, plot_gbest_train, gbest_train_name], axis=1)
    merge_valid = pd.concat([df_idx, plot_gbest_valid, gbest_valid_name], axis=1)
    merged_all = pd.concat([merge_train, merge_valid])
    sns.barplot(x='Fold', y='MAE', data=merged_all, hue='Training/Validation')
    plt.title('Model ' + model + '\nTraining/Validation Mean Absolute Error (MAE)\non Particle G-Best' + ' Predict ' + str(predictDays) + '-day', fontweight='bold', fontsize=24)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=18)
    plt.xlabel('Fold', fontsize=18)
    print('Predict:', str(predictDays), 'Training AVG', np.average(last_gbest_train))
    print('Predict:', str(predictDays), 'Validation AVG', np.average(last_gbest_valid))
    plt.show()
          
if __name__ == '__main__':
    # load saved log
    model = '8-7-1'
    max_t = 100
    # plot_pbest_gbest_folds(model, max_t)
    plot_train_valid(model, max_t, 5)
    plot_train_valid(model, max_t, 10)