from utils.datareader import get_dataset, cross_valid, vectorize
from pso.pso import PSO
from vec_mlp.vec_mlp import VEC_MLP
import pickle
import numpy as np

def main(predictDays: int):
    """running pso

    Args:
        predictDays (int): day to predict
    """
    np.random.seed(736)
    p5_paths = ['dataset/saved/p5/train_folds_p5.data','dataset/saved/p5/valid_folds_p5.data', 'dataset/saved/p5/min_o_p5.data', 'dataset/saved/p5/max_o_p5.data']
    p10_paths = ['dataset/saved/p10/train_folds_p10.data', 'dataset/saved/p10/valid_folds_p10.data', 'dataset/saved/p10/min_o_p10.data', 'dataset/saved/p10/max_o_p10.data']
    use_path = None
    if predictDays == 5:
        use_path = p5_paths
    elif predictDays == 10:
        use_path = p10_paths
    path_train = use_path[0]
    path_valid = use_path[1]
    with open(path_train, 'rb') as f_train:
        train_folds = pickle.load(f_train)
    with open(path_valid, 'rb') as f_valid:
        valid_folds = pickle.load(f_valid)
    f_train.close()
    f_valid.close()
    folds = 10
    max_t = 100
    particle_size = 30
    path_model_train = 'saved/result_train/' + str(predictDays) + 'p/8-4-4-1/'
    path_model_valid = 'saved/result_valid/' + str(predictDays) + 'p/8-4-4-1/'
    # crossvalidation
    valid_maes = []
    for i in range(folds):
        on_fold = 'Fold: ' + str(i+1)
        train_datas = train_folds[i]
        valid_datas = valid_folds[i]
        train = vectorize(train_datas)
        valid = vectorize(valid_datas)
        pso = PSO(particle_size, max_t, [8,4,4,1], train)
        best_particle, lowest_train_mae, log_result_gbest, log_result_pbest_avg = pso.run()
        with open(path_model_train + 'log_gbest' + str(i+1) + '.data', 'wb') as log_gbest:
            pickle.dump(log_result_gbest, log_gbest)
        with open(path_model_train + 'log_avg_best' + str(i+1) + '.data', 'wb') as log_avg_best:
            pickle.dump(log_result_pbest_avg, log_avg_best)
        mlp = VEC_MLP([8,4,4,1])
        mlp.set_weights(best_particle)
        print(on_fold, 'Best Particle MAE on Training set:', lowest_train_mae)
        print(on_fold, 'Validation...')
        mae = mlp.run(valid[0], valid[1])
        valid_maes.append(mae)
        print(on_fold, 'Best Particle MAE on Validation set:', mae)
    with open(path_model_valid + 'best_mae' + '10' + '.data', 'wb') as valid_mae:
        pickle.dump(valid_maes, valid_mae)
    
def save_dataset():
    """pre processing dataset and save for fast load
    """
    dataset_p5, min_o5, max_o5 = get_dataset('dataset/AirQualityUCI.xlsx', 5)
    dataset_p10, min_o10, max_o10 = get_dataset('dataset/AirQualityUCI.xlsx', 10)
    train_folds_p5, test_folds_p5 = cross_valid(dataset_p5)
    train_folds_p10, test_folds_p10 = cross_valid(dataset_p10)
    path_p5_train = 'dataset/saved/p5/train_folds_p5.data'
    path_p5_valid = 'dataset/saved/p5/valid_folds_p5.data'
    path_p5_min_o = 'dataset/saved/p5/min_o_p5.data'
    path_p5_max_o = 'dataset/saved/p5/max_o_p5.data'
    path_p10_train = 'dataset/saved/p10/train_folds_p10.data'
    path_p10_valid = 'dataset/saved/p10/valid_folds_p10.data'
    path_p10_min_o = 'dataset/saved/p10/min_o_p10.data'
    path_p10_max_o = 'dataset/saved/p10/max_o_p10.data'
    with open(path_p5_train, 'wb') as f_p5t:
        pickle.dump(train_folds_p5, f_p5t)
    with open(path_p5_valid, 'wb') as f_p5v:
        pickle.dump(test_folds_p5, f_p5v)
    with open(path_p10_train, 'wb') as f_p10t:
        pickle.dump(train_folds_p10, f_p10t)
    with open(path_p10_valid, 'wb') as f_p10v:
        pickle.dump(test_folds_p10, f_p10v)
    with open(path_p5_min_o, 'wb') as f_om5:
        pickle.dump(min_o5, f_om5)
    with open(path_p5_max_o, 'wb') as f_oma5:
        pickle.dump(max_o5, f_oma5)
    with open(path_p10_min_o, 'wb') as f_om10:
        pickle.dump(min_o10, f_om10)
    with open(path_p10_max_o, 'wb') as f_oma10:
        pickle.dump(max_o10, f_oma10)
    f_p5t.close()
    f_p5v.close()
    f_p10t.close()
    f_p10v.close()
    f_om5.close()
    f_oma5.close()
    f_om10.close()
    f_oma10.close()
    print('saved')
    
if __name__ == '__main__':
    # save_dataset()
    main(10)