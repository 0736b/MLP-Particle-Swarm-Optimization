from utils.datareader import get_dataset, cross_valid
from pso.pso import PSO
import pickle

def main(predictDays: int):
    p5_paths = ['dataset/saved/p5/train_folds_p5.data','dataset/saved/p5/valid_folds_p5.data', 'dataset/saved/p5/min_o_p5.data', 'dataset/saved/p5/max_o_p5.data']
    p10_paths = ['dataset/saved/p10/train_folds_p10.data', 'dataset/saved/p10/valid_folds_p10.data', 'dataset/saved/p10/min_o_p10.data', 'dataset/saved/p10/max_o_p10.data']
    use_path = None
    if predictDays == 5:
        use_path = p5_paths
    elif predictDays == 10:
        use_path = p10_paths
    path_train = use_path[0]
    path_valid = use_path[1]
    path_minO = use_path[2]
    path_maxO = use_path[3]
    with open(path_train, 'rb') as f_train:
        train_folds = pickle.load(f_train)
    with open(path_valid, 'rb') as f_valid:
        valid_folds = pickle.load(f_valid)
    with open(path_minO, 'rb') as f_min:
        min_o = pickle.load(f_min)
    with open(path_maxO, 'rb') as f_max:
        max_o = pickle.load(f_max)
    f_train.close()
    f_valid.close()
    f_min.close()
    f_max.close()
    folds = 10
    max_t = 10
    particle_size = 10
    pso = PSO(particle_size, max_t, [8,3,1], train_folds[0])
    best_particle = pso.run()
    mae = best_particle.run_show(valid_folds[0], min_o, max_o)
    print('Best Particle MAE on Validation set:', mae)
    
def save_dataset():
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
    main(5)