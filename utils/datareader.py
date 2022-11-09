import pandas as pd
import numpy as np
import random

def get_dataset(path: str, predictDays: int):
    """read data from dataset file

    Args:
        path (str): path to dataset

    Returns:
        list: normalized dataset
    """
    dataset = []
    file_data = pd.read_excel('dataset/AirQualityUCI.xlsx')
    file_data.drop('Date', inplace=True, axis=1)
    file_data.drop('Time', inplace=True, axis=1)
    min_v = 0
    max_v = 0
    for column in file_data.columns:
        if column == 'C6H6(GT)':
            min_v = file_data[column].min()
            max_v = file_data[column].max()
        file_data[column] = (file_data[column] - file_data[column].min()) / (file_data[column].max() - file_data[column].min())
    data_datetime = pd.DataFrame(file_data, columns=['Date', 'Time'])
    data_input = pd.DataFrame(file_data, columns=['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'])
    data_output = pd.DataFrame(file_data, columns=['C6H6(GT)'])
    for i in range(len(data_datetime)):
        data = {}
        inputs = []
        output = []
        output_idx = i + (24 * predictDays)
        if not (output_idx >= len(data_datetime)):
            inputs.append(data_input.iloc[i]['PT08.S1(CO)'])
            inputs.append(data_input.iloc[i]['PT08.S2(NMHC)'])
            inputs.append(data_input.iloc[i]['PT08.S3(NOx)'])
            inputs.append(data_input.iloc[i]['PT08.S4(NO2)'])
            inputs.append(data_input.iloc[i]['PT08.S5(O3)'])
            inputs.append(data_input.iloc[i]['T'])
            inputs.append(data_input.iloc[i]['RH'])
            inputs.append(data_input.iloc[i]['AH'])
            output.append(data_output.iloc[output_idx]['C6H6(GT)'])
            data['INPUT'] = inputs
            data['OUTPUT'] = output
            dataset.append(data)
        else:
            break
    return dataset, min_v, max_v

def shuffle_data(dataset):
    """shuffle dataset

    Args:
        dataset (list): dataset

    Returns:
        list: shuffled dataset
    """
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    return shuffled

def split_groups(l, n):
    """split dataset to group with n data per group

    Args:
        l (list): dataset
        n (int): size per group

    Yields:
        list: dataset that splitted to group
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cross_valid(dataset):
    """create train, validation to be feed as input of model for doing cross-validation 10 folds

    Args:
        dataset (list): dataset that want to use

    Returns:
        list: train_dataset, validation_dataset
    """
    shuffled_dataset = shuffle_data(dataset)
    dataset_size = len(shuffled_dataset)
    group_size = int(dataset_size / 10)
    train_dataset_folds = []
    test_dataset_folds = []
    splitted = list(split_groups(shuffled_dataset, group_size))
    if len(splitted) > 10:
        del splitted[10]
    for i in range(len(splitted)):
        sum_fold_train = []
        for j in range(len(splitted)):
            if j != i:
                sum_fold_train += splitted[j].copy()
        fold_test = splitted[i].copy()
        train_dataset_folds.append(sum_fold_train)
        test_dataset_folds.append(fold_test)
    return train_dataset_folds, test_dataset_folds

def normalize(data):
    """min-max normalization

    Args:
        data (list): dataset

    Returns:
        list: normalized dataset
    """
    norm = [(float(i) - min(data)) / (max(data) - min(data)) for i in data]
    return norm

def denormalize(output, min_o, max_o):
    """de-normalization

    Args:
        output (float): output value
        min_o (float): min output from dataset
        max_o (float): max output from dataset

    Returns:
        de-norm output: de-norm output
    """
    # Y_denorm = Y_norm * (max_Y - min_Y) + min_Y
    return (output * (max_o - min_o) + min_o)

def vectorize(dataset):
    """make input output vectorized for mlp

    Args:
        dataset (list): dataset

    Returns:
        dataset_vectorized: dataset_vectorized
    """
    all_input = []
    all_output = []
    for d in dataset:
        all_input.append(d['INPUT'])
        all_output.append(d['OUTPUT'])
    all_input = np.array(all_input)
    all_output = np.array(all_output)
    com_dataset = [all_input, all_output]
    return com_dataset