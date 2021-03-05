import joblib
from os.path import isfile, join
from pathlib import Path
import datetime
from src.utils import get_project_models_dir
import time
from src.neighbors._classification import Knn
import numpy as np

from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.model_selection import cross_val_score

def create_dict(X,y,target_names, dataset_name):
    weights_dict = {
        'uniform': [],
        'distance': [],
        'adaptive': [],
        'k_lst': [1, 2, 3, 5, 7, 9, 11, 13, 15],
        # 'k_lst': [1,2],
        'weights_lst': ['uniform', 'distance', 'adaptive'],
        'names_lst': ['Sem peso', 'Com peso', 'Adaptativo'],
        'magic_number': 5,
        'elapsed_time': 0,
        'X': X,
        'y': y,
        'target_names': target_names,
        'dataset_name': dataset_name
    }
    return weights_dict

def print_elapsed_time(weights_dict):
    time_in_seconds = weights_dict['elapsed_time']
    delta = datetime.timedelta(seconds=time_in_seconds)
    print(f'{time_in_seconds} segundos - {delta} hh:mm:ss')

def has_saved_model(dataset_name):
    
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')

    if(isfile(filename)):
        return True
        weights_dict = joblib.load(filename=filename)
        time_in_seconds = weights_dict['elapsed_time']
        delta = datetime.timedelta(seconds=time_in_seconds)
        print(f'{time_in_seconds} segundos - {delta} hh:mm:ss')
    else:
        return False
        print(f'Arquivo não encontrado \'{filename}\'')
        joblib.dump(value=weights_dict, filename=filename)
        print(f'Arquivo criado \'{filename}\'')
    

def get_saved_model(dataset_name):
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')
    return joblib.load(filename=filename)

def save_model(weights_dict,dataset_name):
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')
    joblib.dump(value=weights_dict, filename=filename)
    
def train_model(X,y,target_names,dataset_name):
    if(has_saved_model(dataset_name)):
        return get_saved_model(dataset_name)
    
    weights_dict = create_dict(X,y,target_names, dataset_name)
    magic_number = weights_dict['magic_number']

    exp_time = time.time()

    cv = KFold(n_splits=10,random_state=1,shuffle=True)

    for weights in weights_dict['weights_lst']:
        print ('*'*10, dataset_name, '*'*10, '\n')

        processing_time = []
        acc = []
        acc_std = []

        for k in weights_dict['k_lst']:

            obj = Knn(n_neighbors=k, weights=weights)

            tmp_proc_time = []
            for x in range(magic_number): # 30 times for statistical relevance
                # Train + Test
                start_time = time.time()
                scores = cross_val_score(obj, X,y, scoring='accuracy', cv=cv)
                tmp_proc_time.append( time.time() - start_time )

            # Saving measurements
            processing_time.append(np.mean(tmp_proc_time))
            acc.append(np.mean(scores))
            acc_std.append(np.std(scores))

        weights_dict[weights].extend([processing_time,acc,acc_std])
        print(f'Processing Time: {weights_dict[weights][0]} - M Number: {magic_number}')
        print(f'Acc: {weights_dict[weights][1]}')
        print(f'Acc Std: {weights_dict[weights][2]}')

    weights_dict['elapsed_time'] = time.time() - exp_time
    print(f"\nExperiment elapsed time {weights_dict['elapsed_time']} (s)")
    
    save_model(weights_dict,dataset_name)
    
    return weights_dict