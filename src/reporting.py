import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from src.utils import get_project_results_dir
from src.neighbors._classification import Knn
from sklearn.model_selection import StratifiedKFold as KFold

import src.plot_utils as pu
import numpy as np

def plot_hq_summary(weights_dict):
    k_lst = weights_dict['k_lst']
    weights_lst = weights_dict['weights_lst']
    dataset_name = weights_dict['dataset_name']
    names_lst = weights_dict['names_lst']

    fmt = ['ro--','g^--','bs--']

    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 6)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')
 
    ax = fig.add_subplot(1,2,1)

    ax.set_xlabel('Parâmetro K')
    ax.set_ylabel('Tempo de Processamento (s)')

    ax.set_axisbelow(True)

    for i in range(3):
        curr_weight = weights_lst[i]
        curr_name = names_lst[i]
        ax.plot(
            k_lst, 
            weights_dict[curr_weight][0], 
            fmt[i], 
            markersize=1.5, 
            linewidth=0.5,
            label=curr_name)
        ax.set_xticks(k_lst)

    plt.legend()
    plt.tight_layout()

    ax = fig.add_subplot(1,2,2)

    ax.set_xlabel('Parâmetro K')
    ax.set_ylabel('Acurácia')

    ax.set_axisbelow(True)
    
    for i in range(3):
        curr_weight = weights_lst[i]
        curr_name = names_lst[i]
        ax.plot(
            k_lst, 
            weights_dict[curr_weight][1], 
            fmt[i], 
            markersize=1.5, 
            linewidth=0.5,
            label=curr_name)
        ax.set_xticks(k_lst)

    plt.legend()
    plt.tight_layout()

    filename = get_project_results_dir().joinpath(dataset_name + '_summary.eps')

    # pu.save_fig(fig, str(filename))
    plt.show()

def plot_hq_versions(weights_dict):
    k_lst = weights_dict['k_lst']
    weights_lst = weights_dict['weights_lst']
    dataset_name = weights_dict['dataset_name']
    names_lst = weights_dict['names_lst']

    fmt = ['ro--','g^--','bs--']

    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 6)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    for i in range(3):
        ax = fig.add_subplot(1,3,i+1)
        ax.set_xlabel('Parâmetro K')
        ax.set_ylabel('Acurácia')
        ax.set_axisbelow(True)

        ax.set_ylim(0.7,1.0)
        curr_weight = weights_lst[i]
        curr_name = names_lst[i]
        ax.errorbar(x=k_lst,
                     y=weights_dict[curr_weight][1],
                     fmt=fmt[i],
                     markersize=1.5, 
                     linewidth=0.5,
                     label=curr_name, 
                     yerr=weights_dict[curr_weight][2])
        ax.set_xticks(k_lst)
        plt.legend(loc='lower center')
        plt.tight_layout()

    filename = get_project_results_dir().joinpath(dataset_name + '_versions.eps')

    # pu.save_fig(fig, str(filename))
    plt.show()

def plot_hq_mtx(weights_dict):
    weights_lst = weights_dict['weights_lst']
    X = weights_dict['X']
    y = weights_dict['y']
    target_names = weights_dict['target_names']
    dataset_name = weights_dict['dataset_name']
    names_lst = weights_dict['names_lst']

    pu.figure_setup()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True, stratify=y)

    fig_size = pu.get_fig_size(15, 4.4)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    for i in range(3):
        ax = fig.add_subplot(1,3,i+1)
        ax.set_axisbelow(True)

        curr_weight = weights_lst[i]
        curr_name = names_lst[i]
        classifier = Knn(n_neighbors=5,weights=curr_weight).fit(X_train, y_train)
        plot_confusion_matrix(classifier, X_test, y_test,
                                display_labels=target_names,
                                ax=ax,
                                cmap=plt.cm.Blues,
                                normalize=None
                                )
        
        ax.set_title(curr_name)

    plt.tight_layout()
    
    filename = get_project_results_dir().joinpath(dataset_name + '_cf_mtx.eps')
    
    # pu.save_fig(fig, str(filename))
    plt.show()

def produce_report(weights_dict):
    plot_hq_summary(weights_dict)
    plot_hq_versions(weights_dict)
    plot_hq_mtx(weights_dict)
