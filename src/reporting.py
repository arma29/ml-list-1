import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from src.utils import get_project_results_dir
from src.neighbors._classification import Knn

def plot_summary(weights_dict):
    k_lst = weights_dict['k_lst']
    weights_lst = weights_dict['weights_lst']
    dataset_name = weights_dict['dataset_name']
    names_lst = weights_dict['names_lst']

    fmt = ['ro--','g^--','bs--']
    
    plt.figure(figsize=(16,4))
    plt.suptitle(f"Dataset: {dataset_name.upper()}", fontsize=16)
    plt.tight_layout()
    
    plt.subplot(1, 2, 1)
    plt.ylabel("Tempo de Processamento (s)")
    plt.xlabel("Parâmetro K")
    for i in range(3):
        curr_weight = weights_lst[i]
        curr_name = names_lst[i]
        plt.plot(k_lst, weights_dict[curr_weight][0], fmt[i], markersize=5,label=curr_name)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.ylabel("Acurácia")
    plt.xlabel("Parâmetro K")
    for i in range(3):
        curr_weight = weights_lst[i]
        curr_name = names_lst[i]
        plt.plot(k_lst, weights_dict[curr_weight][1], fmt[i], markersize=5,label=curr_name)
    plt.legend()
    

    filename = get_project_results_dir().joinpath(dataset_name + '_summary.png')
    plt.savefig(filename)
    
def plot_versions(weights_dict):
    k_lst = weights_dict['k_lst']
    weights_lst = weights_dict['weights_lst']
    dataset_name = weights_dict['dataset_name']
    names_lst = weights_dict['names_lst']

    fmt = ['ro--','g^--','bs--']
    
    plt.figure(figsize=(16,4))
    plt.suptitle(f"Dataset: {dataset_name.upper()}", fontsize=16)
    plt.tight_layout()
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.ylabel("Acurácia")
        plt.xlabel("Parâmetro K")
        plt.ylim(-0.1,1.1)
        curr_weight = weights_lst[i]
        curr_name = names_lst[i]
        plt.errorbar(x=k_lst,
                     y=weights_dict[curr_weight][1],
                     fmt=fmt[i],
                     markersize=5, 
                     label=curr_name, 
                     yerr=weights_dict[curr_weight][2])
        plt.legend()
        
    filename = get_project_results_dir().joinpath(dataset_name + '_versions.png')
    plt.savefig(filename)

def plot_confusion_matrix_versions(weights_dict):
    weights_lst = weights_dict['weights_lst']
    X = weights_dict['X']
    y = weights_dict['y']
    target_names = weights_dict['target_names']
    dataset_name = weights_dict['dataset_name']
    names_lst = weights_dict['names_lst']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))

    for i in range(3):
        curr_weight = weights_lst[i]
        curr_name = names_lst[i]
        classifier = Knn(n_neighbors=2,weights=curr_weight).fit(X_train, y_train)
        ax = axes.flatten()[i]
        plot_confusion_matrix(classifier, X_test, y_test,
                                display_labels=target_names,
                                ax=ax,
                                cmap=plt.cm.Blues,
                                normalize=None
                                )
        
        ax.set_title(curr_name)

    plt.suptitle(f'Dataset: {dataset_name.upper()}', fontsize=16)
    plt.tight_layout()
    
    filename = get_project_results_dir().joinpath(dataset_name + '_cf_mtx.png')
    plt.savefig(filename)
    
    plt.show()


def produce_report(weights_dict):
    plot_summary(weights_dict)
    plot_versions(weights_dict)
    plot_confusion_matrix_versions(weights_dict)