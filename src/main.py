from neighbors._classification import Knn
from sklearn import neighbors, datasets
import numpy as np
from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.model_selection import cross_val_score

def delta(a,b):
    return 1 if a == b else 0

def main():
    mtz_a = [[0, 0], [1, 0], [2, 5]]
    y_a = [7,9,9]
    mtz_query = [[0, 0], [2, 5]]
    y_true = [9,9]

    # import some data to play with
    iris = datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target

    # deve guardar este cara para ler depois
    cv = KFold(n_splits=10,random_state=1,shuffle=True)

    # evaluate
    obj = Knn(n_neighbors=5)
    obj.fit(X,y)
    scores = cross_val_score(obj, X,y, scoring='accuracy', cv=cv, n_jobs=1)
    print('Accuracy Uniform: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    obj2 = Knn(n_neighbors=5,weights='distance')
    obj2.fit(X,y)
    scores = cross_val_score(obj2, X,y, scoring='accuracy', cv=cv, n_jobs=1)
    print('Accuracy Distance: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    obj2 = Knn(n_neighbors=5,weights='adaptive')
    obj2.fit(X,y)
    scores = cross_val_score(obj2, X,y, scoring='accuracy', cv=cv, n_jobs=1)
    print('Accuracy Adaptive: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))



if __name__ == "__main__":
    main()
