from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DT = lambda: DecisionTreeClassifier(max_depth=4, random_state=0)
NB = lambda: GaussianNB()
SVM = lambda: SVC(kernel='poly', random_state=0)

my_classifiers = {
    'DT': DT,
    'NB': NB,
    'SVM': SVM,
}
