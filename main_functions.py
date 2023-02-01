import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io.arff import loadarff
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from utils import *
from classifiers import *

def get_meta_features(df):
    basic_features = {
        "number_objects": df_data.shape[0],
        "number_features": df_data.shape[1] - 1,
        "number_numeric": len(df.select_dtypes(include=[np.number]).axes[1]),
        "number_nominal": len(df.select_dtypes(include=[object]).axes[1]) - 1,
    }
    df_copy = df.copy()
    name_pred = df_copy.columns[-1]
    x = df_copy.drop(name_pred, axis=1)
    y = df_copy[name_pred]
    xs = vectorize_xs(x)
    xs = norm_min_max(xs)
    ys = y.values
    xstat_features = get_statistical_characteristics(xs, 'numeric')
    ystat_features = get_statistical_characteristics(ys[:, None], 'nominal')
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(xs, ys)
    struct_features = {
        "leaves": dt.get_n_leaves(),
        "depth": dt.get_depth(),
    }
    class_features = {}
    all_scores = dict()
    for classifier_name in my_classifiers:
        cls = my_classifiers[classifier_name]()
        cls.fit(xs, ys)
        pred = cls.predict(xs)
        score = f1_score(ys, pred, average='micro')
        all_scores[classifier_name] = score
    best = [k for k, v in all_scores.items() if v == max(all_scores.values())][0]
    best_algo = [best]
    return {'best_algorithm': best_algo, **basic_features,
            **xstat_features, **ystat_features, **struct_features,
            **class_features, **all_scores}


def check_shuffled_dfs():
    print()
    print('=' * 20)
    print()
    global name, raw_data, df_data, str_df, col, features
    name = '/Users/esbessonngmail.com/Downloads/OpenML/data/997.arff'
    raw_data = loadarff(name)
    df_data = pd.DataFrame(raw_data[0])
    str_df = df_data.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df_data[col] = str_df[col]
    features = get_meta_features(df_data)
    print(features.values())
    new_data = df_data.copy().sample(frac=1)
    new_data['binaryClass'][new_data['binaryClass'] == 'N'] = 2
    new_data['binaryClass'][new_data['binaryClass'] == 'P'] = 'N'
    new_data['binaryClass'][new_data['binaryClass'] == 2] = 'P'
    features = get_meta_features(new_data)
    print(features.values())
    print()
    print('=' * 20)
    print()


def create_meta_df():
    print()
    print('=' * 20)
    print()
    global meta_df, name, raw_data, df_data, str_df, col, features
    meta_df = pd.DataFrame()
    dataset_fnames = os.listdir('/Users/esbessonngmail.com/Downloads/OpenML/data')
    for ind, fname in enumerate(dataset_fnames):
        name = f'/Users/esbessonngmail.com/Downloads/OpenML/data/{fname}'
        try:
            raw_data = loadarff(name)
            df_data = pd.DataFrame(raw_data[0])
            str_df = df_data.select_dtypes([object])
            str_df = str_df.stack().str.decode('utf-8').unstack()
            for col in str_df:
                df_data[col] = str_df[col]
            features = get_meta_features(df_data)
            row = pd.DataFrame(features)
            meta_df = pd.concat([meta_df, row], ignore_index=True, axis=0)
            print(f"Nominal file {fname}")
            print(features.values())
        except Exception as e:
            print(f"Numeric file {fname}")
            print(e)
    print(meta_df)
    print()
    print('=' * 20)
    print()


def create_grafics_get_best_scores():
    print()
    print('=' * 20)
    print()
    plt.style.use('ggplot')
    X = meta_df.loc[:, ~meta_df.columns.isin(['best_algorithm', *my_classifiers.keys()])]
    X.fillna(0, inplace=True)
    y = meta_df['best_algorithm']
    colors = []
    color_map = {0: 'red', 1: 'green', 2: 'blue'}
    for i in pd.factorize(y)[0]:
        colors.append(color_map[i])
    for x in [X, norm_min_max(X.values)]:
        pca = PCA(n_components=2)
        x_new = pca.fit_transform(x)
        plt.scatter(x_new[:, 0], x_new[:, 1], c=colors)
        plt.show()

    def naive_score(ys):
        val_class = ys.mode()[0]
        pred = [val_class for _ in range(len(ys))]
        print(f"The most commonly used algorithm (honestly): `{val_class}`")
        return f1_score(ys, pred, average='micro')

    xs = norm_min_max(X.values)
    ys = y.values
    all_scores = dict()
    for classifier_name in my_classifiers:
        cls = my_classifiers[classifier_name]()
        cls.fit(xs, ys)
        pred = cls.predict(xs)
        score = f1_score(ys, pred, average='micro')
        all_scores[classifier_name] = score
    best = [k for k, v in all_scores.items() if v == max(all_scores.values())][0]
    best, scores = [best], all_scores
    naive = naive_score(y)
    print(f'Honestly calculated f-score: {naive}')
    print(f"Predicted efficiency of algorithms: `{scores}`")
    print(f"Best Algorithm: `{best[0]}`")
    print()
    print('=' * 20)
    print()