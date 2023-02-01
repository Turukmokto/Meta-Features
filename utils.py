from sklearn.feature_extraction import DictVectorizer

from extr_funs import *


def combine_dicts(from_dict, to_dict, key_transform=None, value_as_list=True):
    for func in from_dict:
        res_key_name = func if key_transform is None else key_transform(func)
        if value_as_list:
            if res_key_name not in to_dict:
                to_dict[res_key_name] = []
            to_dict[res_key_name].append(from_dict[func])
        else:
            to_dict[res_key_name] = from_dict[func]


def handle_one_feature(*args, feature_type):
    arr = np.nan_to_num(*args, nan=0)
    cur_results = {}
    if feature_type == 'numeric':
        cur_results["max"] = max(arr)
        cur_results["mean"] = sum(arr) / len(arr)
    if feature_type == 'nominal':
        cur_results["uniques_count"] = len(set(arr))
        cur_results["entropy"] = entropy(arr)
    return cur_results


def get_statistical_characteristics(xs, f_type='numeric'):
    first_stat_features = {}
    for feature_num in range(xs.shape[1]):
        res_dict = handle_one_feature(xs[:, feature_num], feature_type=f_type)
        combine_dicts(res_dict, first_stat_features)
    result_stat_features = {}
    for stat_name in first_stat_features:
        stats = first_stat_features[stat_name]
        if len(stats) == 1:
            result_stat_features[stat_name] = stats[0]
        else:
            res_dict = handle_one_feature(stats, feature_type='numeric')
            combine_dicts(res_dict, result_stat_features,
                          key_transform=lambda s: f"{stat_name}/{s}",
                          value_as_list=False)
    return result_stat_features


def vectorize_xs(xs):
    dv = DictVectorizer()
    xs_ = list(map(lambda x: dict(map(lambda t: (str(t[0]), t[1]), list(enumerate(x)))), xs.values))
    vectorized_xs = dv.fit_transform(xs_).toarray()
    return vectorized_xs


def norm_min_max(xs_):
    xs = xs_.copy()
    for col in range(xs.shape[1]):
        delta = (xs[:, col].max() - xs[:, col].min())
        if 0.0 == delta:
            xs[:, col] = xs[:, col] * 0
        else:
            xs[:, col] = (xs[:, col] - xs[:, col].min()) / delta
    return xs