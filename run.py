import warnings

from main_functions import *

warnings.filterwarnings('ignore')


def main():
    check_shuffled_dfs()
    create_meta_df()
    create_grafics_get_best_scores()


if __name__ == '__main__':
    main()
