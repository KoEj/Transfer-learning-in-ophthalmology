import numpy as np
from scipy import stats
import os
import results.results as results

classifiers = ['ResNet_clear', 'ResNet50', 'MAE_clear', 'MAE', 'RETFound_oct', 'RETFound_cfp']
scores_path = os.path.dirname(os.path.abspath(__file__))

def statistics(scoresDone):
    alfa = 0.05
    scores_mean = np.mean(scoresDone, axis=2)
    print(np.around(scores_mean, decimals=3))
    mean_scores = np.mean(scores_mean, axis=0)
    print("Scores mean:", np.around(mean_scores, decimals=3))

    ranks = []
    for scores in scores_mean:
        ranks.append(stats.rankdata(-scores).tolist())
    ranks = np.array(ranks)
    print("Ranks:", ranks)

    mean_ranks = np.mean(ranks, axis=0)
    print("Mean ranks:", mean_ranks)

    alfa = .05
    w_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_rank_value = np.zeros((len(classifiers), len(classifiers)))

    for i in range(len(classifiers)):
        for j in range(len(classifiers)):
            w_statistic[i, j], p_rank_value[i, j] = stats.ranksums(ranks.T[i], ranks.T[j])

    significantlyBetterRank = np.logical_and(w_statistic > 0, p_rank_value <= alfa)
    indexed_matrix_ranks = got_mapped_matrix([significantlyBetterRank])

    t_statistic = np.zeros((3, len(classifiers), len(classifiers)))
    p_value = np.zeros((3, len(classifiers), len(classifiers)))
    
    for i in range(scoresDone.shape[0]):
        # DATASETS
        for j in range(scoresDone.shape[1]):
            # CLASSIFIERS
            for k in range(scoresDone.shape[1]):
                t_statistic[i, j, k], p_value[i, j, k] = t_test_corrected(scoresDone[i, j], scoresDone[i, k])
    
    significantlyBetterStatArray = np.logical_and(t_statistic > 0, p_value <= alfa)
    indexed_matrix_clfs = got_mapped_matrix(significantlyBetterStatArray)

    return indexed_matrix_ranks, indexed_matrix_clfs

def generate_latex_code(scoresDone, table):
    latex_generated = []
    scores_mean = np.mean(scoresDone, axis=2)

    for n, table_element in enumerate(table):
        latex_acc = " & ".join(["{}".format(str(format(x, '.3f'))) for x in scores_mean[n]]) + ' \\\\'
        latex_generated.append(latex_acc)
        mapping_in = {'ResNet_clear': [], 'ResNet50': [], 'MAE_clear': [], 'MAE': [], 'RETFound_oct': [], 'RETFound_cfp': []}
        mapping_out = {'ResNet_clear': 1, 'ResNet50': 2, 'MAE_clear': 3, 'MAE': 4, 'RETFound_oct': 5, 'RETFound_cfp': 6}

        for pair in table_element:
            mapping_in[pair[0]].append(mapping_out[pair[1]])

        latex_labels = '& '
        for key in mapping_in:
            if len(mapping_in[key]) == 0:
                values = "$^-$ & "
                latex_labels = latex_labels + values
            else:
                values = " ".join(["$^{}$".format(str(x)) for x in mapping_in[key]])
                latex_labels = latex_labels + values
                if str(key) != str(list(mapping_in.keys())[-1]):
                    latex_labels = latex_labels + ' & '
                else:
                    latex_labels = latex_labels + ' \\\\ '

        latex_generated.append(latex_labels)
    # print(latex_generated)

    with open(scores_path + 'score.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in latex_generated))


def got_mapped_matrix(array):
    mapping = {0: 'ResNet_clear', 1: 'ResNet50', 2: 'MAE_clear', 3: 'MAE', 4: 'RETFound_oct', 5: 'RETFound_cfp'}
    listOfTrue = np.argwhere(array)

    new_list, temp = [], listOfTrue[-1]
    for item in range(0, temp[0]+1):
        new_list.append([[x[1], x[2]] for x in listOfTrue if x[0] == item])

    return [[(mapping[x[0]], mapping[x[1]]) for x in sublist] for sublist in new_list]


def t_test_corrected(a, b, J=5, k=2):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    As default for 5x5CV
    """
    if J*k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J*k
        ))

    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval


if __name__ == '__main__':
    transrate_scores = results.transrate_scores
    gnb_scores = results.GNB_Accuracy
    knn_scores = results.KNN_Accuracy
    svc_scores = results.SVC_Accuracy

    print("=== Analysis of TRANSRATE ===")
    transrate_ranks_table, transrate_clfs_table = statistics(transrate_scores)
    generate_latex_code(transrate_scores, transrate_clfs_table)

    print('-----GNB-------')
    transrate_ranks_table, transrate_clfs_table = statistics(gnb_scores)
    generate_latex_code(gnb_scores, transrate_clfs_table)

    print('-----KNN-------')
    transrate_ranks_table, transrate_clfs_table = statistics(knn_scores)
    generate_latex_code(knn_scores, transrate_clfs_table)

    print('-----SVC-------')
    transrate_ranks_table, transrate_clfs_table = statistics(svc_scores)
    generate_latex_code(svc_scores, transrate_clfs_table)

    # print("\n=== Analysis of H-SCORE ===")
    # hscore_ranks_table, hscore_clfs_table = statistics(hscore_scores)
    # generate_latex_code(hscore_scores, hscore_clfs_table)