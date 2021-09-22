import numpy as np
import tqdm
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


def perm_test(ts1, ts2, mask, alpha=0.05, corr='bonferroni'):
    ts1_test = ts1[:, mask]
    ts2_test = ts2[:, mask]
    nsub, tm = ts1_test.shape
    ts_mean_diff = ts1_test.mean(0)-ts2_test.mean(0)
    num_of_comparisons = tm
    null_dist_size = int((1/0.05)*num_of_comparisons*50)

    p_values = np.ones(tm)
    for i in tqdm.tqdm(range(null_dist_size)):
        ts1_sur = np.zeros((nsub, tm))
        ts2_sur = np.zeros((nsub, tm))
        sur_samples = np.random.permutation(nsub)
        half = np.random.choice([9, 10])
        ts1_sur[sur_samples[:half]] = ts1_test[sur_samples[:half]]
        ts1_sur[sur_samples[:half]] = ts2_test[sur_samples[:half]]
        ts2_sur[sur_samples[half:]] = ts1_test[sur_samples[half:]]
        ts2_sur[sur_samples[half:]] = ts2_test[sur_samples[half:]]
        p_values += abs(ts_mean_diff) <= (abs(ts1_sur.mean(0)-ts2_sur.mean(0)))

    p_values /= null_dist_size + 1

    if corr == 'bonferroni':
        p_values_corrected = p_values * num_of_comparisons
    elif corr == 'fdr':
        _, p_values_corrected = fdrcorrection(p_values, alpha=0.05,
                                              method='indep', is_sorted=False)

    p_values_sig = p_values_corrected < alpha/2

    # Add tpoints <= 0
    tmp = np.ones_like(mask)
    tmp[mask] = p_values
    p_values = tmp.copy()
    tmp[mask] = p_values_corrected

    p_values_corrected = tmp.copy()
    tmp = np.array([False]*len(mask))
    tmp[mask] = p_values_sig
    p_values_sig = tmp.copy()
    return p_values, p_values_corrected, p_values_sig


def reject_outliers(x, data_y, m=3):
    ids = np.where((abs(x - np.nanmean(x)) < m * np.nanstd(x)) &
                   (abs(data_y - np.nanmean(data_y)) <
                   m * np.nanstd(data_y)))[0]
    return x[ids], data_y[ids], ids


def conditional_probability(x1, x2, data_y, data_mask):
    x1 = x1[:, data_mask].flatten()
    x2 = x2[:, data_mask].flatten()
    y = data_y[:, data_mask].flatten()

    # Remove ouliers from a list

    # Get rid of outliers
    y_1, x1_t, r1 = reject_outliers(y, x1)
    y_2, x2_t, r2 = reject_outliers(y, x2)

    # 1 max and min
    xmin1 = y_1.min()
    xmax1 = y_1.max()
    ymin1 = x1_t.min()
    ymax1 = x1_t.max()

    # 2 max and min
    xmin2 = y_2.min()
    xmax2 = y_2.max()
    ymin2 = x2_t.min()
    ymax2 = x2_t.max()

    # create grid for density estimation
    y, x = np.mgrid[np.min((xmin1, xmin2)):np.max((xmax1, xmax2)):128j,
                    np.min((ymin1, ymin2)):np.max((ymax1, ymax2)):256j]

    # kernel density estimation
    positions = np.vstack([y.ravel(), x.ravel()])
    values = np.vstack([y_1, x1_t])
    kernel = stats.gaussian_kde(values, 'silverman')
    joint_prob_1 = np.reshape(kernel(positions).T, y.shape)

    # marginal probability of y
    prob_y1 = np.sum(joint_prob_1, axis=1)

    # kernel density estimate
    positions = np.vstack([y.ravel(), x.ravel()])
    values = np.vstack([y_2, x2_t])
    kernel = stats.gaussian_kde(values, 'silverman')

    joint_prob_2 = np.reshape(kernel(positions).T, y.shape)

    # marginal probability of y2
    prob_y2 = np.sum(joint_prob_2, axis=1)

    # conditional probabilities
    prob_cond1 = joint_prob_1/prob_y1[:, None]
    prob_cond2 = joint_prob_2/prob_y2[:, None]

    return prob_cond1, prob_cond2, y[:, 0], x[0, :]
