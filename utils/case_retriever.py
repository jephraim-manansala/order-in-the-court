import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from tqdm import tqdm


def nearest_k(query, objects, k, dist=euclidean):
    """Return the nearest k objects to the query based on dist metric"""

    dists = np.array([dist(query, obj) for obj in objects])
    indices = np.argsort(dists)[:k]
    total_dist = np.sum(dists)
    return indices, 100 * ((total_dist - dists)/total_dist)[indices]


def pooled_within_ssd(X, y, centroids, dist):
    """Calculate the pooled with ssd"""

    from collections import Counter
    counts = Counter(y.astype(int))
    return (sum(dist(x_i, centroids[y_i])**2 / (2*counts[y_i])
                for x_i, y_i in zip(X, y.astype(int))))


def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    """Return the gap statistic based on the clusterer"""

    rng = np.random.default_rng(random_state)
    orig_wcss = pooled_within_ssd(X, y, centroids, dist)
    gaps = []
    for _ in range(b):
        x_ref = rng.uniform(np.min(X, axis=0), np.max(X, axis=0),
                            size=(X.shape))
        y_ref = clusterer.fit_predict(x_ref)
        ref_wcss = pooled_within_ssd(x_ref, y_ref, clusterer.cluster_centers_,
                                     dist)
        gaps.append(np.log(ref_wcss) - np.log(orig_wcss))
    return np.mean(gaps), np.std(gaps)


def cluster_range(X, clusterer, k_start, k_stop, actual=None):
    """Returns a dictionary of internal and external validation values over
    varying values of k. """

    ys = []
    inertias = []
    chs = []
    scs = []
    gss = []
    gssds = []
    ps = []
    amis = []
    ars = []
    for k in tqdm(range(k_start, k_stop+1)):
        clusterer_k = clone(clusterer).set_params(n_clusters=k)
        y = clusterer_k.fit_predict(X)

        gs = gap_statistic(X, y, clusterer_k.cluster_centers_,
                                 euclidean, 5,
                                 clone(clusterer).set_params(n_clusters=k),
                                 random_state=1337)
        ys.append(y)
        inertias.append(clusterer_k.inertia_)
        chs.append(calinski_harabasz_score(X,y))
        scs.append(silhouette_score(X,y))
        gss.append(gs[0])
        gssds.append(gs[1])
        rdict = {'ys':ys, 'inertias':inertias, 'chs':chs, 'scs':scs,
                'gss':gss, 'gssds':gssds}
        if actual is not None:
            ps.append(purity(actual,y))
            amis.append(adjusted_mutual_info_score(actual,y))
            ars.append(adjusted_rand_score(actual,y))
            rdict['ps']= ps
            rdict['amis']= amis
            rdict['ars']= ars
    return rdict


def plot_clusters(X, ys):
    """Plot clusters given the design matrix and cluster labels"""

    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(2, k_max//2, dpi=150, sharex=True, sharey=True,
                           figsize=(7,4), subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01))
    for k,y in zip(range(2, k_max+1), ys):
        if k < k_mid:
            ax[0][k%k_mid-2].scatter(*zip(*X), c=y, s=2, alpha=0.8,
                           cmap='viridis')
            ax[0][k%k_mid-2].set_title('$k=%d$'%k)
        else:
            ax[1][k%k_mid].scatter(*zip(*X), c=y, s=2, alpha=0.8,
                           cmap='viridis')
            ax[1][k%k_mid].set_title('$k=%d$'%k)
    return ax


def plot_internal(inertias, chs, scs, gss, gssds):
    """Plot internal validation values"""

    fig, ax = plt.subplots()
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE')
    ax.plot(ks, chs, '-ro', label='CH')
    ax.set_xlabel('$k$')
    ax.set_ylabel('SSE/CH')
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.errorbar(ks, gss, gssds, fmt='-go', label='Gap statistic')
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient')
    ax2.set_ylabel('Gap statistic/Silhouette')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    return ax


def _nearest_k(query, objects, k, dist):
    return np.argsort([dist(query, obj) for obj in objects])[:k]


def precision(confusion):
    """Return the precision given a confusion matrix"""

    pres = confusion.loc['relevant'].relevant/confusion.loc['relevant'].sum()
    return pres if not np.isnan(pres) else 1.0


def recall(confusion):
    """Return the recall given a confusion matrix"""

    rec = confusion.loc['relevant'].relevant/confusion.relevant.sum()
    return rec if not np.isnan(rec) else 1.0


def get_confusion(actual, results, all_labels):
    """Return a confusion matrix"""

    res = np.array(all_labels)[results]
    ac = np.delete(np.array(all_labels), results)
    data = [
        [(res == actual).sum(), (res != actual).sum()],
        [(ac == actual).sum(), (ac != actual).sum()]
    ]
    return pd.DataFrame(data,
                        columns=['relevant', 'irrelevant'],
                        index=['relevant', 'irrelevant'])


def closest_euc(query, objects, label, all_labels=None):
    """Return closest euc, precision and recall"""

    ret = _nearest_k(query, objects.to_numpy(), 10, euclidean)
    all_labels_ = (all_labels == label).astype('int')
    ret_idx = objects.iloc[ret].index.tolist()
    c_matrix = get_confusion(1, ret, all_labels_)
    return ret_idx, precision(c_matrix), recall(c_matrix)


def auc_pr(query, objects, dist, actual, all_labels):
    """Return area under the curve pr"""

    all_labels = np.asarray(all_labels)
    results = _nearest_k(query, objects, len(all_labels), dist)
    rs = (all_labels[results] == actual).cumsum()
    N = (all_labels == actual).sum()
    prcn = rs / np.arange(1, len(rs)+1)
    rcll = rs / N
    r = np.array([0.] + rcll.tolist())
    p = np.array([1.] + prcn.tolist())
    idx = np.array(range(1,len(r)))
    re_delta = r[idx] - r[idx - 1]
    pr_delta = p[idx] + p[idx - 1]
    return np.sum(re_delta * pr_delta) / 2


def compute_auc_euc(query, objects, label, dist, all_labels):
    """Return area under the curve euc"""

    all_labels = (all_labels == label).astype('int')
    return auc_pr(query, objects.to_numpy(), dist, 1, all_labels)
