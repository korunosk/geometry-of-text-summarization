import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

import torch
from torch.utils.tensorboard import SummaryWriter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def make_pytorch_projector(log_dir, embeddings, global_step):
    '''Exports PyTorch projector'''
    writer = SummaryWriter(log_dir)
    writer.add_embedding(embeddings['mat'],
                         metadata=embeddings['labels'],
                         tag=embeddings['tag'],
                         global_step=global_step)
    writer.close()


def project_pca(embs, t, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    pts = pca.fit_transform(embs)
    return pts[:t], pts[t:]


def project_tsne(embs, t):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=5000, verbose=1, random_state=42)
    pts = tsne.fit_transform(embs)
    return pts[:t], pts[t:]


def plot_hull(ax, title, proj, document_pts, summary_pts):
    '''
    :param ax:              Axes containing the subplot
    :param title:           Title of the subplot
    :param proj:            Projection type for labeling the x- and y- axis
    :param document_pts:    Projected document points
    :param summary_pts:     Projected summary points
    '''
    def plot_pts(pts, alpha):
        ax.scatter(pts[:,0], pts[:,1], alpha=alpha)
        hull = ConvexHull(pts)
        idx = np.concatenate((hull.vertices, hull.vertices[:1]))
        ax.plot(pts[idx,0], pts[idx,1], linestyle='dashed', lw=2)

    plot_pts(document_pts, alpha=0.8)
    plot_pts(summary_pts, alpha=1)

    ax.set_title(title)
    ax.set_xlabel(f'{proj}-0')
    ax.set_ylabel(f'{proj}-1')


def plot_corr_coeff(ax, topics, scores):
    def make_indices(w, n):
        if n % 2 == 0:
            idx = w * (np.arange(n//2) + 0.5)
            idx = np.concatenate((-idx[::-1], idx))
        else:
            idx = w * (np.arange(n//2) + 1.0)
            idx = np.concatenate((-idx[::-1], [0], idx))
        return idx

    n = len(scores)
    w = 0.8 / 3 # n - there won't be more than 3
    x = np.arange(len(topics))
    idx = make_indices(w, n)
    for i in range(n): 
        ax.bar(x + idx[i],
               scores[i]['values'],
               width=w,
               label=scores[i]['label'])
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Kendall tau')
    ax.legend(loc='upper right')
    ax.grid()


def plot_corr(ax, df, topic_id, m):
    df_tmp = df[df['Topic ID'] == topic_id]
    x = df_tmp[m].values
    y = df_tmp['Pyramid Score'].values
    t = df_tmp['Summary ID'].values
    ax.scatter(x, y)
    for i in range(df_tmp.shape[0]):
        ax.annotate(t[i], (x[i]+0.1, y[i]))
    ax.set_xlabel(m)
    ax.set_ylabel('Pyramid Score')


def plot_loss(ax, loss):
    ax.plot(loss)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')