# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pandas as pd
import numpy as np
import scipy.stats as stats
from StringIO import StringIO
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display, FileLink
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.cm as cm


def load_df(data_s):
    df = pd.read_csv(StringIO(data_s.replace(',', '.')),
                     sep='\t', encoding='utf8')
    df = df.transpose().reset_index()
    columns = df.iloc[0].values
    columns[0] = 'mission'
    columns[1] = 'type'
    df.columns = columns
    df = df.drop(0)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df


def transpose_df(df, type_=None):
    if type_ is not None:
        assert type_ in df.type.unique()
        df = df[df.type == type_]
    df2 = df.drop('type', axis=1).transpose()
    columns = df2.iloc[0].values
    df2.columns = columns
    df2 = df2.drop('mission')
    df2 = df2.apply(pd.to_numeric, errors='ignore')
    return df2


def plot_and_save_corr(df, method='pearson',
                       title_postfix=u'',
                       figsize=(16, 6),
                       fn_postfix='all',
                       fn_templ='mineral_composition_corr_{method}_{postfix}'):

    fn = fn_templ.format(**{'method': method, 'postfix': fn_postfix})

    df_corr = df.corr(method=method)
    display(df_corr)

    df_corr.to_csv(fn + '.csv', encoding='utf8')
    display(FileLink(fn + '.csv'))

    m = np.abs(df_corr)
    print(u"Минимальное значение: {:.4g}".format(m.values.min()))

    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    fig.suptitle(u'Корреляционая матрица по ненормированному составу '
                 u'грунта. {}\n'
                 u'Метод: {}'.format(title_postfix, method),
                 fontsize=14)
    ax = axes[0]
    sns.heatmap(m, cmap='hot_r', vmin=0, ax=ax)
    ax.set_title(u'Диапазон 0-1')
    ax = axes[1]
    sns.heatmap(m, cmap='hot_r', ax=ax)
    ax.set_title(u'Повышенный контраст')
    ax.text(0.5, -0.1,
            u"Минимальное значение: {:.4g}".format(m.values.min()),
            size=12, ha="center",
            transform=ax.transAxes)

    fig.savefig(fn + '.png')
    plt.show()

    display(FileLink(fn + '.png'))


def pairplot(df, title=None, figsize=(8, 8), save=False,
             fn_postfix='all',
             fn_templ='mineral_composition_pair_plot_{postfix}'):
    g = sns.PairGrid(df)
    g = g.map_upper(plt.scatter)
    g = g.map_lower(regplot_perason)
    g = g.map_diag(plt.hist)
    g.fig.set_size_inches(figsize[0], figsize[1])

    if title is not None:
        g.fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()

    if save:
        fn = fn_templ.format(**{'postfix': fn_postfix})
        g.fig.savefig(fn + ".png")
        display(FileLink(fn + '.png'))


def regplot_perason(x, y, **kw):
    ax = sns.regplot(x, y, **kw)
    ax_annotate(ax, x, y, stats.pearsonr)
    return ax


def ax_annotate(ax, x, y, func, template=None, stat=None, loc="best",
                **kwargs):
    default_template = "{stat} = {val:.3g};\npvalue = {p:.2g}"

    # Call the function and determine the form of the return value(s)
    out = func(x, y)
    try:
        val, p = out
    except TypeError:
        val, p = out, None
        default_template, _ = default_template.split(";")

    # Set the default template
    if template is None:
        template = default_template

    # Default to name of the function
    if stat is None:
        stat = func.__name__

    # Format the annotation
    if p is None:
        annotation = template.format(stat=stat, val=val)
    else:
        annotation = template.format(stat=stat, val=val, p=p)

    # Draw an invisible plot and use the legend to draw the annotation
    # This is a bit of a hack, but `loc=best` works nicely and is not
    # easily abstracted.
    #phantom, = ax.plot(x, y, linestyle="-", alpha=1)
    #ax.legend([phantom], [annotation], loc=loc, **kwargs)
    #phantom.remove()

    ax.set_title(annotation, fontsize=10)


def bounds(xx, increase=0.1):
    xmin = xx.min()
    xmax = xx.max()
    w = xmax - xmin

    res_min = xmin - increase * w / 2
    res_max = xmax + increase * w / 2
    return res_min, res_max


def plot_representation(points, X_type=None, X_labels=None,
                        title=None, ax=None, figsize=(8, 6), cmap=cm.rainbow,
                        title_fontsize=14,
                        fontsize=12):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    xx = points[:, 0]
    yy = points[:, 1]

    wx = xx.max() - xx.min()
    wy = yy.max() - yy.min()

    colors = cmap(np.linspace(0, 1, len(set(X_type))))

    for y, c in zip(set(X_type), colors):
        ax.scatter(xx[X_type == y], yy[X_type == y], c=c, label=y)

    for i, point in enumerate(zip(xx, yy)):
        ax.text(point[0] + 0.02 * wx,
                point[1] + 0.02 * wy, X_labels.values[i], fontsize=fontsize)

    ax.set_xlim(*bounds(xx))
    ax.set_ylim(*bounds(yy))
    ax.legend(fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)


def plot_mds(X, metric='cosine', ax=None, save=False,
             X_type=None, X_labels=None):

    mds = MDS(dissimilarity="precomputed", random_state=321)

    transformed = mds.fit_transform(pairwise_distances(X, metric=metric))

    title = 'MDS {}'.format(metric)

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_representation(transformed, title=title, ax=ax,
                        X_type=X_type, X_labels=X_labels)

    fig.suptitle(u'Понижение пространства признаков '
                 u'(минерального состава грунта)')
    if save:
        fn = 'mineral_composition_mds_{}.png'.format(metric)
        fig.savefig(fn)
        display(FileLink(fn))
