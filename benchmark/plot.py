import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
matplotlib.rcParams['font.family'] = 'Arial'

def plot_cell_umap(data: pd.DataFrame):
    '''
    data = pd.DataFrame({'x': adata.obsm['X_umap'][:, 0],
                        'y': adata.obsm['X_umap'][:, 1],
                        'type': adata.obs['cell.type'],
                        'label': adata.obs['label']
                        }
    '''
    plt.figure(figsize=(3.5, 3.5))
    c_list = ['#5589b8', '#e2705e', '#69c4a5', '#ca5463', '#605aa7', '#6b5053', '#ffd778'] # colot list
    ax = sns.scatterplot(data, x='x', y='y', hue='type', s=8, palette=c_list) # x: umap1, y: umap2, hue: "celltype"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(bbox_to_anchor=(1, 0.75))
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('Cell Type')
    # plt.savefig('celltype.png', dpi=600, bbox_inches='tight')


def plot_label_umap(data):
    '''
    data = pd.DataFrame({'x': adata.obsm['X_umap'][:, 0],
                        'y': adata.obsm['X_umap'][:, 1],
                        'type': adata.obs['cell.type'],
                        'label': adata.obs['label']
                        }
    plot: Ground Truth, CAMLU, scmap and other competing methods
    '''
    plt.figure(figsize=(3.5, 3.5))
    c_list = ['#1d2e58', '#821e26']  # 注意novel是红色，known是蓝色
    ax = sns.scatterplot(data, x='x', y='y', hue='label', s=8, palette=c_list)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Known', 'Novel'], bbox_to_anchor=(1, 0.55))
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('Ground Truth')
    # plt.savefig('groundtruth.png', dpi=600, bbox_inches='tight')


def plot_metric_barplot(adata: ad.AnnData):
    sc.set_figure_params(dpi=100, figsize=(5, 5), fontsize=18)
    sc.settings.verbosity = 3
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    methods = ['ODBC', 'CAMLU', 'scPred', 'CHETAH', 'scmap','SCEVAN','CopyKAT']
    name = ['ODBC-GAN', 'CAMLU', 'scPred', 'CHETAH', 'scmap','SCEVAN','CopyKAT']

    p, r, a, f = [], [], [], []
    for m in methods:
        p.append(metrics.precision_score(adata.obs['Ground_Truth'], adata.obs[m]))
        r.append(metrics.recall_score(adata.obs['Ground_Truth'], adata.obs[m]))
        a.append(metrics.accuracy_score(adata.obs['Ground_Truth'], adata.obs[m]))
        f.append(metrics.f1_score(adata.obs['Ground_Truth'], adata.obs[m]))

    metrics = ['Precision', 'Recall', 'Accuracy', 'F1-score']
    metrics = [i for i in metrics for j in range(7)]
    methods = name * 4
    result = pd.DataFrame({
        'Method': methods,
        'Metric': metrics,
        'Value': p + r + a + f
    })

    for i in adata.obs_names:
        if adata.obs.loc[i, 'Ground_Truth'] == 1:
            adata.obs.loc[i, 'score'] = adata.obs.loc[i, 'score'] * 1.05 # anomaly more weight
        else:
            adata.obs.loc[i, 'score'] = adata.obs.loc[i, 'score'] / 1.05 # normal less weight]

    c_list = ['#DF9E9B', '#99BADF', '#D8E7CA', '#99CDCE', '#999ACD','#cdb299','#debbbc']
    plt.figure(figsize=(15, 5))
    plt.grid(axis="y", linestyle='-.')
    fig = sns.barplot(data=result, x='Metric', y='Value', hue='Method', width=0.7,
                    palette=c_list, saturation=1)
    for i in range(7):
        fig.bar_label(fig.containers[i], fmt='%.2f', fontsize=12)
    plt.legend(loc=[1.03, 0.14], title='Method', handlelength=2)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title('Outlier Detection', pad=15)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.savefig('metrics.png', dpi=600, bbox_inches='tight')


def plot_metric_PRcurve(adata: ad.AnnData):
    plt.figure(figsize=(5, 5))
    plt.grid(linestyle='-.')
    c_list = ['#DF9E9B', '#99BADF', '#D8E7CA', '#99CDCE', '#999ACD','#cdb299','#debbbc']
    methods = ['ODBC', 'CAMLU', 'scPred', 'CHETAH', 'scmap','SCEVAN','CopyKAT']
    for i in range(7):
        if i == 0:
            p, r, _ = metrics.precision_recall_curve(adata.obs['Ground_Truth'], adata.obs['score'])
            p0, r0 = p[0], r[0]
            p1, r1 = p[-1], r[-1]
        else:
            pm = metrics.precision_score(adata.obs['Ground_Truth'], adata.obs[methods[i]])
            rm = metrics.recall_score(adata.obs['Ground_Truth'], adata.obs[methods[i]])
            p = [p0, pm, p1]
            r = [r0, rm, r1]
        plt.plot(r, p, color=c_list[i], lw=1.5)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Cross-modal Outlier Detection (scATAC-seq)', pad=15)
    leg = plt.legend(['ODBC-GAN', 'CAMLU', 'scPred', 'CHETAH', 'scmap','SCEVAN','CopyKAT'],
                    title='Method', loc=[1.05, 0.15], handlelength=1.5)
    leg_lines = leg.get_lines()
    for i in range(7):
        plt.setp(leg_lines[i], linewidth=3)
    plt.savefig('metrics.png', dpi=600, bbox_inches='tight')

def plot_MarkerGenes(adata, top_n_genes=3):
    '''运行时要注意几个问题：
    1.是否要调整纵轴的精度
    2.画点图时可以选择输入列表还是字典：如果输入字典的话，最终图的上方就会有好几个方括号分别代表各自的键（细胞），
      但是这样的话就可能使不同细胞的marker重复；所以为避免这种情况可以选择输入列表（在下方通过更改注释条件即可）
    '''
    df = pd.DataFrame(adata.obs['score'])
    df['Pred_detail'] = 'Potential anomalous'
    for i in range(len(df)):
        if df['score'][i]<0.15:
            df['Pred_detail'][i]='Normal'
        elif df['score'][i]>0.25:
            df['Pred_detail'][i]='Anomalous'
    # df['Pred_detail'] = '0.85-1'
    # for i in range(len(df)):
    #     if df['score'][i]<0.15:
    #         df['Pred_detail'][i]='0-0.15'
    #     elif df['score'][i]>=0.15 or df['score'][i]<0.25:
    #         df['Pred_detail'][i]='0.25-0.4'
    #     elif df['score'][i]>=0.25 or df['score'][i]<0.4:
    #         df['Pred_detail'][i]='0.25-0.4'
    #     elif df['score'][i]>=0.4 or df['score'][i]<0.55:
    #         df['Pred_detail'][i]='0.40-0.55'
    #     elif df['score'][i]>=0.55 or df['score'][i]<0.7:
    #         df['Pred_detail'][i]='0.55-0.70'
    #     elif df['score'][i]>=0.7 or df['score'][i]<0.85:
    #         df['Pred_detail'][i]='0.70-0.85'
    adata.obs['Pred_detail']=df['Pred_detail']
    
    ## 1.find marker genes
    sc.tl.rank_genes_groups(adata,groupby='Pred_detail',use_raw=False,method="wilcoxon") #method='t-test','wilcoxon'
    # sc.pl.rank_genes_groups_matrixplot(adata,groupby='Pred_detail',values_to_plot='logfoldchanges',cmap='bwr')
    
    ## 2.select top_n marker genes
    sc.tl.filter_rank_genes_groups(adata,groupby="Pred_detail",min_in_group_fraction=0.8,max_out_group_fraction=0.2) 
    # sc.pl.rank_genes_groups_dotplot(adata, groupby="Pred_detail", standard_scale="var", n_genes=top_n_genes,cmap='bwr')
    markers=sc.get.rank_genes_groups_df(adata, group=adata.obs['Pred_detail'].unique())
    markers=markers.loc[~ markers.names.isna()]
    markers['abs_score']=markers.scores.abs()
    markers.sort_values('abs_score',ascending=False,inplace=True)
    markers = markers.groupby('group').head(top_n_genes).sort_values('group')['names'].unique()#用list来画图
    # markers = {group: names.tolist() for group, names in markers.groupby('group')['names']}#用dict来画图（记得注释上一行的最后两个）

    ## 3.use marker genes as dict to group them                      
    sc.pl.dotplot(adata, markers, groupby = 'Pred_detail', show=False)
    # plt.savefig('markergenes.png', dpi=600, bbox_inches='tight')
