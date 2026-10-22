# -----------------------------------------------------------------------------
# This file is adapted from PatchTST (https://github.com/PatchTST/PatchTST)
# Original project: “PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.”
# Original repository license: Apache-2.0
# Copyright (c) Nie et al.
# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def visualize_input(input_features, index, writer_path):
    mpl.use('Agg')
    if not os.path.exists(os.path.join(os.path.join(writer_path, 'inputs/'))):
        os.makedirs(os.path.join(os.path.join(writer_path, 'inputs/')))
    channel_num = input_features.shape[0]
    rows = channel_num // 2 + 1
    color_list = []
    plt.figure(constrained_layout=True)
    for j in range(channel_num):
        plt.subplot(rows, 2, j+1)
        data_draw_tmp = input_features[j, :]
        color_tmp = np.random.rand(3)
        color_list.append(color_tmp)
        plt.plot(data_draw_tmp, c=color_tmp)
        plt.xlabel("Variable:{:.0f}/Timestep".format(j))
        
    input_figure_name = '{:d}.png'.format(index)
    input_figure_path = os.path.join(os.path.join(writer_path, 'inputs/'), input_figure_name)
    plt.savefig(input_figure_path)
    plt.close()
    
    input_file_name = '{:d}.npy'.format(index)
    input_save_path = os.path.join(os.path.join(writer_path, 'inputs/'), input_file_name)
    np.save(input_save_path,input_features)
    return color_list

def visualize_variable_corr(corr_feature, ch_num, index, writer_path):
    mpl.use('Agg')
    if not os.path.exists(os.path.join(os.path.join(writer_path, 'corrs/'))):
        os.makedirs(os.path.join(os.path.join(writer_path, 'corrs/')))
    
    corr_mat = corr_feature.reshape(ch_num, ch_num)
    ax = plt.matshow(corr_mat, cmap=plt.cm.Blues)
    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.title("Corr_Mat:{:.0f}".format(index))
    corr_figure_name = '{:d}.png'.format(index)
    corr_figure_path = os.path.join(os.path.join(writer_path, 'corrs/'), corr_figure_name)
    plt.savefig(corr_figure_path)
    plt.close()
    
    corr_file_name = '{:d}.npy'.format(index)
    corr_save_path = os.path.join(os.path.join(writer_path, 'corrs/'), corr_file_name)
    np.save(corr_save_path,corr_mat)
    
    return

def visualize_graph(graph_A, index, writer_path):
    mpl.use('Agg')
    if not os.path.exists(os.path.join(os.path.join(writer_path, 'graphs/'))):
        os.makedirs(os.path.join(os.path.join(writer_path, 'graphs/')))
        
    ax = plt.matshow(graph_A, cmap=plt.cm.Blues)
    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.title("Graph A:{:.0f}".format(index))
    graph_figure_name = '{:d}.png'.format(index)
    graph_figure_path = os.path.join(os.path.join(writer_path, 'graphs/'), graph_figure_name)
    plt.savefig(graph_figure_path)
    plt.close()
    
    graph_file_name = '{:d}.npy'.format(index)
    graph_save_path = os.path.join(os.path.join(writer_path, 'graphs/'), graph_file_name)
    np.save(graph_save_path,graph_A)
    
    return
