import torch
import matplotlib.pyplot as plt
import numpy as np
from .decomposition_utils import generate_domain_bounding_hyperplanes, get_model_parameters, make_coordinate_to_weights_dict
import csv

def get_network_value_list(config, dataloader, model):
    result_list = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred_thresh = torch.clamp(pred, min=0.0, max=float(config.num_labels)-1)
            for i in range(pred.shape[0]):
                result_list.append(float(pred_thresh[i]))
    return result_list

def make_figure(config, train_dataloader, model, data, total_hyperplane_list, job_index):
    result_list = get_network_value_list(config, train_dataloader, model)

    plt.style.use('_mpl-gallery-nogrid')

    fig1 = plt.figure(figsize=(15,5))
    ax = fig1.add_subplot(111)

    p = config.figure_width

    ax.set(xlim=(-p, p), xticks=np.arange(-p, p, step=1),
        ylim=(-p, p), yticks=np.arange(-p, p, step=1))
    ax.set_title(f'Learned Decomposition of Phase Space')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    #   Plot datapoints and color according to network certainty of classification
    scatterx1 = []
    scattery1 = []

    for i in range(len(data)):
        scatterx1.append(float(data[i][0]))
        scattery1.append(float(data[i][1]))

    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right = 0.9)

    for hyperplane in total_hyperplane_list:
        if np.isclose(np.asarray(hyperplane.normal_vec[1]), np.asarray([0])):
            plt.axvline(x = -hyperplane.bias/hyperplane.normal_vec[0], c = 'k', linewidth=0.5)
        elif np.isclose(np.asarray(hyperplane.normal_vec[0]), np.asarray([0])):
            plt.axhline(y = -hyperplane.bias/hyperplane.normal_vec[1], c = 'k', linewidth=0.5)
        else:
            normal_vec_0 = np.dot(np.asarray(hyperplane.normal_vec), np.asarray([1, 0]))
            normal_vec_1 = np.dot(np.asarray(hyperplane.normal_vec), np.asarray([0, 1]))
            yintercept = -hyperplane.bias/normal_vec_1
            slope = -normal_vec_0/normal_vec_1 
            b = float(yintercept)
            m = float(slope)
            x = np.linspace(-config.figure_width, config.figure_width, 10)
            y = m * x + b
            ax.plot(x, y, c = 'k', linewidth = 1)

    scatter = ax.scatter(scatterx1, scattery1, marker ='o', s = 5, cmap = 'viridis', c = result_list)
    cbar = fig1.colorbar(scatter, orientation = 'horizontal', fraction=0.046, pad=0.15)
    cbar.set_label(label = 'Numerical value of network on training data')
    filename = f'output/figures/system{config.system}/{job_index}-decomposition.png'
    plt.savefig(filename)
    plt.close(fig1)

def make_loss_plots(config, job_index, test_loss_list, train_loss_list):
    fig2 = plt.figure(figsize=(15,5))
    ax = fig2.add_subplot(111)
    example_type = config.example_type
    system = config.system
    ax.set_title(example_type + f' System {system} Example {job_index}: Loss for Test and Training Datasets')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss')

    timelist=list(range(1,config.epochs+1))

    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right = 0.9)
    ax.set(xticks=np.arange(0, config.epochs+1, step=50))
    ax.plot(timelist, test_loss_list, linewidth = .8, c = 'red', label = 'Test Loss')
    ax.plot(timelist, train_loss_list, linewidth = .8, c = 'k', label = 'Train Loss')
    ax.legend()
    filename = f'output/figures/system{system}/{job_index}-loss.png'
    plt.savefig(filename)
    plt.close(fig2)
