from .hyperplane import Hyperplane
import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_domain_bounding_hyperplanes(config):
    data_bounds_list = config.data_bounds
    d = len(data_bounds_list)//2
    domain_bounding_hyperplanes = []
    for i in range(0, d):
        normal_vec = np.zeros(d)
        normal_vec[i] = 1
        H1 = Hyperplane(normal_vec, -data_bounds_list[2*i]) # Why is this minus and not plus?
        H2 = Hyperplane(normal_vec, -data_bounds_list[2*i + 1])
        domain_bounding_hyperplanes.extend([H1, H2])
    return domain_bounding_hyperplanes

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
    total_hyperplane_list.extend(generate_domain_bounding_hyperplanes(config))
    result_list = get_network_value_list(config, train_dataloader, model)
    data_bounds = config.data_bounds
    x_min = float(data_bounds[0])
    x_max = float(data_bounds[1])
    y_min = data_bounds[2]
    y_max = data_bounds[3]
    width = float(x_max - x_min)
    height = float(y_max - y_min)

    ratio = width/height

    plt.style.use('_mpl-gallery-nogrid')

    fig1 = plt.figure(figsize=(10, 10 * ratio))
    ax = fig1.add_subplot(111)

    extra_room_x = width * 0.05 * ratio
    extra_room_x = 0
    extra_room_y = height * 0.05
    extra_room_y = 0
    ax.set(xlim=(x_min - extra_room_x, x_max + extra_room_x), xticks=np.arange(x_min, x_max, step=1),
        ylim=(y_min - extra_room_y, y_max + extra_room_y), yticks=np.arange(y_min, y_max, step=1))
    ax.set_title(f'Learned Decomposition of Phase Space (System {config.system})')
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
            plt.axvline(x = -hyperplane.bias/hyperplane.normal_vec[0], ymin = y_min, ymax = y_max, c = 'k', linewidth=0.5)
        elif np.isclose(np.asarray(hyperplane.normal_vec[0]), np.asarray([0])):
            plt.axhline(y = -hyperplane.bias/hyperplane.normal_vec[1], xmin = x_min, xmax = x_max, c = 'k', linewidth=0.5)
        else:
            normal_vec_0 = np.dot(np.asarray(hyperplane.normal_vec), np.asarray([1, 0]))
            normal_vec_1 = np.dot(np.asarray(hyperplane.normal_vec), np.asarray([0, 1]))
            yintercept = -hyperplane.bias/normal_vec_1
            slope = -normal_vec_0/normal_vec_1 
            b = float(yintercept)
            m = float(slope)
            x = np.linspace(x_min, x_max, 10)
            y = m * x + b
            ax.plot(x, y, c = 'k', linewidth = 2)

    scatter = ax.scatter(scatterx1, scattery1, marker ='o', s = 5, cmap = 'viridis', c = result_list)
    cbar = fig1.colorbar(scatter, orientation = 'horizontal', fraction=0.05, pad=.11, format="%.2f")
    cbar.set_label(label = 'Value of network on training data')
    filename = f'output/figures/system{config.system}/{job_index}-decomposition.png'
    plt.savefig(filename)
    plt.close(fig1)

def make_loss_plots(config, job_index, test_loss_list, train_loss_list):
    fig2 = plt.figure(figsize=(15,5))
    ax = fig2.add_subplot(111)
    example_type = config.example_type
    system = config.system
    ax.set_title('Test and train loss: '+ example_type + f' (System {system}, Example {job_index})')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss')

    timelist=list(range(1,config.epochs+1))

    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right = 0.9)
    if config.epochs > 100:
        step = config.epochs/100
    else:
        step = config.epochs/10
    ax.set(xticks=np.arange(0, config.epochs+1, step=step))
    ax.plot(timelist, test_loss_list, linewidth = 1.2, c = 'blueviolet', linestyle = 'dashed', label = 'Test Loss')
    ax.plot(timelist, train_loss_list, linewidth = 1, c = 'darkorange', label = 'Train Loss')
    ax.legend()
    filename = f'output/figures/system{system}/{job_index}-loss.png'
    plt.savefig(filename)
    plt.close(fig2)
