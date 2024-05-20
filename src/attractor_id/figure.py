from .hyperplane import Hyperplane
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

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

def make_decomposition_figure(config, model, total_hyperplane_list, show, file_name):
    total_hyperplane_list.extend(generate_domain_bounding_hyperplanes(config))
    #result_list = get_network_value_list(config, figure_dataloader, model)
    data_bounds = config.data_bounds
    x_min = float(data_bounds[0])
    x_max = float(data_bounds[1])
    y_min = float(data_bounds[2])
    y_max = float(data_bounds[3])
    width = float(x_max - x_min)
    height = float(y_max - y_min)

    ratio = width/height

    plt.style.use('_mpl-gallery-nogrid')

    fig1 = plt.figure(figsize=(10, 10 * ratio + 2))
    ax = fig1.add_subplot(111)

    #extra_room_x = width * 0.05 * ratio
    extra_room_x = 0
    #extra_room_y = height * 0.05
    extra_room_y = 0

    if x_max > 10:
        step = 10
    else:
        step = 1
    ax.set(xlim=(x_min - extra_room_x, x_max + extra_room_x), xticks=np.arange(x_min + 1, x_max, step=step),
        ylim=(y_min - extra_room_y, y_max + extra_room_y), yticks=np.arange(y_min + 1, y_max, step=step))
    #ax.set_title(f'Learned Decomposition of Phase Space')
    ax.set_xlabel('x', fontsize = 22)
    ax.set_ylabel('y', fontsize = 22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    #   Plot datapoints and color according to network certainty of classification
    scatterx = []
    scattery = []
    #result_list = []

#    for i in range(len(data)):
#        scatterx1.append(float(data[i][0]))
#        scattery1.append(float(data[i][1]))
    subdivisions = 250
    result_list = []
    for i in range(subdivisions):
        x = x_min + (width/subdivisions) * i
        for j in range(subdivisions):
            y = y_min + (height/subdivisions) * j
            scatterx.append(float(x))
            scattery.append(float(y))

            values = np.zeros(2)
            values[0] += float(x)
            values[1] += float(y)
            data_tensor = torch.tensor(np.asarray([values]), dtype = torch.float32)
            pred = model(data_tensor)
           # pred_thresh = torch.clamp(pred, min=0.0, max=float(config.num_labels)-1)
            result_list.append(float(pred[0]))

    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right = 0.9)

    if len(total_hyperplane_list) > 24:
        linewidth = 1
    else:
        linewidth = 5
    for hyperplane in total_hyperplane_list:
        if np.isclose(np.asarray(hyperplane.normal_vec[1]), np.asarray([0])):
            plt.axvline(x = -hyperplane.bias/hyperplane.normal_vec[0], ymin = y_min, ymax = y_max, c = 'w', linewidth=linewidth)
        elif np.isclose(np.asarray(hyperplane.normal_vec[0]), np.asarray([0])):
            plt.axhline(y = -hyperplane.bias/hyperplane.normal_vec[1], xmin = x_min, xmax = x_max, c = 'w', linewidth=linewidth)
        else:
            normal_vec_0 = np.dot(np.asarray(hyperplane.normal_vec), np.asarray([1, 0]))
            normal_vec_1 = np.dot(np.asarray(hyperplane.normal_vec), np.asarray([0, 1]))
            yintercept = -hyperplane.bias/normal_vec_1
            slope = -normal_vec_0/normal_vec_1 
            b = float(yintercept)
            m = float(slope)
            x = np.linspace(x_min, x_max, 10)
            y = m * x + b
            ax.plot(x, y, c = 'w', linewidth = linewidth)

    scatter = ax.scatter(scatterx, scattery, marker ='o', s = 6, cmap = 'viridis', c = result_list, alpha = 1)
    cbar = fig1.colorbar(scatter, orientation = 'horizontal', fraction=0.05, pad=.13, format="%.2f")
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label = 'Network value', fontsize = 22)

    plt.savefig(file_name)
    if show:
        plt.show()
    plt.close(fig1)

def make_loss_plots(config, system, example_index, test_loss_list, train_loss_list):
    fig2 = plt.figure(figsize=(15,5))
    ax = fig2.add_subplot(111)
    example_type = config.example_type
    ax.set_title('Test and train loss: '+ example_type + f' (Example {example_index})')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss')

    timelist=list(range(1,config.epochs+1))

    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right = 0.9)
   # if config.epochs > 100:
      #  step = config.epochs/100
   # else:
    step = config.epochs/10
    ax.set(xticks=np.arange(0, config.epochs+1, step=step))
    ax.plot(timelist, test_loss_list, linewidth = 1.2, c = 'blueviolet', linestyle = 'dashed', label = 'Test Loss')
    ax.plot(timelist, train_loss_list, linewidth = 1, c = 'darkorange', label = 'Train Loss')
    ax.legend()
    filename = f'output/figures/{system}/{example_index}-loss.png'
    plt.savefig(filename)
    plt.close(fig2)

def label_to_polytope_color_and_hatch(config, label):
    num_labels = config.num_labels
    if num_labels > 4:
        print('Warning: Polytope plot only implemented for 2, 3, or 4 label figures')

    ''' Colors are from https://rpubs.com/mjvoss/psc_viridis '''
    color_0 = '#440154FF'
    color_1 = '#2A788EFF'
    color_extra = '#22A884FF'
    color_2 = '#FDE725FF'
    color_u = '#7AD151FF'

    hatch_0 = '.'
    hatch_1 = 'O'
    hatch_2 = 'o'
    hatch_u = '*'
    hatch_extra = 'O.'
    if num_labels == 2:
        if label == 0:
            return color_0, hatch_0
        if label == 1:
            return color_2, hatch_2
        if label == 2:
            return color_u, hatch_u
    if num_labels == 3:
        if label == 0:
            return color_0, hatch_0
        if label == 1:
            return color_1, hatch_1
        if label == 2:
            return color_2, hatch_2
        if label == 3:
            return color_u, hatch_u
    if num_labels == 4:
        if label == 0:
            return color_0, hatch_0
        if label == 1:
            return color_1, hatch_1
        if label == 2:
            return color_extra, hatch_extra
        if label == 3:
            return color_2, hatch_2
        if label == 4:
            return color_u, hatch_u

def plot_polytopes(config, cube_list_for_polytope_figure, show, file_name):

    data_bounds = config.data_bounds
    x_min = float(data_bounds[0])
    x_max = float(data_bounds[1])
    y_min = float(data_bounds[2])
    y_max = float(data_bounds[3])
    width = float(x_max - x_min)
    height = float(y_max - y_min)

    extra_room_x = 0
    extra_room_y = 0

    ratio = width/height

    plt.style.use('_mpl-gallery-nogrid')

    fig = plt.figure(figsize=(10, 10 * ratio + 2))

    ax = fig.add_subplot(111)

    if x_max > 10:
        step = 10
    else:
        step = 1

    ax.set(xlim=(x_min - extra_room_x, x_max + extra_room_x), xticks=np.arange(1 + x_min - extra_room_x, x_max + extra_room_x, step=step),
        ylim=(y_min - extra_room_y, y_max + extra_room_y), yticks=np.arange(1 + y_min - extra_room_y, y_max + extra_room_y, step=step))
  #  ax.set_title(f'Labeled Cubical Decomposition of Phase Space', fontsize = 18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_xlabel('x', fontsize = 22)
    ax.set_ylabel('y', fontsize = 22)

    if len(cube_list_for_polytope_figure) > 30:
        linewidth = 1
    else:
        linewidth = 5

    for cube in cube_list_for_polytope_figure:
        x = []
        y = []
        for vertex in cube.vertex_list:
            x.append(vertex[0])
            y.append(vertex[1])
        label = cube.label
        color, hatch = label_to_polytope_color_and_hatch(config, label)

        ax.fill(x, y, facecolor=color, edgecolor='white', linewidth=linewidth, hatch=hatch)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])

    plt.subplots_adjust(left=0.1, bottom=0.25, top=0.9, right = 0.9)

    if config.num_labels == 2:
        color_0, hatch_0 = label_to_polytope_color_and_hatch(config, 0)
        circ0 = mpatches.Patch(facecolor=color_0,hatch=hatch_0,edgecolor='white',label='Label 0')
        color_1, hatch_1 = label_to_polytope_color_and_hatch(config, 1)
        circ1 = mpatches.Patch(facecolor=color_1,hatch=hatch_1,edgecolor='white',label='Label 1')
        color_u, hatch_u = label_to_polytope_color_and_hatch(config, 2)
        circu = mpatches.Patch(facecolor=color_u,hatch=hatch_u,edgecolor='white',label='Uncertain')
        ax.legend(handles = [circ0, circ1, circu], loc='lower center', fontsize=22, bbox_to_anchor=(0.5, 0.12),
        bbox_transform=fig.transFigure, fancybox=True, facecolor='white', framealpha=1, ncol=3)
        
    if config.num_labels == 3:
        color_0, hatch_0 = label_to_polytope_color_and_hatch(config, 0)
        circ0 = mpatches.Patch(facecolor=color_0,hatch=hatch_0,label='Label 0')
        color_1, hatch_1 = label_to_polytope_color_and_hatch(config, 1)
        circ1 = mpatches.Patch(facecolor=color_1,hatch=hatch_1,label='Label 1')
        color_2, hatch_2 = label_to_polytope_color_and_hatch(config, 2)
        circ2 = mpatches.Patch(facecolor=color_2,hatch=hatch_2,label='Label 2')
        color_u, hatch_u = label_to_polytope_color_and_hatch(config, 3)
        circu = mpatches.Patch(facecolor=color_u,hatch=hatch_u,label='Uncertain')
        ax.legend(handles = [circ0, circ1, circ2, circu], loc='lower center', fontsize=20, bbox_to_anchor=(0.5, 0.12),
        bbox_transform=fig.transFigure, fancybox=True, facecolor='white', framealpha=1, ncol=4)
        
    if config.num_labels == 4:
        color_0, hatch_0 = label_to_polytope_color_and_hatch(config, 0)
        circ0 = mpatches.Patch(facecolor=color_0,hatch=hatch_0,label='Label 0')
        color_1, hatch_1 = label_to_polytope_color_and_hatch(config, 1)
        circ1 = mpatches.Patch(facecolor=color_1,hatch=hatch_1,label='Label 1')
        color_2, hatch_2 = label_to_polytope_color_and_hatch(config, 2)
        circ2 = mpatches.Patch(facecolor=color_2,hatch=hatch_2,label='Label 2')
        color_3, hatch_3 = label_to_polytope_color_and_hatch(config, 3)
        circ3 = mpatches.Patch(facecolor=color_3,hatch=hatch_3,label='Label 3')
        color_u, hatch_u = label_to_polytope_color_and_hatch(config, 4)
        circu = mpatches.Patch(facecolor=color_u,hatch=hatch_u,label='Uncertain')
        ax.legend(handles = [circ0, circ1, circ2, circ3, circu], loc='lower center', fontsize=20, bbox_to_anchor=(0.5, 0.06),
        bbox_transform=fig.transFigure, fancybox=True, facecolor='white', framealpha=1, ncol = 5)
       # ax.legend2(handles = [circ3, circu], loc='lower center', fontsize=20, bbox_to_anchor=(0.5, 0.03),
       # bbox_transform=fig.transFigure, fancybox=True, facecolor='white', framealpha=1, ncol=2)

    # Put a legend to the right of the current axis

 #   fig,(ax) = plt.subplots()
        
   # fig.subplots_adjust(bottom=0.3, wspace=0.33)

    plt.savefig(file_name)

    if show:
        plt.show()
    plt.close(fig)




