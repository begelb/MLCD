from .hyperplane import Hyperplane
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import csv
import random

''' This file contains the code to plot data, plot the cubical decomposition overlayed with the learned function, plot the labeled cubes, and plot the decomposition cubes with random colors '''

''' Get the color and hatch style for each label depending on the total  number of labels '''
def label_to_polytope_color_and_hatch(config, label):

    # Get the number of labels
    num_labels = config.num_labels
    if num_labels > 4:
        raise Exception('Polytope plot only implemented for 2, 3, or 4 label figures')

    # Define colors
    ''' Colors are from https://rpubs.com/mjvoss/psc_viridis '''
    color_0 = '#440154FF'
    color_1 = '#2A788EFF'
    color_extra = '#22A884FF'
    color_2 = '#FDE725FF'
    color_u = '#7AD151FF'

    # Define hatch styles 
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

''' Get the x_min, x_max, y_min, and y_max of the domain according to the config file '''
def get_x_y_min_max(config):
    data_bounds = config.data_bounds

    x_min = float(data_bounds[0][0])
    x_max = float(data_bounds[0][1])
    y_min = float(data_bounds[1][0])
    y_max = float(data_bounds[1][1])

    return x_min, x_max, y_min, y_max

''' Plot data '''
def plot_data(system, config, file_name):

    # Set figure font
    if system == 'linear_separatrix':
        font_size = 30
    else:
        font_size = 28

    font = {'family' : 'serif',
            'size'   : font_size}
    plt.rc('font', **font)

    # Set figure size
    x_min, x_max, y_min, y_max = get_x_y_min_max(config)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    ratio = width/height
    plt.style.use('_mpl-gallery-nogrid')

    if system == 'ellipsoidal_2d':
        fig1 = plt.figure(figsize = (7.25, 8))
    elif system == 'radial_bistable':
        fig1 = plt.figure(figsize = (7.25, 8.5))
    elif system == 'radial_tristable':
        fig1 = plt.figure(figsize = (7.25, 8.5))
    elif system == 'linear_separatrix':
        fig1 = plt.figure(figsize = (8, 7))
        #fig1 = plt.figure(figsize=(10 + 4, 10 * ratio + 4.75))
    else:
        #fig1 = plt.figure(figsize=(10 + 2, 10 * ratio + 4.75))
        fig1 = plt.figure(figsize = (7, 7))

    # Adjust layout
    ax = fig1.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])

    # Adjust layout and set xticks and yticks
    if system == 'ellipsoidal_2d':
        plt.subplots_adjust(left=0.19, bottom=0.3, top=0.9, right = 0.9)
        step = 2
        ax.set(xlim=(x_min, x_max), xticks=[-4, -2, 0, 2, 4],
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    elif system == 'radial_bistable':
        #plt.subplots_adjust(left=0.19, bottom=0.25, top=0.9, right = 0.9)
        plt.subplots_adjust(left=0.19, bottom=0.3, top=0.9, right = 0.9)
        step = 2
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    elif system == 'radial_tristable':
        #plt.subplots_adjust(left=0.19, bottom=0.25, top=0.9, right = 0.9)
        plt.subplots_adjust(left=0.19, bottom=0.3, top=0.9, right = 0.9)
        step = 2
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    elif system == 'linear_separatrix':
        plt.subplots_adjust(left=0.19, bottom=0.3, top=0.9, right = 0.9)
        ax.set(xlim=(x_min, x_max), xticks=[-2, 0, 2],
            ylim=(y_min, y_max), yticks=[-3, 0, 3])
    
    # Add vertical and horizontal axis labels
    # Set font size of the xticks and yticks
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks()
    plt.yticks()

    # Plot 1/6th of the data
    if system == 'radial_bistable':
        density = 1
    else:
        density = 6
    data = []
    counter = 0
    with open(config.train_data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            counter += 1
            if counter % density == 0:
                data.append((float(row[0]), float(row[1]), int(float(row[2]))))

    # Separate the data by ground-truth label
    x0 = [d[0] for d in data if d[2] == 0]
    y0 = [d[1] for d in data if d[2] == 0]
    x1 = [d[0] for d in data if d[2] == 1]
    y1 = [d[1] for d in data if d[2] == 1]
    x2 = [d[0] for d in data if d[2] == 2]
    y2 = [d[1] for d in data if d[2] == 2]

    # Get colors that correspond to labels
    color_0, hatch_0 = label_to_polytope_color_and_hatch(config, 0)
    color_1, hatch_1 = label_to_polytope_color_and_hatch(config, 1)

    # Set marker size
    if system == 'radial_bistable':
        marker_size = 90
    if system == 'ellipsoidal_2d':
        marker_size = 90
    if system == 'radial_tristable':
        marker_size = 90
    if system == 'linear_separatrix':
        marker_size = 100

    # Make scatter plot of data with labels 0 and 1
    ax.scatter(x0, y0, c=color_0, marker='^', label='Label 0', s = marker_size)
    ax.scatter(x1, y1, c=color_1, marker='o', label='Label 1', s = marker_size)
    
    # Add data with label 2 if necessary
    if config.num_labels > 2:
        color_2, hatch_2 = label_to_polytope_color_and_hatch(config, 2)
        ax.scatter(x2, y2, c=color_2, marker='s', label='Label 2', s = marker_size)
    
    if system == 'linear_separatrix':
        number_columns = 2
    elif system == 'ellipsoidal_2d':
        number_columns = 2
    elif system == 'radial_bistable':
        number_columns = 2
    elif system == 'radial_tristable':
        number_columns = 2

    # Add legend
    
    if system != 'radial_tristable':
        legend = ax.legend(loc='lower center', bbox_to_anchor=(0.545, 0.04),
            bbox_transform=fig1.transFigure, fancybox=True, scatterpoints=1, ncol = number_columns)
    else:
        legend = ax.legend(loc='lower center', bbox_to_anchor=(0.545, 0.0),
            bbox_transform=fig1.transFigure, fancybox=True, scatterpoints=1, ncol = number_columns)
    
    # Set size of markers in legend
    for handle in legend.legendHandles:
        handle.set_sizes([500])

    # Save the plot as a png and as a svg file 
    plt.savefig(file_name + '.png')
    plt.savefig(file_name + '.svg')

''' Generate the lines that bound the domain '''
def generate_domain_bounding_hyperplanes(config):

    # Read the domain and dimension from the configuration file
    data_bounds_list = config.data_bounds
    d = config.dimension

    # Create list of hyperplanes that support the rectangle that represents the domain
    domain_bounding_hyperplanes = []
    # The normal vectors are initialized as (1, 0) and (0, 1)
    for i in range(0, d):
        normal_vec = np.zeros(d)
        normal_vec[i] = 1
        # The class Hyperplane expects a normal vector n and a bias b to describe a Hyperplane by the equation <x, n> + b = 0, thus the negative sign below
        H1 = Hyperplane(normal_vec, -data_bounds_list[i][0])
        H2 = Hyperplane(normal_vec, -data_bounds_list[i][1])
        domain_bounding_hyperplanes.extend([H1, H2])
    return domain_bounding_hyperplanes

''' Plot the cubical decomposition together with a colormap depicting the learned function '''
def make_decomposition_figure(config, model, total_hyperplane_list, show, file_name, system):

    if system == 'linear_separatrix':
        font_size = 30
    else:
        font_size = 28
    # Set font properties
    font = {'family' : 'serif',
            'size'   : font_size}
    plt.rc('font', **font)

    # Add hyperplanes that support the rectangular domain to the list of hyperplanes to plot
    total_hyperplane_list.extend(generate_domain_bounding_hyperplanes(config))

    # Set layout properties
    x_min, x_max, y_min, y_max = get_x_y_min_max(config)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    ratio = width/height
    plt.style.use('_mpl-gallery-nogrid')

    if system == 'ellipsoidal_2d':
        fig1 = plt.figure(figsize = (7.25, 8.5))
    elif system == 'radial_bistable':
        fig1 = plt.figure(figsize = (7.25, 8.5))
    elif system == 'radial_tristable':
        fig1 = plt.figure(figsize = (7.25, 8))
    elif system == 'linear_separatrix':
        fig1 = plt.figure(figsize = (8, 7))
        #fig1 = plt.figure(figsize=(10 + 4, 10 * ratio + 4.75))
    else:
        #fig1 = plt.figure(figsize=(10 + 2, 10 * ratio + 4.75))
       # fig1 = plt.figure(figsize=(10 + 2, 10 * ratio + 4.75))
        fig1 = plt.figure(figsize = (7, 7))
    
    ax = fig1.add_subplot(111)

    # Adjust layout and set axis properties 
    step = 2

    if system == 'linear_separatrix':
        plt.subplots_adjust(left=0.19, bottom=0.1, top=0.9, right = 0.9)
        ax.set(xlim=(x_min, x_max), xticks=[-2, 0, 2],
            ylim=(y_min, y_max), yticks=[-3, 0, 3])
    elif system == 'ellipsoidal_2d':
        plt.subplots_adjust(left=0.19, bottom=0.1, top=0.9, right = 0.9)
        ax.set(xlim=(x_min, x_max), xticks=[-4, -2, 0, 2, 4],
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    elif system == 'radial_bistable':
        plt.subplots_adjust(left=0.19, bottom=0.1, top=0.9, right = 0.9)
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    elif system == 'radial_tristable':
        plt.subplots_adjust(left=0.19, bottom=0.1, top=0.9, right = 0.9)
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    else:
        #plt.subplots_adjust(left=0.19, bottom=0.25, top=0.9, right = 0.9)
        plt.subplots_adjust(left=0.19, bottom=0.1, top=0.9, right = 0.9)
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks()
    plt.yticks()

    # Plot points on a uniform grid and color according to the value of the network
    scatterx = []
    scattery = []
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
            result_list.append(float(pred[0]))
    scatter = ax.scatter(scatterx, scattery, marker ='o', s = 6, cmap = 'viridis', c = result_list, alpha = 1)
    max_network_value = max(result_list)
    # Plot lines with width determinted by total number
    if len(total_hyperplane_list) > 24:
        linewidth = 3
    else:
        linewidth = 5
    for hyperplane in total_hyperplane_list:
        # Plot vertical and horizontal lines
        if np.isclose(np.asarray(hyperplane.normal_vec[1]), np.asarray([0])):
            plt.axvline(x = -hyperplane.bias/hyperplane.normal_vec[0], ymin = y_min, ymax = y_max, c = 'w', linewidth=linewidth)
        elif np.isclose(np.asarray(hyperplane.normal_vec[0]), np.asarray([0])):
            plt.axhline(y = -hyperplane.bias/hyperplane.normal_vec[1], xmin = x_min, xmax = x_max, c = 'w', linewidth=linewidth)
        # Plot all other lines
        else:
            # Compute slope and y-intercept
            normal_vec_0 = np.dot(np.asarray(hyperplane.normal_vec), np.asarray([1, 0]))
            normal_vec_1 = np.dot(np.asarray(hyperplane.normal_vec), np.asarray([0, 1]))
            yintercept = -hyperplane.bias/normal_vec_1
            slope = -normal_vec_0/normal_vec_1 
            b = float(yintercept)
            m = float(slope)
            # Plot line using equation
            x = np.linspace(x_min, x_max, 10)
            y = m * x + b
            ax.plot(x, y, c = 'w', linewidth = linewidth)

    # Add colorbar
    if system == 'radial_tristable':
        print('max: ', max_network_value)
        cbar = fig1.colorbar(scatter, orientation = 'horizontal', fraction=0.05, pad=.13, format="%.0f", anchor = (0.5, 0.0), ticks = [0.0, 1.0, max_network_value])
    elif system == 'linear_separatrix':
        cbar = fig1.colorbar(scatter, orientation = 'horizontal', fraction=0.05, pad=0.20, format="%.1f", anchor = (0.5, 0.5), ticks = [0.0, 0.5, 1.0])
    elif system == 'ellipsoidal_2d':
        cbar = fig1.colorbar(scatter, orientation = 'horizontal', fraction=0.05, pad=0.20, format="%.1f", anchor = (0.5, 0.5), ticks = [0.0, 0.5, 1.0])
    elif system == 'radial_bistable':
        cbar = fig1.colorbar(scatter, orientation = 'horizontal', fraction=0.05, pad=0.20, format="%.1f", anchor = (0.5, 0.5), ticks = [0.0, 0.5, 1.0])
    cbar.ax.tick_params()

    # Save figure as png and svg file
    plt.savefig(file_name + '.png')
    plt.savefig(file_name + '.svg')

    # Show and close figure
    if show:
        plt.show()
    plt.close(fig1)

''' Plot the change of the test and train loss over the epochs of training '''
def make_loss_plots(config, test_loss_list, train_loss_list, file_name, show):
    font = {'family' : 'serif',
            'size'   : 15}
    plt.rc('font', **font)
    # Adjust plot layout
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right = 0.9)

    # Set title and axis labels
    ax.set_title('Test and train loss')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss')

    # Get list of epoch indices
    timelist=list(range(1,len(test_loss_list)+1))

    # Add x_ticks for every 10 epochs
    step = len(timelist)/10
    ax.set(xticks=np.arange(0, len(test_loss_list), step=step))

    # Plot linear interpoloation of the test and train loss
    ax.plot(timelist, test_loss_list, linewidth = 3.2, c = 'blueviolet', linestyle = 'dashed', label = 'Test Loss')
    ax.plot(timelist, train_loss_list, linewidth = 3, c = 'darkorange', label = 'Train Loss')

    # Add legend
    ax.legend()

    # Show
    if show:
        plt.show()

    # Save and close
    plt.savefig(file_name)
    plt.close(fig)

# Plot decomposition from neural network with multicolor polytopes
def plot_multicolor_polytopes(config, cube_list_for_polytope_figure, show, file_name, system):

    # Define font style
    font = {'family' : 'serif',
            'size'   : 45}
    plt.rc('font', **font)

    # Set layout
    x_min, x_max, y_min, y_max = get_x_y_min_max(config)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    ratio = width/height
    plt.style.use('_mpl-gallery-nogrid')
    step = 2
    if system != 'linear_separatrix':
        fig = plt.figure(figsize=(10 + 2, 10 * ratio + 4.75))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(left=0.19, bottom=0.25, top=0.9, right = 0.9)
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    elif system == 'linear_separatrix':
        fig = plt.figure(figsize=(10 + 4, 10 * ratio + 4.75))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(left=0.19, bottom=0.3, top=0.9, right = 0.9)
        ax.set(xlim=(x_min, x_max), xticks=[-2, 0, 2],
            ylim=(y_min, y_max), yticks=[-3, 0, 3])
        
    # Set axis labels and font size for axis ticks
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    ax.set_xlabel('x', fontsize = 45, fontfamily = 'serif')
    ax.set_ylabel('y', fontsize = 45, fontfamily = 'serif')

    # Set linewidth depending on number of cubes
    if len(cube_list_for_polytope_figure) > 100:
        linewidth = 3
    else:
        linewidth = 5

    # Produce randomly ordered list of colors
        
    #colors = ['#440154FF', '#471164FF', '#472B7AFF', '#424086FF', '#39568CFF', '#31688EFF', '#287C8EFF', '#238A8DFF', '#1F968BFF', '#20A387FF', '#2DB27DFF', '#4AC16DFF', '#73D055FF', '#A0DA39FF', '#DCE319FF', '#FDE725FF', '#5EC962FF', '#85D24BFF', '#B8DE29FF', '#FDE725FF', '#F7E225FF', '#F4DF24FF', '#F1DC24FF', '#EFD924FF', '#EDB424FF']
    #colors = ['#440154FF', '#471164FF', '#472B7AFF', '#424086FF', '#39568CFF', '#31688EFF', '#287C8EFF', '#238A8DFF', '#1F968BFF', '#20A387FF', '#2DB27DFF', '#4AC16DFF', '#73D055FF', '#A0DA39FF', '#DCE319FF', '#FDE725FF', '#E89AC7FF', '#D08BB1FF', '#B97C9AFF', '#A26D84FF', '#895D6EFF', '#6F4D58FF', '#563C42FF', '#3D2B2BFF', '#261A15FF']
    colors = ['#440154FF', '#471164FF', '#472B7AFF', '#424086FF', '#39568CFF', '#31688EFF', '#287C8EFF', '#238A8DFF', '#1F968BFF', '#20A387FF', '#2DB27DFF', '#4AC16DFF', '#73D055FF', '#A0DA39FF', '#5EC962FF', '#85D24BFF', '#B8DE29FF', '#F7E225FF', '#EFD924FF', '#EDB424FF', '#DCE319FF', '#238A8DFF', '#2DB27DFF']
    #colors = ['#440154FF', '#3B0A45FF', '#6A1D87FF', '#8A2D86FF', '#B43F6CFF', '#D64D3EFF', '#F26B3CFF', '#F8993FFF', '#F5A846FF', '#F7C03FFF', '#F2E126FF', '#F4D03FFF', '#F7B41DFF', '#F3A22EFF', '#F5B0D0FF', '#B084F4FF', '#6D27A3FF', '#6C8E9EFF', '#4B9AB9FF', '#3975A9FF', '#1F3C5BFF', '#0F6E9FFF', '#0047ABFF', '#002C77FF', '#003D5CFF']

    random.shuffle(colors)

    # Plot cubes with colors
    for i, cube in enumerate(cube_list_for_polytope_figure):
        x = []
        y = []
        for vertex in cube.vertex_list:
            x.append(vertex[0])
            y.append(vertex[1])
        j = i % len(colors)
        color = colors[j]

        ax.fill(x, y, facecolor=color, edgecolor='white', linewidth=linewidth)

    # Save as png and svg
    plt.savefig(file_name + '.png')
    plt.savefig(file_name + '.svg')

    # Show
    if show:
        plt.show()
    
    # Close
    plt.close(fig)

''' Produce plot of labeled polytopes from cubical decomposition '''
def plot_polytopes(config, cube_list_for_polytope_figure, show, file_name, system):

    if system == 'linear_separatrix':
        font_size = 30
    else:
        font_size = 28
    # Set font properties
    font = {'family' : 'serif',
            'size'   : font_size}
    plt.rc('font', **font)

    # Set layout
    x_min, x_max, y_min, y_max = get_x_y_min_max(config)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    ratio = width/height
    plt.style.use('_mpl-gallery-nogrid')
    if system == 'ellipsoidal_2d':
        fig = plt.figure(figsize = (7.25, 8.5))
        #For making plot without legend
        #fig = plt.figure(figsize = (6, 6))
    elif system == 'radial_bistable':
        fig = plt.figure(figsize = (7.25, 8.5))
    elif system == 'radial_tristable':
        fig = plt.figure(figsize = (7.25, 8.5))
    elif system == 'linear_separatrix':
        fig = plt.figure(figsize = (8, 7))
    else:
        fig1 = plt.figure(figsize = (7, 7))
        #fig = plt.figure(figsize=(10 + 2, 10 * ratio + 4.75))
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    if system != 'linear_separatrix':
        plt.subplots_adjust(left=0.19, bottom=0.3, top=0.9, right = 0.9)
        # For making the ellipsoidal figure with no legend
        #plt.subplots_adjust(left=0.19, bottom=0.15, top=0.9, right = 0.9)
    elif system == 'linear_separatrix':
        plt.subplots_adjust(left=0.19, bottom=0.3, top=0.9, right = 0.9)
    step = 2

    # Set axes
    if system == 'linear_separatrix':
        ax.set(xlim=(x_min, x_max), xticks=[-2, 0, 2],
            ylim=(y_min, y_max), yticks=[-3, 0, 3])
    elif system == 'ellipsoidal_2d':
        ax.set(xlim=(x_min, x_max), xticks=[-4, -2, 0, 2, 4],
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    elif system == 'radial_bistable':
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    elif system == 'radial_tristable':
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    else:
        ax.set(xlim=(x_min, x_max), xticks=np.arange(x_min + 1, x_max, step=step),
            ylim=(y_min, y_max), yticks=np.arange(y_min + 1, y_max, step=step))
    
    plt.xticks()
    plt.yticks()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Set line width of polytope boundaries 
    if len(cube_list_for_polytope_figure) > 100:
        linewidth = 3
    else:
        linewidth = 5

    # Plot polytopes with color and hatch style according to label 
    for cube in cube_list_for_polytope_figure:
        x = []
        y = []
        for vertex in cube.vertex_list:
            x.append(vertex[0])
            y.append(vertex[1])
        label = cube.label
        color, hatch = label_to_polytope_color_and_hatch(config, label)

        ax.fill(x, y, facecolor=color, edgecolor='white', linewidth=linewidth, hatch=hatch)

    # Update font style 
    font_properties = {
        'family': 'serif',
        'size': 28,
        'style': 'italic'
    }

    #Create legend 
    if config.num_labels == 2:
        color_0, hatch_0 = label_to_polytope_color_and_hatch(config, 0)
        circ0 = mpatches.Patch(facecolor=color_0,hatch=hatch_0,edgecolor='white',label='N\u2080')
        color_1, hatch_1 = label_to_polytope_color_and_hatch(config, 1)
        circ1 = mpatches.Patch(facecolor=color_1,hatch=hatch_1,edgecolor='white',label='N\u2081')
        color_u, hatch_u = label_to_polytope_color_and_hatch(config, 2)
        circu = mpatches.Patch(facecolor=color_u,hatch=hatch_u,edgecolor='white',label='U')
        if system == 'radial_tristable':
            num_columns = 2
        elif system == 'linear_separatrix':
            num_columns = 3
        else:
            num_columns = 3
        if system != 'linear_separatrix':# 0.545, -0.01
            ax.legend(handles = [circ0, circu, circ1], loc='lower center', bbox_to_anchor=(0.545, 0.02), prop=font_properties,
            bbox_transform=fig.transFigure, fancybox=True, facecolor='white', framealpha=1, ncol=num_columns)
        elif system == 'linear_separatrix':
            ax.legend(handles = [circ0, circu, circ1], loc='lower center', bbox_to_anchor=(0.545, 0.02), prop=font_properties,
        bbox_transform=fig.transFigure, fancybox=True, facecolor='white', framealpha=1, ncol=num_columns)
        
    if config.num_labels == 3:
        color_0, hatch_0 = label_to_polytope_color_and_hatch(config, 0)
        circ0 = mpatches.Patch(facecolor=color_0,hatch=hatch_0,edgecolor='white',label='N\u2080')
        color_1, hatch_1 = label_to_polytope_color_and_hatch(config, 1)
        circ1 = mpatches.Patch(facecolor=color_1,hatch=hatch_1,edgecolor='white',label='N\u2081')
        color_2, hatch_2 = label_to_polytope_color_and_hatch(config, 2)
        circ2 = mpatches.Patch(facecolor=color_2,hatch=hatch_2,edgecolor='white',label='N\u2082')
        color_u, hatch_u = label_to_polytope_color_and_hatch(config, 3)
        circu = mpatches.Patch(facecolor=color_u,hatch=hatch_u,edgecolor='white',label='U')
        ax.legend(handles = [circ0, circ1, circ2, circu], loc='lower center', bbox_to_anchor=(0.545, 0.0), prop=font_properties,
        bbox_transform=fig.transFigure, fancybox=True, facecolor='white', framealpha=1, ncol=2)
        
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
        ax.legend(handles = [circ0, circ1, circ2, circ3, circu], loc='lower center', bbox_to_anchor=(0.5, 0.06),
        bbox_transform=fig.transFigure, fancybox=True, facecolor='white', framealpha=1, ncol = 5)

    # Save 
    plt.savefig(file_name + '.png')
    plt.savefig(file_name + '.svg')

    # Show
    if show:
        plt.show()

    # Close 
    plt.close(fig)




