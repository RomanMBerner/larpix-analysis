from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn

def plot_h1(input_list,x_min,x_max,n_bins,axis_labels,save_name):
    seaborn.set(rc={'figure.figsize':(15, 10),})
    seaborn.set_context('talk') # or paper

    # Define parameters of the frame
    fig = plt.figure() # plt.figure(figsize=(width,height))
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    ax.spines['bottom'].set_color('0.5') #'black', ...
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_color('0.5')
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_color('0.5')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_color('0.5')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_visible(True)

    # Ticks, grid and ticks labels
    ax.tick_params(direction='in', length=10, width=2,                  # direction, length and width of the ticks (in, out, inout)
                   colors='0.5',                                       # color of the ticks ('black', '0.5')
                   bottom=True, top=True, right=True, left=True,       # whether to draw the respective ticks
                   zorder = 10.,                                       # tick and label zorder
                   pad = 10.,                                          # distance between ticks and tick labels
                   labelsize = 17,                                     # size of the tick labels
                   labelright=False, labeltop=False)                   # wether to draw the tick labels on axes

    n, bins, patches = plt.hist(input_list,\
                                bins=n_bins,\
                                range=[x_min,x_max],\
                                histtype='stepfilled',\
                                stacked=False,\
                                linewidth=3,\
                                alpha=0.5) # histtype='step' or 'stepfilled', label='track_length'

    # Legend
    #plt.legend(loc=[0.75,0.85], prop={'size': 17}) # loc='upper right'

    # Axis labels
    plt.xlabel(axis_labels[0], fontsize=20, labelpad=20)
    plt.ylabel(axis_labels[1], fontsize=20, labelpad=20)

    # Logarithmic y axis
    #plt.ylim(bottom=0.9) #, top=200)
    #plt.yscale('linear') # linear, log

    # Save figure
    plt.savefig(save_name, dpi=400) # bbox_inches='tight'
    plt.close()
    

def plot_h2(input_lists,x_bins,y_bins,axis_labels,save_name):
    # Size
    seaborn.set(rc={'figure.figsize':(12.3, 10),})

    fig, ax = plt.subplots()

    # Define parameters of the frame
    #fig = plt.figure() # plt.figure(figsize=(width,height))
    #fig.patch.set_facecolor('white')
    #fig.patch.set_alpha(0.0)
    #ax = fig.add_subplot(111)
    #ax.patch.set_facecolor('#ababab') # #ababab
    ax.patch.set_alpha(0.0)
    ax.spines['bottom'].set_color('0.5') #'black', ...
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_color('0.5')
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_color('0.5')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_color('0.5')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_visible(True)

    # Ticks, grid and ticks labels
    ax.tick_params(direction='in', length=10, width=2,                  # direction, length and width of the ticks (in, out, inout)
                   colors='0.5',                                       # color of the ticks ('black', '0.5')
                   bottom=True, top=True, right=True, left=True,       # whether to draw the respective ticks
                   zorder = 10.,                                       # tick and label zorder
                   pad = 10.,                                          # distance between ticks and tick labels
                   labelsize = 17,                                     # size of the tick labels
                   labelright=False, labeltop=False)                   # wether to draw the tick labels on axes
                   #labelrotation=45.                                  # rotation of the labels
                   #grid_color='black',                                # grid
                   #grid_alpha=0.0,
                   #grid_linewidth=1.0,
    # colors='black','0.5'

    plt.hist2d(input_lists[0], input_lists[1], bins=[x_bins,y_bins])#, cmap=plt.cm.viridis), weights=weights_list
    # Color maps: viridis, plasma, magma, inferno

    '''x = np.linspace(-0.5,10.5)
    y = np.linspace(-0.5,10.5)
    plt.plot(x, y, '--', color="b")'''

    # Colorbar
    #help(colorbar)
    colorbar = plt.colorbar()
    colorbar.set_label(axis_labels[2], rotation=270, fontsize=20)
    colorbar.ax.tick_params(labelsize=20)
    #v1 = np.linspace(z.min(), z.max(), 8, endpoint=True)
    #plt.colorbar(ticks=v1)

    # Axes
    plt.xlabel(axis_labels[0], fontsize=20, labelpad=20)
    plt.ylabel(axis_labels[1], fontsize=20, labelpad=20)
    plt.tick_params(labelsize=20)

    # Change label offset of axes
    from matplotlib import rcParams
    rcParams['axes.labelpad'] = 35

    #ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

    # Save figure
    fig_name = 'h2_trackLength_vs_trackNHits.png'
    plt.savefig(save_name, dpi=400) # bbox_inches='tight'
    plt.close()
    
def plot_h3_unweighted(input_lists,x_bins,y_bins,z_bins,axis_labels,save_name):
    # Size
    #seaborn.set(rc={'figure.figsize':(12.3, 10),})

    #fig, ax = plt.subplots(111, projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define parameters of the frame
    #fig = plt.figure() # plt.figure(figsize=(width,height))
    #fig.patch.set_facecolor('white')
    #fig.patch.set_alpha(0.0)
    #ax = fig.add_subplot(111)
    #ax.patch.set_facecolor('#ababab') # #ababab
    ax.patch.set_alpha(0.0)
    ax.spines['bottom'].set_color('0.5') #'black', ...
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_color('0.5')
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_color('0.5')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_color('0.5')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_visible(True)

    # Ticks, grid and ticks labels
    ax.tick_params(direction='in', length=10, width=2,                  # direction, length and width of the ticks (in, out, inout)
                   colors='0.5',                                       # color of the ticks ('black', '0.5')
                   bottom=True, top=True, right=True, left=True,       # whether to draw the respective ticks
                   zorder = 10.,                                       # tick and label zorder
                   pad = 10.,                                          # distance between ticks and tick labels
                   labelsize = 17,                                     # size of the tick labels
                   labelright=False, labeltop=False)                   # wether to draw the tick labels on axes
                   #labelrotation=45.                                  # rotation of the labels
                   #grid_color='black',                                # grid
                   #grid_alpha=0.0,
                   #grid_linewidth=1.0,
    # colors='black','0.5'

    ax.set_xlim((x_bins[0],x_bins[-1]))
    ax.set_ylim((y_bins[0],y_bins[-1]))
    ax.set_zlim((z_bins[0],z_bins[-1]))
    
    ax.set_xlabel(axis_labels[0], fontsize=20, labelpad=20)
    ax.set_ylabel(axis_labels[1], fontsize=20, labelpad=20)
    ax.set_zlabel(axis_labels[2], fontsize=20, labelpad=20)

    img = ax.scatter(input_lists[0], input_lists[1], input_lists[2]) #, c=input_lists[3]) #, cmap=plt.hot())#, cmap=plt.cm.viridis)

    # Colorbar
    #help(colorbar)
    #cbar = plt.colorbar(img, fraction=0.03, pad=0.07) # fraction: height; pad: distance of label to the color bar
    #cbar.set_label(axis_labels[3], rotation=90, fontsize=20)
    #colorbar = fig.colorbar(img,fraction=0.046, pad=0.04)
    #colorbar = plt.colorbar()
    #colorbar.set_label(axis_labels[3], rotation=270, fontsize=20)
    #colorbar.tick_params(labelsize=40)
    #v1 = np.linspace(min(input_lists[3]), max(input_lists[3]), 8, endpoint=True)
    #plt.colorbar(ticks=v1)

    # Axes
    #plt.xlabel(axis_labels[0], fontsize=20, labelpad=20)
    #plt.ylabel(axis_labels[1], fontsize=20, labelpad=20)
    #plt.tick_params(labelsize=20)

    # Change label offset of axes
    from matplotlib import rcParams
    rcParams['axes.labelpad'] = 35

    #ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

    # Save figure
    plt.savefig(save_name, dpi=400) # bbox_inches='tight'
    plt.close()
    
    
def plot_profile_of_2D_hist(gc_bin_i_list,ly_mean_bin_i_list,x_err,ly_std_bin_i_list,plot_folder_list,axis_labels,x_min,x_max,y_min,y_max,save_name):
    # Size
    seaborn.set(rc={'figure.figsize':(12.3, 10),})
    seaborn.set_context('talk') # or paper

    # Define parameters of the frame
    fig = plt.figure() # plt.figure(figsize=(width,height))
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    ax.spines['bottom'].set_color('0.5') #'black', ...
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_color('0.5')
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_color('0.5')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_color('0.5')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_visible(True)

    # Ticks, grid and ticks labels
    ax.tick_params(direction='in', length=10, width=2,                  # direction, length and width of the ticks (in, out, inout)
                   colors='0.5',                                       # color of the ticks ('black', '0.5')
                   bottom=True, top=True, right=True, left=True,       # whether to draw the respective ticks
                   zorder = 10.,                                       # tick and label zorder
                   pad = 10.,                                          # distance between ticks and tick labels
                   labelsize = 17,                                     # size of the tick labels
                   labelright=False, labeltop=False)                   # wether to draw the tick labels on axes
    # Axis limits
    ax.set_xlim((x_min,x_max))
    ax.set_ylim((y_min,y_max))

    # Axis labels
    plt.xlabel(axis_labels[0], fontsize=20, labelpad=20)
    plt.ylabel(axis_labels[1], fontsize=20, labelpad=20)

    # TODO: Select representatives!
    # Don't plot all 10 series but restrict to a few only...
    restrict_to_entries_1 = [1,5,9]
    #restrict_to_entries_2 = [0,4,8]
    #restrict_to_entries_3 = [2,3,6,7]

    for i in range(10):
        if i not in restrict_to_entries_1:
            continue
        plt.errorbar(gc_bin_i_list[i], ly_mean_bin_i_list[i], xerr=x_err, yerr=ly_std_bin_i_list[i], fmt='o', label=plot_folder_list[i][0:3]+' kV/cm') # fmt='-o'
    
    # Legend
    plt.legend(loc=[0.75,0.8], prop={'size': 17}) # loc='upper right'
    
    # Save figure
    plt.savefig(save_name, dpi=400) # bbox_inches='tight'
    plt.close()
    

def plot_event_2D(input_lists,x_bins,y_bins,axis_labels,save_name):
    # Size
    seaborn.set(rc={'figure.figsize':(12.3, 10),})

    fig, ax = plt.subplots()

    # Define parameters of the frame
    #fig = plt.figure() # plt.figure(figsize=(width,height))
    #fig.patch.set_facecolor('white')
    #fig.patch.set_alpha(0.0)
    #ax = fig.add_subplot(111)
    #ax.patch.set_facecolor('#ababab') # #ababab
    ax.patch.set_alpha(0.0)
    ax.spines['bottom'].set_color('0.5') #'black', ...
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_color('0.5')
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_color('0.5')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_color('0.5')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_visible(True)

    # Ticks, grid and ticks labels
    ax.tick_params(direction='in', length=10, width=2,                  # direction, length and width of the ticks (in, out, inout)
                   colors='0.5',                                       # color of the ticks ('black', '0.5')
                   bottom=True, top=True, right=True, left=True,       # whether to draw the respective ticks
                   zorder = 10.,                                       # tick and label zorder
                   pad = 10.,                                          # distance between ticks and tick labels
                   labelsize = 17,                                     # size of the tick labels
                   labelright=False, labeltop=False)                   # wether to draw the tick labels on axes
                   #labelrotation=45.                                  # rotation of the labels
                   #grid_color='black',                                # grid
                   #grid_alpha=0.0,
                   #grid_linewidth=1.0,
    # colors='black','0.5'

    plt.hist2d(input_lists[0], input_lists[1], weights=input_lists[2], bins=[x_bins,y_bins]) #, cmap=plt.hot())#, cmap=plt.cm.viridis)

    # Colorbar
    #help(colorbar)
    colorbar = plt.colorbar()
    colorbar.set_label(axis_labels[2], rotation=270, fontsize=20)
    colorbar.ax.tick_params(labelsize=20)
    #v1 = np.linspace(z.min(), z.max(), 8, endpoint=True)
    #plt.colorbar(ticks=v1)

    # Axes
    plt.xlabel(axis_labels[0], fontsize=20, labelpad=20)
    plt.ylabel(axis_labels[1], fontsize=20, labelpad=20)
    plt.tick_params(labelsize=20)

    # Change label offset of axes
    from matplotlib import rcParams
    rcParams['axes.labelpad'] = 35

    #ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

    # Save figure
    plt.savefig(save_name, dpi=400) # bbox_inches='tight'
    plt.close()
    

def plot_event_3D(input_lists,x_bins,y_bins,z_bins,axis_labels,save_name):
    # Size
    #seaborn.set(rc={'figure.figsize':(12.3, 10),})

    #fig, ax = plt.subplots(111, projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define parameters of the frame
    #fig = plt.figure() # plt.figure(figsize=(width,height))
    #fig.patch.set_facecolor('white')
    #fig.patch.set_alpha(0.0)
    #ax = fig.add_subplot(111)
    #ax.patch.set_facecolor('#ababab') # #ababab
    ax.patch.set_alpha(0.0)
    ax.spines['bottom'].set_color('0.5') #'black', ...
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_color('0.5')
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_color('0.5')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_color('0.5')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_visible(True)

    # Ticks, grid and ticks labels
    ax.tick_params(direction='in', length=10, width=2,                  # direction, length and width of the ticks (in, out, inout)
                   colors='0.5',                                       # color of the ticks ('black', '0.5')
                   bottom=True, top=True, right=True, left=True,       # whether to draw the respective ticks
                   zorder = 10.,                                       # tick and label zorder
                   pad = 10.,                                          # distance between ticks and tick labels
                   labelsize = 17,                                     # size of the tick labels
                   labelright=False, labeltop=False)                   # wether to draw the tick labels on axes
                   #labelrotation=45.                                  # rotation of the labels
                   #grid_color='black',                                # grid
                   #grid_alpha=0.0,
                   #grid_linewidth=1.0,
    # colors='black','0.5'

    ax.set_xlim((x_bins[0],x_bins[-1]))
    ax.set_ylim((y_bins[0],y_bins[-1]))
    ax.set_zlim((z_bins[0],z_bins[-1]))
    
    ax.set_xlabel(axis_labels[0], fontsize=20, labelpad=20)
    ax.set_ylabel(axis_labels[1], fontsize=20, labelpad=20)
    ax.set_zlabel(axis_labels[2], fontsize=20, labelpad=20)

    img = ax.scatter(input_lists[0], input_lists[1], input_lists[2], c=input_lists[3]) #, cmap=plt.hot())#, cmap=plt.cm.viridis)

    # Colorbar
    #help(colorbar)
    cbar = plt.colorbar(img, fraction=0.03, pad=0.07) # fraction: height; pad: distance of label to the color bar
    cbar.set_label(axis_labels[3], rotation=90, fontsize=20)
    #colorbar = fig.colorbar(img,fraction=0.046, pad=0.04)
    #colorbar = plt.colorbar()
    #colorbar.set_label(axis_labels[3], rotation=270, fontsize=20)
    #colorbar.tick_params(labelsize=40)
    #v1 = np.linspace(min(input_lists[3]), max(input_lists[3]), 8, endpoint=True)
    #plt.colorbar(ticks=v1)

    # Axes
    #plt.xlabel(axis_labels[0], fontsize=20, labelpad=20)
    #plt.ylabel(axis_labels[1], fontsize=20, labelpad=20)
    #plt.tick_params(labelsize=20)

    # Change label offset of axes
    from matplotlib import rcParams
    rcParams['axes.labelpad'] = 35

    #ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

    # Save figure
    plt.savefig(save_name, dpi=400) # bbox_inches='tight'
    plt.close()
    

def plot_LY_vs_GCx_selection(input_lists,x_bins,y_bins,axis_labels,save_name,cut_at_xmin,cut_at_xmax):
    # Size
    seaborn.set(rc={'figure.figsize':(12.3, 10),})
    seaborn.set_context('talk') # or paper

    fig, ax = plt.subplots()

    # Define parameters of the frame
    #fig = plt.figure() # plt.figure(figsize=(width,height))
    #fig.patch.set_facecolor('white')
    #fig.patch.set_alpha(0.0)
    #ax = fig.add_subplot(111)
    #ax.patch.set_facecolor('#ababab') # #ababab
    ax.patch.set_alpha(0.0)
    ax.spines['bottom'].set_color('0.5') #'black', ...
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_color('0.5')
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_color('0.5')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_color('0.5')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_visible(True)

    # Ticks, grid and ticks labels
    ax.tick_params(direction='in', length=10, width=2,                  # direction, length and width of the ticks (in, out, inout)
                   colors='0.5',                                       # color of the ticks ('black', '0.5')
                   bottom=True, top=True, right=True, left=True,       # whether to draw the respective ticks
                   zorder = 10.,                                       # tick and label zorder
                   pad = 10.,                                          # distance between ticks and tick labels
                   labelsize = 17,                                     # size of the tick labels
                   labelright=False, labeltop=False)                   # wether to draw the tick labels on axes
                   #labelrotation=45.                                  # rotation of the labels
                   #grid_color='black',                                # grid
                   #grid_alpha=0.0,
                   #grid_linewidth=1.0,
    # colors='black','0.5'

    plt.hist2d(input_lists[0], input_lists[1], bins=[x_bins,y_bins])#, cmap=plt.cm.viridis), weights=weights_list
    # Color maps: viridis, plasma, magma, inferno

    '''x = np.linspace(-0.5,10.5)
    y = np.linspace(-0.5,10.5)
    plt.plot(x, y, '--', color="b")'''

    # Colorbar
    #help(colorbar)
    colorbar = plt.colorbar()
    colorbar.set_label(axis_labels[2], rotation=270, fontsize=20)
    colorbar.ax.tick_params(labelsize=20)
    #v1 = np.linspace(z.min(), z.max(), 8, endpoint=True)
    #plt.colorbar(ticks=v1)

    # Axes
    plt.xlabel(axis_labels[0], fontsize=20, labelpad=20)
    plt.ylabel(axis_labels[1], fontsize=20, labelpad=20)
    plt.tick_params(labelsize=20)
    
    plt.plot([-155, 155],[cut_at_xmin,cut_at_xmax])

    # Change label offset of axes
    from matplotlib import rcParams
    rcParams['axes.labelpad'] = 35

    #ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

    # Save figure
    fig_name = 'h2_trackLength_vs_trackNHits.png'
    plt.savefig(save_name, dpi=400) # bbox_inches='tight'
    plt.close()