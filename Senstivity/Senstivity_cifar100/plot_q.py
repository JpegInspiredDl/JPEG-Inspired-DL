import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
font = 25
# Set the default text font size
# plt.rc('font', size=14, weight='bold')
plt.rc('figure', figsize = (8,6))
# plt.rc('figure', autolayout = True)
plt.rc('font', size=font+4)
# Set the axes title font size
plt.rc('axes', titlesize=font)
# Set the axes labels font size
plt.rc('axes', labelsize=font)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=font)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=font)
# Set the legend font size
plt.rc('legend', fontsize=15)
# Set the font size of the figure title
plt.rc('figure', titlesize=font)


def normalize(arr, factor):
    if factor == 0:
        factor = np.max(arr)
    arr = arr/factor
    return arr, factor


plot_data = lambda x: np.load(x)
channel_colors = {"Y": 'k', "Cb" : 'b', "Cr": 'r', 'CbCr': 'b'}
for model in [ 'resnet110']:
    plt.figure()
    for channel in ["Y", "Cb", "Cr"]:
        plt.plot(np.arange(1, 65), plot_data("./SenMap/{}_{}.npy".format(channel, model)), channel_colors[channel], label="{} channel".format(channel))   
    plt.ylabel("Sensitivity")
    plt.xlabel("Frequency Index")
    # Set the x-axis tick positions and labels
    tick_positions = np.arange(1, 65, 15)  # Tick positions every 5 units
    tick_labels = np.arange(1, 65, 15)  # Corresponding tick labels
    plt.xticks(tick_positions, tick_labels)
    plt.grid(True, linestyle='dashed', linewidth=1)  # Customize the grid lines
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.legend(loc='upper right')

    plt.tight_layout()
    # plt.figure.autolayout()
    plt.savefig("./plots/{}.png".format(model), dpi=900)
    print(model)


    plt.figure()

    q_max = 5
    Y_sens  = plot_data("./SenMap/{}_{}.npy".format("Y", model))
    Cb_sens = plot_data("./SenMap/{}_{}.npy".format("Cb", model))
    Cr_sens = plot_data("./SenMap/{}_{}.npy".format("Cr", model))

    Y_sens               = 1 / Y_sens 
    CbCr_sens            =  2 / (Cb_sens + Cr_sens)
    _    , factor        = normalize(Y_sens   , 0)
    factor               = factor / q_max
    Y_sens    , _        = normalize(Y_sens   , factor)
    CbCr_sens , _        = normalize(CbCr_sens, factor)


    plt.plot(np.arange(1, 65), Y_sens, channel_colors['Y'], label="{} channel".format('Y'))  
    plt.plot(np.arange(1, 65), CbCr_sens, channel_colors['CbCr'], label="{} channel".format('CbCr')) 

    plt.ylabel("Q Steps")
    plt.xlabel("Frequency Index")
    # Set the x-axis tick positions and labels
    tick_positions = np.arange(1, 65, 15)  # Tick positions every 5 units
    tick_labels = np.arange(1, 65, 15)  # Corresponding tick labels
    plt.xticks(tick_positions, tick_labels)
    plt.grid(True, linestyle='dashed', linewidth=1)  # Customize the grid lines
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.legend(loc='upper right')

    plt.tight_layout()
    # plt.figure.autolayout()
    plt.savefig("./plots/{}_Q_initial.png".format(model), dpi=900)

