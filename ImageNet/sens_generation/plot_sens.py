import numpy as np
import matplotlib.pyplot as plt
# from utils import plot_config
# from results_fromFiles import *

# plt.style.use('ggplot')
# plt.style.use('bmh')

# def plot_config():
#     plt.figure()
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

    # return plt

def normalize(arr, factor):
    if factor == 0:
        factor = np.max(arr)
    arr = arr/factor
    return arr, factor


plot_data = lambda x: np.load(x)
channel_colors = {"Y": 'k', "Cb" : 'b', "Cr": 'r', 'CbCr': 'b'}
# for model in [ 'vgg13', 'mobilevitv2_050', 'vit_tiny']:
# for model in ['DenseNet121', 'Regnet400mf', 'Mnasnet','Squeezenet', 'ConvNeXt_tiny', 'mobilenet_v2']:
for model in ['Shufflenetv2_05']:
    # plt = plot_config()
    plt.figure()
    for channel in ["Y", "Cb", "Cr"]:
        try:
            plt.plot(np.arange(1, 65), plot_data("./SenMap/{}_{}.npy".format(channel, model)), channel_colors[channel], label="{} channel".format(channel))   
        except:
            plt.plot(np.arange(1, 65), plot_data("./SenMap/{}{}.npy".format(channel, model)), channel_colors[channel], label="{} channel".format(channel))   
        # plt.plot(np.arange(1, 65), plot_data("../DCT_coefficient/SenMap/{}{}.npy".format(channel, model)), channel_colors[channel], label="{} channel".format(channel))    
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
    plt.savefig("./plots/{}.pdf".format(model), dpi=900)
    print(model)


    plt.figure()

    q_max = 5
    try:
        Y_sens  = plot_data("./SenMap/{}_{}.npy".format("Y", model))
    except:
        Y_sens  = plot_data("./SenMap/{}{}.npy".format("Y", model))

    try:
        Cb_sens  = plot_data("./SenMap/{}_{}.npy".format("Cb", model))
    except:
        Cb_sens  = plot_data("./SenMap/{}{}.npy".format("Cb", model))

    try:
        Cr_sens  = plot_data("./SenMap/{}_{}.npy".format("Cr", model))
    except:
        Cr_sens  = plot_data("./SenMap/{}{}.npy".format("Cr", model))


    Y_sens               =  1 / Y_sens 
    CbCr_sens            =  2 / (Cb_sens + Cr_sens)
    _    , factor        = normalize(Y_sens   , 0)
    factor               = factor / q_max
    Y_sens    , _        = normalize(Y_sens   , factor)
    CbCr_sens , _        = normalize(CbCr_sens, factor)
    # min_Y = min(Y_sens)
    # max_Y = max(Y_sens)

    # Y_sens = ((Y_sens - min_Y)/(max_Y-min_Y)) * (q_max - 1) + 1
    # CbCr_sens = ((CbCr_sens - min_Y)/(max_Y-min_Y)) * (q_max - 1) + 1

    # print(min(Y_sens), max(Y_sens))

    plt.plot(np.arange(1, 65), Y_sens, channel_colors['Y'], label="{} channel".format('Y'))  
    plt.plot(np.arange(1, 65), CbCr_sens, channel_colors['CbCr'], label="{} channel".format('CbCr')) 
    plt.hlines(y=1, xmin=1, xmax=64, linewidth=1, color='r')  

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
