import pandas as pd
import json
from bokeh.plotting import figure, output_notebook, show
import numpy as np
from bokeh.charts.utils import cycle_colors
import sweeper

import matplotlib.pyplot as plt


def generateLegendName(log_name):
    s = ''
    if log_name.find('resnet-pre-act') >= 0:
        s = s + 'resNet-'
    if log_name.find('wide-resnet') >= 0:
        s = s + 'WRN-'

    pd = log_name.find('_d')
    pw = log_name.find('w', pd+1)
    pdrop = log_name.find('_drop')
    if pw >= 0:
        s = s + log_name[pd+2:pw]
        if pdrop >= 0:
            s = s + '-' + log_name[pw+1:pdrop-1]
            s = s + '-' + log_name[pdrop+5:]
        else:
            s = s + '-' + log_name[pw+1:]
    else:
        s = s + log_name[pd+2:]
    
    return s

def generateLegendNames(log_names):
    return [generateLegendName(log) for log in log_names]

def plotLogs(log_names, suffix = ''):
    # parse log files, extracting json entry with stats per epoch and creating pandas DataFrame
    frames = [pd.DataFrame(sweeper.loadLog('../logs/'+log+'/log.txt')) for log in log_names]
    colors = ['red','blue','green','black','purple','orange','yellow']

    # this searches constant parameters across different runs to generate legends
    
    legends = generateLegendNames(log_names)
    
    # TODO: improve this, add hovers etc.
    p = plt.figure(figsize=(12, 6))
    ax = plt.subplot(121)

    plt.title('train error')
    plt.xlabel('epoch')
    plt.ylabel('train error')
    for i,frame in enumerate(frames):
        plt.plot(frame['epoch'], 100-frame['train_acc'], label=legends[i])

    ax.legend(loc='upper right', shadow=False)

    ax = plt.subplot(122)

    plt.title('test error')
    plt.xlabel('epoch')
    plt.ylabel('test error')
    for i,frame in enumerate(frames):
        plt.plot(frame['epoch'], 100-frame['test_acc'], label=legends[i])

    ax.legend(loc='upper right', shadow=False)
    #plt.show()
    plt.savefig('error_' + suffix + '.pdf') 


    p = plt.figure()
    ax = plt.subplot(111)

    plt.title('log loss')
    plt.xlabel('epoch')
    plt.ylabel('log loss')
    for i,frame in enumerate(frames):
        plt.plot(frame['epoch'], np.log(frame['loss']), label=legends[i])

    ax.legend(loc='upper right', shadow=False)

    #plt.show()
    plt.savefig('loss_' + suffix + '.pdf') 

    train_time = [sum(frame['train_time'])/3600 for i,frame in enumerate(frames)]
    x = np.arange(len(train_time))
    p = plt.figure()

    plt.title('training time')
    plt.ylabel('training time (hour)')
    plt.bar(x+0.25, train_time, 0.5)
    plt.xticks(x+0.5, legends)

    #plt.show()
    plt.savefig('trainTime_' + suffix + '.pdf') 

    n_parameters = [frame['n_parameters'][0] for i,frame in enumerate(frames)]
    x = np.arange(len(n_parameters))
    p = plt.figure()
    plt.bar(x+0.25, n_parameters, 0.5)
    plt.xticks(x+0.5, legends)

    plt.title('#parameter')
    plt.ylabel('#parameter')

    #plt.show()
    plt.savefig('nparameters_' + suffix + '.pdf') 



plotLogs([
    'wide-resnet_d40w1',
    'wide-resnet_d40w2',
    'wide-resnet_d40w4',
    'wide-resnet_d40w6',
    'wide-resnet_d40w8'
    ], 'width')

plotLogs([
    'resnet-pre-act_d11',
    'resnet-pre-act_d47',
    'resnet-pre-act_d164',
    'resnet-pre-act_d227'
    ], 'depth')

