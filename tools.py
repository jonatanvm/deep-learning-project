import numpy as np
import pandas as pd
import torch

classes = pd.read_csv('wnids.txt', header=None).values
words = pd.read_csv('words.txt', sep="\t", header=None, names=['class', 'words'])
words = words.set_index('class')


def save_net(net, filename):
    try:
        torch.save(net.state_dict(), filename)
        print('Model saved to %s' % filename)
    except:
        raise Exception('The notebook should be run or validated with skip_training=True.')


def show_predictions(net, device, testloader):
    # Let us display random images from the test set, the ground truth labels and the network's predictions
    net.eval()
    with torch.no_grad():
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        # Compute predictions
        images = images.to(device)
        y = net(images)

    truth = np.array([words.loc[classes[labels[j]][0]].values[0] for j in range(5)])
    pred = np.array([words.loc[classes[j][0]].values[0] for j in y.argmax(dim=1)])
    for i in range(len(truth)):
        print("Prediction %s:\nTurth: %s\nPred: %s\n" % (i + 1, truth[i], pred[i]))
