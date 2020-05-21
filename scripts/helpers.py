import torch
from scripts import dlc_practical_prologue as prologue
import numpy as np
import matplotlib.pyplot as plt


def encode_to_one_hot(target):
    n = target.size(0)
    result = torch.zeros((n, 2))
    return result.scatter(1, target.reshape(n, 1).long(), 1)


def normalize(input, mean, std):
    input.sub_(mean).div_(std)


def divide_pairs(img_input, classes, one_hot_classes=False):
    """ Divides the pairs into two distinct datasets """
    n_img = img_input.size(0)
    img_input_1 = img_input[:, 0, :, :].reshape(n_img, 1, 14, 14)
    img_input_2 = img_input[:, 1, :, :].reshape(n_img, 1, 14, 14)

    img_classes_1 = prologue.convert_to_one_hot_labels(img_input_1, classes[:, 0]) if one_hot_classes else classes[:, 0]
    img_classes_2 = prologue.convert_to_one_hot_labels(img_input_2, classes[:, 1]) if one_hot_classes else classes[:, 1]

    img_classes_1.reshape(-1, 1)
    img_classes_2.reshape(-1, 1)

    return img_input_1, img_input_2, img_classes_1, img_classes_2


def shuffle_data(tr_input_1, tr_input_2, tr_classes_1, tr_classes_2, tr_targets):
    """Shuffles the data randomly """
    N = tr_targets.size(0)
    idx = torch.randperm(N)

    new_targets = tr_targets[idx]
    new_tr_input_1 = tr_input_1[idx]
    new_tr_input_2 = tr_input_2[idx]
    new_tr_classes_1 = tr_classes_1[idx]
    new_tr_classes_2 = tr_classes_2[idx]

    return new_tr_input_1, new_tr_input_2, new_tr_classes_1, new_tr_classes_2, new_targets


def compute_nb_errors(targets, offset, predicted, mini_batch_size):
    errors = 0
    # Count the number of errors
    for k in range(mini_batch_size):
        if not torch.equal(torch.eq(targets[offset + k], predicted[k]), torch.tensor([True, True])):
            errors += 1
    return errors


def gen_results_plot(errors, errors_naive, errors_noWS, errors_naive_noWS):
    """Generate results plot and save it in root folder as results.pdf"""
    mean, std = errors.mean(), errors.std()
    mean_naive, std_naive = errors_naive.mean(), errors_naive.std()
    mean_noWS, std_noWS = errors_noWS.mean(), errors_noWS.std()
    mean_naive_noWS, std_naive_noWS = errors_naive_noWS.mean(), errors_naive_noWS.std()

    model_names = ['Naive', 'Naive with WS', 'Enhanced', 'Enhanced with WS']
    x_pos = np.arange(len(model_names))
    means = [1 - mean_naive_noWS, 1 - mean_naive, 1 - mean_noWS, 1 - mean]
    stds = [std_naive_noWS, std_naive, std_noWS, std]

    # Build the plot
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 8)
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_title('Accuracy test of the models')
    ax.yaxis.grid(True)

    for i, m in enumerate(means):
        ax.text(i - 0.05, 0.65, "{:0.3f}".format(m))

    fig.savefig('results.pdf', format='pdf', orientation='landscape')
