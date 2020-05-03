import torch
import dlc_practical_prologue as prologue


def encode_to_one_hot(target):
    n = target.size(0)
    result = torch.zeros((n, 2))
    return result.scatter(1, target.reshape(n, 1).long(), 1)


def normalize(input, mean, std):
    input.sub_(mean).div_(std)


def process_data(img_input, classes, one_hot_classes=False):
    n_img = img_input.size(0)
    img_input_1 = img_input[:, 0, :, :].reshape(n_img, 1, 14, 14)
    img_input_2 = img_input[:, 1, :, :].reshape(n_img, 1, 14, 14)

    img_classes_1 = prologue.convert_to_one_hot_labels(img_input_1, classes[:, 0]) if one_hot_classes else classes[:, 0]
    img_classes_2 = prologue.convert_to_one_hot_labels(img_input_2, classes[:, 1]) if one_hot_classes else classes[:, 1]

    img_classes_1.reshape(-1, 1)
    img_classes_2.reshape(-1, 1)

    return img_input_1, img_input_2, img_classes_1, img_classes_2


def shuffle_data(tr_input_1, tr_input_2, tr_classes_1, tr_classes_2, tr_targets):
    N = tr_targets.size(0)
    idx = torch.randperm(N)

    new_targets = tr_targets[idx]
    new_tr_input_1 = tr_input_1[idx]
    new_tr_input_2 = tr_input_2[idx]
    new_tr_classes_1 = tr_classes_1[idx]
    new_tr_classes_2 = tr_classes_2[idx]

    return new_tr_input_1, new_tr_input_2, new_tr_classes_1, new_tr_classes_2, new_targets
