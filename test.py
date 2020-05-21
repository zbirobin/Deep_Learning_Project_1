from scripts.helpers import *
from scripts.NaiveSiamese import NaiveSiamese
from scripts.EnhancedSiamese import EnhancedSiamese

"""General parameters used for training and testing the model"""
N = 1000  # Number of data samples in training and test set
verbose = True  # for debug printing
nb_rounds = 10  # Number of training rounds
nb_epochs = 75  # Number of training epochs
lr = 0.5 * 1e-1  # learning rate of the gradient descent algorithm
nb_hidden = 500  # Size of the hidden layer used in DigitNet

"""        **** Data loading and pre-processing ****       """
# load data from source
train_input, train_target, train_classes, \
test_input, test_target, test_classes = prologue.generate_pair_sets(N)

# use 1-hot encoding for targets
train_target = encode_to_one_hot(train_target)
test_target = encode_to_one_hot(test_target)

mean = train_input.mean(dim=(0, 2, 3), keepdim=True)
std = train_input.std(dim=(0, 2, 3), keepdim=True)

# Normalize data by removing mean and subtracting by the std
normalize(train_input, mean, std)
normalize(test_input, mean, std)

# Divides the pairs into two distinct datasets
train_input_left, train_input_right, train_classes_left, train_classes_right = divide_pairs(train_input, train_classes)
test_input_left, test_input_right, test_classes_left, test_classes_right = divide_pairs(test_input, test_classes)

"""         **** Train & Test procedure ****        """
# Containers for error rates ( 0 < error < 1 )
errors_naive_noWS = np.zeros(nb_rounds)
errors_noWS = np.zeros(nb_rounds)
errors = np.zeros(nb_rounds)
errors_naive = np.zeros(nb_rounds)

for i in range(0, nb_rounds):
    print('Round {}'.format(i))

    # Data shuffling
    train_input_left, train_input_right, train_classes_left, train_classes_right, train_target = shuffle_data(
        train_input_left,
        train_input_right,
        train_classes_left,
        train_classes_right,
        train_target)

    """Weights reinitialization"""
    # Naive siamese without weight sharing
    naive_siamese_noWS = NaiveSiamese(nb_hidden=nb_hidden, weight_sharing=False)
    # Enhanced siamese without weight sharing
    enhanced_siamese_noWS = EnhancedSiamese(nb_hidden=nb_hidden, weight_sharing=False)
    # Naive siamese with weight sharing
    naive_siamese = NaiveSiamese(nb_hidden=nb_hidden)
    # Enhanced siamese weight sharing
    enhanced_siamese = EnhancedSiamese(nb_hidden=nb_hidden)

    """Model variants training"""
    naive_siamese_noWS.train(train_input_left, train_input_right, train_target, nb_epochs=nb_epochs, lr=lr,
                             verbose=verbose)

    enhanced_siamese_noWS.train(train_input_left, train_input_right, train_classes_left, train_classes_right,
                                train_target,
                                nb_epochs=nb_epochs, lr=lr, verbose=verbose)

    naive_siamese.train(train_input_left, train_input_right, train_target, nb_epochs=nb_epochs, lr=lr, verbose=verbose)

    enhanced_siamese.train(train_input_left, train_input_right, train_classes_left, train_classes_right, train_target,
                           nb_epochs=nb_epochs, lr=lr, verbose=verbose)

    """Model variants testing"""
    errors_naive_noWS[i] = naive_siamese_noWS.test(test_input_left, test_input_right, test_target)
    errors_noWS[i] = enhanced_siamese_noWS.test(test_input_left, test_input_right, test_target)
    errors_naive[i] = naive_siamese.test(test_input_left, test_input_right, test_target)
    errors[i] = enhanced_siamese.test(test_input_left, test_input_right, test_target)

"""       **** Generate plot *****          """
# Generates results (see results.pdf in root)
gen_results_plot(errors, errors_naive, errors_noWS, errors_naive_noWS)

print("Please see results.pdf in root")
