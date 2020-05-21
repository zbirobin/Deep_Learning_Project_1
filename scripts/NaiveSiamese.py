import torch
from torch import nn
from scripts import helpers
from scripts.Nets import DigitNet, CompNet


class NaiveSiamese:
    """
    Implementation of the naive model ignoring the train classes having two variations
        1. With weight sharing : using a single instance of a digitNet module
        2. Without weight sharing : using two clones of a digitNet module
    """

    def __init__(self, nb_hidden, weight_sharing=True):
        if weight_sharing:
            model_digit = DigitNet(nb_hidden)
            self.model = CompNet(model_digit, weight_sharing=weight_sharing)
        else:
            model_digit_1 = DigitNet(nb_hidden)
            model_digit_2 = DigitNet(nb_hidden)
            self.model = CompNet(model_digit_1, model_digit_2, weight_sharing=weight_sharing)

    def train(self, train_input_1, train_input_2, train_target, mini_batch_size=25,
              nb_epochs=25, lr=1e-1, verbose=False):
        # SGD optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # Binary Cross Entropy Loss used for digit comparison
        criterion = nn.BCELoss()

        for e in range(nb_epochs):
            if verbose and e % 5 == 1:
                print("Epochs {}".format(e))
                print("loss = {}".format(loss))

            for b in range(0, train_input_1.size(0), mini_batch_size):
                # Take a mini batch from the image data
                train_input_1_ = train_input_1.narrow(0, b, mini_batch_size)
                train_input_2_ = train_input_2.narrow(0, b, mini_batch_size)

                # Forward pass for digit comparison using BCELoss
                output_comp = self.model(train_input_1_, train_input_2_)

                # Take a mini batch from the target data
                batch_target = train_target.narrow(0, b, mini_batch_size)

                # Measure the comparison results
                loss = criterion(output_comp, batch_target)

                # Apply the backward step
                self.model.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self, data_input_1, data_input_2, data_target, mini_batch_size=25):
        """Test method using the error rate as a metric """
        # Init the the number of errors
        nb_data_errors = 0
        # Number of samples
        N = data_input_1.size(0)

        for b in range(0, data_input_1.size(0), mini_batch_size):
            # Run the model on a mini batch of the images
            output = self.model(data_input_1.narrow(0, b, mini_batch_size),
                                data_input_2.narrow(0, b, mini_batch_size), train=False)

            # Get the targets in 1-hot encoding
            predicted_targets = helpers.encode_to_one_hot(torch.argmax(output, 1, keepdim=True))

            # Count the number of errors
            nb_data_errors += helpers.compute_nb_errors(data_target, b, predicted_targets, mini_batch_size)

        return nb_data_errors / N
