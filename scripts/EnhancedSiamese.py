import torch
from torch import nn
from scripts.Nets import DigitNet, CompNet
from scripts import helpers


class EnhancedSiamese:
    """
        Implementation of the Enhanced model taking advantages of the train classes having two variations
            1. With weight sharing : using a single instance of a digitNet module
            2. Without weight sharing : using two clones of a digitNet module
        """

    def __init__(self, nb_hidden, weight_sharing=True):
        if weight_sharing:
            self.model_digit = DigitNet(nb_hidden)
            self.model_comp = CompNet(self.model_digit, weight_sharing=weight_sharing)
        else:
            self.model_digit_1 = DigitNet(nb_hidden)
            self.model_digit_2 = DigitNet(nb_hidden)
            self.model_comp = CompNet(self.model_digit_1, self.model_digit_2, weight_sharing=weight_sharing)

        self.weight_sharing = weight_sharing

    def train(self,
              train_input_1, train_input_2, train_classes_1, train_classes_2, train_target,
              mini_batch_size=25, nb_epochs=50, lr=1e-1, verbose=False):
        # SGD optimizer
        optimizer_comp = torch.optim.SGD(self.model_comp.parameters(), lr=lr)

        # Cross Entropy Loss used for the digit recognition (includes a softmax mapping)
        criterion_digit = nn.CrossEntropyLoss()
        # Binary Cross Entropy Loss used for digit comparison
        criterion_comp = nn.BCELoss()

        loss = 0
        loss_img = 0
        loss_comp = 0
        for e in range(nb_epochs):
            if verbose and (e % 5 == 1):
                print("Epochs {}".format(e))
                print("loss = {}, loss_img = {}, loss_comp = {}".format(loss, loss_img, loss_comp))

            for b in range(0, train_input_1.size(0), mini_batch_size):

                # digit classification depending on weight sharing configuration
                # Run a forward pass using a mini batch of the image data
                if self.weight_sharing:

                    output_img_1 = self.model_digit(train_input_1.narrow(0, b, mini_batch_size))
                    output_img_2 = self.model_digit(train_input_2.narrow(0, b, mini_batch_size))

                    self.model_digit.zero_grad()
                else:

                    output_img_1 = self.model_digit_1(train_input_1.narrow(0, b, mini_batch_size))
                    output_img_2 = self.model_digit_2(train_input_2.narrow(0, b, mini_batch_size))

                    self.model_digit_1.zero_grad()
                    self.model_digit_2.zero_grad()

                # Measure the recognition results using CELoss
                loss_img_1 = criterion_digit(output_img_1, train_classes_1.narrow(0, b, mini_batch_size))
                loss_img_2 = criterion_digit(output_img_2, train_classes_2.narrow(0, b, mini_batch_size))
                loss_img = loss_img_1 + loss_img_2

                # Continue the forward pass for the digit comparison
                output_comp = self.model_comp(train_input_1.narrow(0, b, mini_batch_size),
                                              train_input_2.narrow(0, b, mini_batch_size))

                # Take a mini batch from the target data
                batch_target = train_target.narrow(0, b, mini_batch_size)

                # Measure the comparison results using BCELoss
                loss_comp = criterion_comp(output_comp, batch_target)
                loss = loss_img + loss_comp

                # Apply the backward step
                self.model_comp.zero_grad()
                loss.backward()
                optimizer_comp.step()

    def test(self, data_input_1, data_input_2, data_target, mini_batch_size=25):
        """Test method using the error rate as a metric """
        # Init the the number of errors
        nb_data_errors = 0
        # Number of samples
        N = data_input_1.size(0)

        for b in range(0, data_input_1.size(0), mini_batch_size):
            # Run the model on a mini batch of the images
            output_comp = self.model_comp(data_input_1.narrow(0, b, mini_batch_size),
                                          data_input_2.narrow(0, b, mini_batch_size), train=False)

            # Get the targets in 1-hot encoding using rounding ( x < 0.5 --> 0 , x >= 0.5 --> 1)
            predicted_targets = torch.round(output_comp)

            # Count the number of errors
            nb_data_errors += helpers.compute_nb_errors(data_target, b, predicted_targets, mini_batch_size)

        return nb_data_errors / N
