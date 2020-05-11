import torch
from torch import nn
import helpers
from Nets import DigitNet, CompNet


class NaiveSiamese:

    def __init__(self, nb_hidden, weight_sharing = True):
        model_digit = DigitNet(nb_hidden)
        self.model = CompNet(model_digit, weight_sharing = weight_sharing)

    def train(self, train_input_1, train_input_2, train_target, mini_batch_size=25,
              nb_epochs=25, lr=1e-1, verbose=False):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for e in range(nb_epochs):
            if verbose and e % 5 == 1:
                print("Epochs {}".format(e))
                print("loss = {}".format(loss))

            for b in range(0, train_input_1.size(0), mini_batch_size):
                train_input_1_ = train_input_1.narrow(0, b, mini_batch_size)
                train_input_2_ = train_input_2.narrow(0, b, mini_batch_size)

                # Forward pass
                output_comp = self.model(train_input_1_, train_input_2_)

                batch_target = train_target.narrow(0, b, mini_batch_size)

                loss = criterion(output_comp, batch_target)

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

    def compute_errors(self, data_input_1, data_input_2, data_target, mini_batch_size=25):

        nb_data_errors = 0

        N = data_input_1.size(0)

        for b in range(0, data_input_1.size(0), mini_batch_size):
            output = self.model(data_input_1.narrow(0, b, mini_batch_size),
                                data_input_2.narrow(0, b, mini_batch_size), train=False)

            predicted_targets = helpers.encode_to_one_hot(torch.argmax(output, 1, keepdim=True))

            for k in range(mini_batch_size):

                if not torch.equal(torch.eq(data_target[b + k], predicted_targets[k]), torch.tensor([True, True])):
                    # if data_target[b + k] != predicted_targets[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors / N
