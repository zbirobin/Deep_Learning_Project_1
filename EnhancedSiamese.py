import torch
from torch import nn
from Nets import DigitNet, CompNet


class EnhancedSiamese:

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

        optimizer_comp = torch.optim.SGD(self.model_comp.parameters(), lr=lr)
        criterion_digit = nn.CrossEntropyLoss()
        criterion_comp = nn.BCELoss()

        loss = 0
        loss_img = 0
        loss_comp = 0
        for e in range(nb_epochs):
            if verbose and (e % 5 == 1):
                print("Epochs {}".format(e))
                print("loss = {}, loss_img = {}, loss_comp = {}".format(loss, loss_img, loss_comp))

            for b in range(0, train_input_1.size(0), mini_batch_size):

                if self.weight_sharing:

                    # digit classification
                    output_img_1 = self.model_digit(train_input_1.narrow(0, b, mini_batch_size))
                    output_img_2 = self.model_digit(train_input_2.narrow(0, b, mini_batch_size))

                    self.model_digit.zero_grad()

                else:
                    # digit classification
                    output_img_1 = self.model_digit_1(train_input_1.narrow(0, b, mini_batch_size))
                    output_img_2 = self.model_digit_2(train_input_2.narrow(0, b, mini_batch_size))

                    self.model_digit_1.zero_grad()
                    self.model_digit_2.zero_grad()

                loss_img_1 = criterion_digit(output_img_1, train_classes_1.narrow(0, b, mini_batch_size))
                loss_img_2 = criterion_digit(output_img_2, train_classes_2.narrow(0, b, mini_batch_size))
                loss_img = loss_img_1 + loss_img_2

                output_comp = self.model_comp(train_input_1.narrow(0, b, mini_batch_size),
                                              train_input_2.narrow(0, b, mini_batch_size))

                batch_target = train_target.narrow(0, b, mini_batch_size)

                loss_comp = criterion_comp(output_comp, batch_target)
                loss = loss_img + loss_comp

                self.model_comp.zero_grad()
                loss.backward()
                optimizer_comp.step()

    def test(self,
             data_input_1, data_input_2, data_target, mini_batch_size=25):

        nb_data_errors = 0

        N = data_input_1.size(0)

        for b in range(0, data_input_1.size(0), mini_batch_size):

            output_comp = self.model_comp(data_input_1.narrow(0, b, mini_batch_size),
                                          data_input_2.narrow(0, b, mini_batch_size), train=False)

            predicted_targets = torch.round(output_comp)

            for k in range(mini_batch_size):
                # print(torch.eq(data_target[b + k], output_comp[k]))
                if not torch.equal(torch.eq(data_target[b + k], predicted_targets[k]), torch.tensor([True, True])):
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors / N
