# author: Robert Andreas Fritsch 1431348
# author: Jan Zoppe 1433409
# date: 2021-02-05

import torch
import torch.nn as nn

from model import Network
from data import ChristmasImages


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(net, path):
    net.eval()
    net = net.to(device())

    data_loader = ChristmasImages(path, training=False, categorized=True).data_loader()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device()), data[1].to(device())

            total += labels.size(0)

            _, predicted = torch.max(net(inputs).data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy of the network on ', path, 'images: %d %%' % (100 * accuracy))
    return accuracy


def test_train(net):
    return test(net, './data/train')


def export_predictions(net):
    net.eval()
    net.to(device())

    data_loader = ChristmasImages('./data/val', training=False, categorized=False).data_loader(batch_size=1)

    f = open("predictions.csv", "w")
    f.write('Id,Category\n')

    with torch.no_grad():
        for i, image in enumerate(data_loader):
            outputs = net(image.to(device()))

            _, p = torch.max(outputs, 1)

            f.write(str(i))
            f.write(',')
            f.write(str(p.item()))
            f.write('\n')
            f.flush()

    f.flush()
    f.close()


def train(net):
    # configure training

    optimizer_learning_rate = 0.00001
    print_train_every_batch = 50
    epochs = 3

    train_break = 0.95

    # early break
    train_accuracy = test_train(net)

    if train_accuracy > train_break:
        return net

    # configure network
    net.train()
    net = net.to(device())

    # create the data loader
    data_loader = ChristmasImages('./data/train', training=True, categorized=True).data_loader()

    # initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(net.parameters(), lr=optimizer_learning_rate, momentum=optimizer_momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_learning_rate)

    # loop over the dataset multiple times
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(data_loader):

            inputs = data[0].to(device())
            labels = data[1].to(device())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i % print_train_every_batch) == (print_train_every_batch - 1):
                print('[%2d, %4d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_train_every_batch))
                running_loss = 0.0

        current_train_accuracy = test_train(net)

        # load model if there is a big negative improvement
        if (train_accuracy - 0.02) > current_train_accuracy:
            net.load_model()
            print('Model loaded!')
            continue

        # dont save the model if there was no improvement
        if train_accuracy > current_train_accuracy:
            print('Model new chance!')
            continue

        # save the model if there was an improvement
        train_accuracy = current_train_accuracy
        net.save_model()
        print('Model saved!')

        if train_accuracy >= train_break:
            break

    return net


def main():

    # net = Network()
    # while test_cheat(net) < 0.13:
    #     net = Network()
    # net.save_model()

    net = Network.load_model()
    net = train(net)

    export_predictions(net)


if __name__ == "__main__":
    main()
