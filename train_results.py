'''
Classes and methods related to abstracting and handling results later (saved in .pkl files)

Author: Diedre Carmo
https://github.com/dscarmo
'''
from matplotlib import pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn


class TrainResults():
    '''
    Stores training results
    '''
    @staticmethod
    def load(filepath):
        tr = None
        with open(filepath, "rb") as f:
            tr = pickle.load(f)
        return tr

    def __init__(self, val_loss, val_acc, train_loss, train_acc, nepochs, loss_name, metric_name, plot_title="",  val_class=None,
                 train_class=None, metric_name_class="Class Accuracy"):
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.nepochs = nepochs
        self.loss_name = loss_name
        self.metric_name = metric_name
        self.plot_title = plot_title
        self.metric_name_class = metric_name_class

        self.train_class = train_class
        self.val_class = val_class

    def plot(self, plot_title=None, show=True, loss_only=False, o="", generate_new_figure=False, ylim=1.0, classify=False,
             lower_ylim=0.0):
        # epoch_range = range(1, self.nepochs + 1)
        epoch_range = range(len(self.train_acc))

        plt_title = ''
        if show:
            if plot_title is None:
                plt_title = self.plot_title
            else:
                plt_title = plot_title
            if not generate_new_figure:
                plt.figure(num=plt_title)

        # old pkls support
        if classify:
            classify = (self.train_class is not None) and (self.val_class is not None)

        if not loss_only:
            if generate_new_figure:
                plt.figure(plt_title + " metrics")
            else:
                plt.subplot(1, 2+1*classify, 1)
            plt.ylabel("Dice")
            plt.xlabel("Epoch")

            if ylim is not None:
                plt.ylim(lower_ylim, ylim)
            plt.plot(epoch_range, self.val_acc, '-', label=o + ' val')
            plt.legend()

            if ylim is not None:
                plt.ylim(lower_ylim, ylim)
            plt.plot(epoch_range, self.train_acc, '-', label=o + ' train')
            plt.legend()

            if classify:
                if generate_new_figure:
                    plt.figure(num=plt_title + " classification metrics")
                else:
                    plt.subplot(1, 2+1*classify, 2)
                plt.ylabel(self.metric_name_class)
                plt.xlabel("Epoch")

                if ylim is not None:
                    plt.ylim(lower_ylim, ylim)
                plt.plot(epoch_range, self.val_class, '-', label=o + ' val')
                plt.legend()

                if ylim is not None:
                    plt.ylim(lower_ylim, ylim)
                plt.plot(epoch_range, self.train_class, '-', label=o + ' train')
                plt.legend()

            if generate_new_figure:
                plt.figure(plt_title + " loss")
            else:
                plt.subplot(1, 3, 3)
        plt.ylabel(self.loss_name)
        plt.xlabel("Epoch")
        if ylim is not None:
            plt.ylim(lower_ylim, ylim)
        plt.plot(epoch_range, self.val_loss, '-', label=o + ' val')
        plt.legend()
        if ylim is not None:
            plt.ylim(lower_ylim, ylim)
        plt.plot(epoch_range, self.train_loss, '-', label=o + ' train')
        plt.legend()
        if show:
            plt.show()

    def save(self, filepath):
        '''
        Saves itself
        '''
        if filepath[-4:] != ".pkl":
            print("WARNING: save path not valid! should be a .pkl file, trying to save anyway")
        with open(filepath, "wb") as output_file:
            pickle.dump(self, output_file)


# DEPRECATED, will be removed once backwards compatibility is assured with new class
class OLD_TrainResults():
    '''
    Stores training results
    '''
    @staticmethod
    def load(filepath):
        tr = None
        with open(filepath, "rb") as f:
            tr = pickle.load(f)
        return tr

    def __init__(self, val_loss, val_acc, train_loss, train_acc, nepochs, loss_name, metric_name, plot_title=""):
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.nepochs = nepochs
        self.loss_name = loss_name
        self.metric_name = metric_name
        self.plot_title = plot_title

    def plot(self, plot_title=None, show=True, loss_only=False, o=""):
        # epoch_range = range(1, self.nepochs + 1)
        epoch_range = range(len(self.train_acc))
        if show:
            if plot_title is None:
                plt.figure(num=self.plot_title)
            else:
                plt.figure(num=plot_title)
        if not loss_only:
            plt.subplot(1, 2, 1)
            plt.ylabel(self.metric_name)
            plt.xlabel("Epoch")
            plt.ylim(0.0, 1.0)
            plt.plot(epoch_range, self.val_acc, '-', label=o + ' val')
            plt.legend()
            plt.ylim(0.0, 1.0)
            plt.plot(epoch_range, self.train_acc, '-', label=o + ' train')
            plt.legend()
            plt.subplot(1, 2, 2)
        plt.ylabel(self.loss_name)
        plt.xlabel("Epoch")
        plt.ylim(0.0, 1.0)
        plt.plot(epoch_range, self.val_loss, '-', label=o + ' val')
        plt.legend()
        plt.ylim(0.0, 1.0)
        plt.plot(epoch_range, self.train_loss, '-', label=o + ' train')
        plt.legend()
        if show:
            plt.show()

    def save(self, filepath):
        '''
        Saves itself
        '''
        if filepath[-4:] != ".pkl":
            print("WARNING: save path not valid! should be a .pkl file, trying to save anyway")
        with open(filepath, "wb") as output_file:
            pickle.dump(self, output_file)


class TestModel(nn.Module):
    '''
    Single convolution test model
    '''
    def __init__(self, in_ch, out_ch):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3)

    def forward(self, x):
        x = self.conv(x)
        return x


def test_display(wintitle, input, output):
    plt.figure(num=wintitle)
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(input.squeeze().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Output")
    plt.imshow(output.squeeze().detach().numpy(), cmap='gray')


def tr_test():
    '''
    Tests train results with test model
    '''
    testimg = torch.rand((1, 1, 1000, 1000))

    test_model = TestModel(1, 1)
    test_output = test_model(testimg)

    test_display("Pre-save", testimg, test_output)

    nepochs = 10
    vl = np.random.rand(nepochs)
    va = np.random.rand(nepochs)
    tl = np.random.rand(nepochs)
    ta = np.random.rand(nepochs)

    tr = TrainResults(test_model, vl, va, tl, ta, nepochs, "test loss", "test metric")
    tr.plot(plot_title="Pre-save plot", show=False)
    test_path = "./test.pkl"
    tr.save(test_path)

    loaded_tr = TrainResults.load(test_path)
    loaded_tr.plot(plot_title="Post-load plot", show=False)
    loaded_test_output = loaded_tr.best_model(testimg)
    test_display("Post-load", testimg, loaded_test_output)
    plt.show()


def openall(maindir='.'):
    import glob
    from os.path import join as add_path
    from os.path import basename
    opened = 0
    for filename in glob.iglob(add_path(maindir, "*.pkl")):
        plt.figure(basename(filename))
        tr = TrainResults.load(filename)
        tr.plot(show=False)
        opened += 1
    if opened == 0:
        print("No .pkl TrainResults files found on {}".format(maindir if maindir != '.' else "current directory"))
    else:
        print("Finished. Opened {} results.".format(opened))
        plt.show()


if __name__ == "__main__":
    from sys import argv
    if argv[1] == "tr_test":
        tr_test()
    elif argv[1] == "openall":
        try:
            maindir = argv[2]
        except IndexError:
            print("No directory passed, using current directory")
            maindir = '.'
        openall(maindir)
