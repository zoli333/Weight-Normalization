import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import os
import time
import numpy as np
import random
import torch.backends.cudnn
from model import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_determenistic_mode(SEED, disable_cudnn):
    # https://darinabal.medium.com/deep-learning-reproducible-results-using-pytorch-42034da5ad7
    # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
    torch.manual_seed(SEED)                       # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)                             # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)              # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False    # Causes cuDNN to deterministically select an algorithm,
                                                 # possibly at the cost of reduced performance
                                                 # (the algorithm itself may be nondeterministic).
        torch.backends.cudnn.deterministic = True # Causes cuDNN to use a deterministic convolution algorithm,
                                                  # but may slow down performance.
                                                  # It will not guarantee that your training process is deterministic
                                                  # if you are using other libraries that may use nondeterministic algorithms
    else:
        torch.backends.cudnn.enabled = False # Controls whether cuDNN is enabled or not.
                                         # If you want to enable cuDNN, set it to True.


set_determenistic_mode(12345, False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_generator = torch.Generator()
train_generator.manual_seed(0)

sample_batch_generator = torch.Generator()
sample_batch_generator.manual_seed(1)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 100
sample_batch_size = 500
learning_rate = 0.003


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, worker_init_fn=seed_worker, generator=train_generator)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

nr_batches_train = float(trainset.data.shape[0]) / batch_size

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


normalizer_types = ['no_norm', 'weight', 'weight', 'batch_norm', 'weight_mean_only_batch_norm', 'weight_mean_only_batch_norm']
inits = ['gaussian', 'gaussian', 'gaussian_datadep', 'gaussian', 'gaussian', 'gaussian_datadep']
lrs = [0.0003, 0.003, 0.003, 0.003, 0.003, 0.003]
names = ['no_norm', 'weight_norm', 'weight_norm_with_init', 'batch_norm', 'weight_norm_and_mean_only_batch_norm', 'weight_norm_and_mean_only_batch_norm_with_init']
sample_batch_sizes = [0, 0, 500, 0, 0, 500]
is_skipped = [1, 1, 1, 1, 1, 1]


assert len(normalizer_types) == len(inits) == len(lrs) == len(names) == len(sample_batch_sizes) == len(is_skipped)


def test(input_model):
    # get final accuracy
    correct = 0
    total = 0
    input_model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = input_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100. * float(correct) / total} %')
    input_model.train()
    return correct, total


if __name__ == '__main__':
    for norm_type_idx, norm_type in enumerate(normalizer_types):
        if is_skipped[norm_type_idx] is None:
            continue

        model = None
        sample_batch = None

        begin_time = time.time()

        sample_batch_size = sample_batch_sizes[norm_type_idx]

        if sample_batch_size != 0:
            sample_batch_generator.manual_seed(123123141)
            samplebatchloader = torch.utils.data.DataLoader(trainset, batch_size=sample_batch_size,
                                                            shuffle=True, worker_init_fn=seed_worker,
                                                            generator=sample_batch_generator)
            dataiter = iter(samplebatchloader)
            sample_batch, _ = next(dataiter)
            sample_batch = sample_batch.to(device)
        else:
            sample_batch = None

        learning_rate = lrs[norm_type_idx]
        model = Model(normalizer=norm_type, init=inits[norm_type_idx], sample_batch=sample_batch).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        exp_dir = names[norm_type_idx] + "_" + "{}".format(learning_rate).replace('.', 'p') \
                  + '_sbs' + str(sample_batch_size) + '_' + inits[norm_type_idx]

        os.makedirs(exp_dir, exist_ok=True)

        train_results_file = open(exp_dir + '/train_results.csv', 'w')
        test_results_file = open(exp_dir + '/test_results.csv', 'w')

        for epoch in range(100):  # loop over the dataset multiple times
            # set the generator seed
            train_generator.manual_seed(123456789 + epoch)
            running_loss = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()

            train_err = running_loss / nr_batches_train
            print(f'{epoch + 1}, {train_err:.3f}')
            train_results_file.write('%d, %.3f\n' % (epoch + 1, train_err))
            train_results_file.flush()

            if epoch in (0, 9, 19, 29, 39, 49, 59, 69, 79, 99):
                current_time = time.time()
                correct, total = test(model)
                test_err = 100. * correct / total
                test_results_file.write('%d, %d, %.5f\n' % (epoch + 1, current_time - begin_time, test_err))
                test_results_file.flush()

        print('Finished Training')
