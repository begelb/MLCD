import torch
import numpy as np
from torch import nn
from .network import Regression_Cubical_Network_One_Nonlinearity, get_optimizer

def loss_reduction(init_loss, final_loss, threshold):
    reduction = (init_loss - final_loss)/final_loss
    if reduction > threshold:
        return True
    else:
        return False

''' Train loop returns the loss on the last batch of the epoch '''
def train_loop(dataloader, model, loss_fn, optimizer, batch_size, num_labels):
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)

        '''
        By default, .backward() in PyTorch 2.1.1 computes the accumulated gradients (see https://pytorch.org/docs/stable/_modules/torch/autograd.html)
        The following .zero_grad() call sets the accumulated gradient to zero
        '''
        
        optimizer.zero_grad()

        pred = torch.reshape(pred, (batch_size, -1))
        
       # print('pred: ', pred)
       # print('y: ', y)
      #  pred_thresh = torch.clamp(pred, min=0.0, max=float(num_labels - 1))

        loss = loss_fn(pred, y)
        loss.backward(retain_graph = True)

        optimizer.step()

        if batch == num_batches - 1:
            loss = loss.item()
            return loss
    
def test_loop(dataloader, model, loss_fn, num_labels):
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred_thresh = torch.clamp(pred, min=0.0, max = float(num_labels - 1))
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss
    
def train_and_test(config, N, train_dataloader, test_dataloader, batch_size, epochs, patience, reduction_thresh):

    restart_count = 0

    ''' Train and test the network '''

    # Keep training until the train loss reduces by at least reduction_thresh * 100 percent
    # If training ends and this condition is not met, the network is reinitialized
    while True:

        ''' Set up neural network and training variables'''
        loss_fn = nn.MSELoss()
        model = Regression_Cubical_Network_One_Nonlinearity(N, batch_size, config)
        optimizer = get_optimizer(config, model)

        test_loss_list = []
        train_loss_list = []

        for epoch_number in range(epochs):

            # Train
            loss_train = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, config.num_labels)
            train_loss_list.append(loss_train)

            # Test
            loss_test = test_loop(test_dataloader, model, loss_fn, config.num_labels)
            test_loss_list.append(loss_test)

            if config.verbose:
                print(f"Epoch {epoch_number + 1}/{epochs}")
                print(f"Test loss: {loss_test:>7f}")
                print(f"Train loss: {loss_train:>7f}")

            # Early stopping condition compares the current test loss mean over 'patience' epochs to the test loss mean over 'patience' epochs at the last step
            # If the test loss mean has increased, then stop training
            if epoch_number >= patience:
                print('np.mean(test_loss_list[-patience:]) ', np.mean(test_loss_list[-patience:]))
                print('test_loss_list[-patience-1:-1] ', np.mean(test_loss_list[-patience-1:-1]))
                if np.mean(test_loss_list[-patience:]) > np.mean(test_loss_list[-patience-1:-1]):
                    print('stopping patience')

                    # Check if the train loss reduced by at least reduction_thresh * 100 percent
                    loss_reduced = loss_reduction(train_loss_list[0], train_loss_list[-1], 0.1)
                    if loss_reduced:
                        return model, train_loss_list, test_loss_list, restart_count
                    else:
                        restart_count += 1
                        break
                else:
                    print('not stopping patience')
        else: 
            loss_reduced = loss_reduction(train_loss_list[0], train_loss_list[-1], 0.1)
            if loss_reduced:
                return model, train_loss_list, test_loss_list, restart_count
            else:
                restart_count += 1

def compute_accuracy(model, dataloader, config, labeling_threshold):
    num_labels = config.num_labels
    num_batches = len(dataloader)
    correct_num = 0
    total_samples = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            pred_thresh = torch.clamp(pred, min=0.0, max = float(num_labels - 1))

            for index, elmt in enumerate(pred_thresh):
                prediction = elmt[0]
                label = y[index][0]

                if label - labeling_threshold <= prediction <= label + labeling_threshold:
                    correct_num += 1
                total_samples += 1

    accuracy = correct_num/total_samples
    return accuracy