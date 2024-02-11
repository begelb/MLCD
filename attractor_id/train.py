import torch
from torch import nn
from .network import Regression_Cubical_Network_One_Nonlinearity, get_optimizer

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
        pred_thresh = torch.clamp(pred, min=0.0, max=float(num_labels - 1))

        loss = loss_fn(pred_thresh, y)
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
            test_loss += loss_fn(pred_thresh, y).item()
    test_loss /= num_batches
    return test_loss
    
def train_and_test(config, N, train_dataloader, test_dataloader, batch_size, epochs):

    ''' Set up neural network and training variables'''
    loss_fn = nn.MSELoss()
    model = Regression_Cubical_Network_One_Nonlinearity(N, batch_size, config)
    optimizer = get_optimizer(config, model)

    test_loss_list = []
    train_loss_list = []

    ''' Train and test the network '''

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

    return model, train_loss_list, test_loss_list