import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os

from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

def train_discrete(model, iterator, opt, args):
    model.train()

    loss_hist = []
    
    # # Calculate class weights if weighted_loss is True
    # if args.weighted_loss:
    #     epsilon = 1e-6  # Small constant to prevent division by zero
    #     class_weights = torch.tensor(args.class_dist, dtype=torch.float32).to(DEVICE)
    #     class_weights = 1.0 / (class_weights + epsilon)  # Adding epsilon to avoid division by zero
    #     class_weights = class_weights / class_weights.sum()  # Normalize weights
    #     criterion = nn.CrossEntropyLoss(weight=class_weights)
    # else:
    #     criterion = nn.CrossEntropyLoss()


    # Do one pass over the data accessed by the training iterator
    # Upload the data in each batch to the GPU (if applicable)
    # Zero the accumulated gradient in the optimizer 
    # Compute the cross_entropy loss with and without weights  
    # Compute the derivatives of the loss w.r.t. network parameters
    # Take a step in the approximate gradient direction using the optimizer opt  
    
    for i_batch, batch in enumerate(iterator):

        x = batch['image'].to(DEVICE)
        y = batch['cmd'].to(DEVICE)

        # Zero the gradients
        opt.zero_grad()

        # Forward pass: Compute model output and loss
        logits = model(x)

        if args.weighted_loss:
            class_weights = torch.tensor(1.0/args.class_dist, dtype=torch.float32, device=DEVICE)
            loss = F.cross_entropy(logits, y, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, y)
        
        # Backward pass: Compute gradients
        loss.backward()

        # Update model parameters
        opt.step()

        
        loss = loss.detach().cpu().numpy()
        loss_hist.append(loss)
        
        PRINT_INTERVAL = int(len(iterator) / 3)        
        if (i_batch + 1) % PRINT_INTERVAL == 0:
            print ('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t Time: {:10.3f}'.format(
                i_batch, len(iterator),
                i_batch / len(iterator) * 100,
                np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0),
                time.time() - args.start_time,
            ))


def accuracy(y_pred, y_true):
    "y_true is (batch_size) and y_pred is (batch_size, K)"
    _, y_max_pred = y_pred.max(1)
    correct = ((y_true == y_max_pred).float()).mean() 
    acc = correct * 100
    return acc


def test_discrete(model, iterator, opt, args):
    model.train()
    
    acc_hist = []
    
    for i_batch, batch in enumerate(iterator):
        x = batch['image']
        y = batch['cmd']

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        logits = model(x)
        y_pred = F.softmax(logits, 1)

        acc = accuracy(y_pred, y)
        acc = acc.detach().cpu().numpy()
        acc_hist.append(acc)
        
    avg_acc = np.asarray(acc_hist).mean()
    
    print ('\tVal: \tAcc: {}  Time: {:10.3f}'.format(
        avg_acc,
        time.time() - args.start_time,
    ))
    
    return avg_acc
    
def get_class_distribution(iterator, args):
    class_dist = np.zeros((args.n_steering_classes,), dtype=np.float32)
    for i_batch, batch in enumerate(iterator):
        y = batch['cmd'].detach().numpy().astype(np.int32)
        class_dist += np.bincount(y, minlength=args.n_steering_classes)
        
    return (class_dist / sum(class_dist))

    
def main(args):
    
    data_transform = transforms.Compose([ transforms.ToPILImage(),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                          transforms.RandomRotation(degrees=80),
                                          transforms.ToTensor()])
    
    training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)
    
    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=data_transform)
    
    training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    
    opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
    args.start_time = time.time()
    
    print (driving_policy)
    print (opt)
    print (args)

    args.class_dist = get_class_distribution(training_iterator, args)

    best_val_accuracy = 0 
    for epoch in range(args.n_epochs):
        print ('EPOCH ', epoch)

        # Train the model
        train_discrete(driving_policy, training_iterator, opt, args)

        # Validate the model
        val_accuracy = test_discrete(driving_policy, validation_iterator, opt, args)

        # Save model if the validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(driving_policy.state_dict(), args.weights_out_file)
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")

        
        # Train the driving policy
        # Evaluate the driving policy on the validation set
        # If the accuracy on the validation set is a new high then save the network weights 
        
        
    return driving_policy

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights",
                        required=True)
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurence",
                        default=False)
    
    args = parser.parse_args()

    main(args)
        
    
