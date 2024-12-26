from train_policy import *
from racer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
import matplotlib.pyplot as plt

from driving_policy import DiscreteDrivingPolicy
from train_policy import train_discrete, test_discrete
from dataset_loader import DrivingDataset
from utils import DEVICE

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='./weights/learner_{i}.weights')
    parser.add_argument("--dagger_iterations", help="", type=int, default=10)
    args = parser.parse_args()

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where
    #####

    #print ('TRAINING LEARNER ON INITIAL DATASET')
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

    args.class_dist = get_class_distribution(training_iterator, args)
    args.weighted_loss = True

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
            torch.save(driving_policy.state_dict(), './weights/learner_0.weights')
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")

    cumulative_errors = []

    for i in range(1, args.dagger_iterations + 1):
        # Load weights and execute DAgger iteration
        args.learner_weights = args.weights_out_file.format(i=i-1)
        steering_network = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
        steering_network.load_state_dict(torch.load(args.learner_weights))

        # Use steering_network in racer environment
        args.run_id = i
        args.save_expert_actions = True
        args.expert_drives = False  # Ensure it is the network trying to control the car
        args.timesteps = 100000
        args.out_dir = './dataset/train'
        cumulative_error = run(steering_network, args)
        print(f"------------------------CUMULATIVE TRACKING ERROR IS {cumulative_error}----------------------------")
        cumulative_errors.append(cumulative_error)

        # Aggregate new data and retrain
        training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)
        training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)

        opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
        args.start_time = time.time()

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
                torch.save(driving_policy.state_dict(), args.weights_out_file.format(i=i))
                print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
        
    args.learner_weights = args.weights_out_file.format(i=args.dagger_iterations)
    steering_network = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    steering_network.load_state_dict(torch.load(args.learner_weights))

    # Use steering_network in racer environment
    args.save_expert_actions = False
    args.expert_drives = False  # Ensure it is the network trying to control the car
    cumulative_error = run(steering_network, args)
    cumulative_errors.append(cumulative_error)
   
    # Plot cumulative errors
    plt.figure()
    plt.plot(range(args.dagger_iterations + 1), cumulative_errors, marker='o')
    plt.title('Cumulative Cross-Track Error Across DAgger Iterations')
    plt.xlabel('DAgger Iteration')
    plt.ylabel('Cumulative Cross-Track Error')
    plt.grid(True)
    plt.savefig('dagger_iterations_try2.png')
    plt.show()
