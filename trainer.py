import os
import torch
import logging
from tqdm import tqdm

import os
import torch
import logging
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, val_metric, checkpoint_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.val_metric_train = val_metric
        self.val_metric_test = val_metric
        self.best_val_metric = float('-inf')
        self.best_checkpoint = None

    def save_checkpoint(self, epoch, val_metric):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metric': val_metric
        }, checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")

        # Update the best checkpoint if necessary
        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.best_checkpoint = checkpoint_path
            logging.info(f"New best checkpoint at {checkpoint_path} with validation metric {val_metric:.4f}")

        # Delete old checkpoints
        self.delete_old_checkpoints(checkpoint_path)

    def delete_old_checkpoints(self, latest_checkpoint):
        checkpoints = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        checkpoints.remove(latest_checkpoint)
        if self.best_checkpoint in checkpoints:
            checkpoints.remove(self.best_checkpoint)
        for checkpoint in checkpoints:
            os.remove(checkpoint)
            logging.info(f"Deleted old checkpoint {checkpoint}")

    def load_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        if not checkpoints:
            logging.info("No checkpoints found. Starting from scratch.")
            return 0
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Loaded checkpoint from {checkpoint_path}, starting at epoch {start_epoch}")

        # Load the best validation metric and checkpoint
        if 'val_metric' in checkpoint:
            self.best_val_metric = checkpoint['val_metric']
            self.best_checkpoint = checkpoint_path
        return start_epoch

    def train(self, num_epochs):
        start_epoch = self.load_checkpoint()
        self.model.to(self.device)
        for epoch in range(start_epoch, num_epochs):
            logging.info('\n')
            logging.info(f'Starting epoch {epoch} out of {num_epochs}')
            self.model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                self.val_metric_train.update(outputs, labels)

            epoch_loss = running_loss / len(self.train_loader)
            val_metric_train = self.val_metric_test.compute()
            val_loss, val_metric = self.validate()
            logging.info(f"Epoch {epoch} out of {num_epochs}, Training Loss: {epoch_loss:.4f}, Training Metric: {val_metric_train:.4f}, Validation Loss: {val_loss:.4f}, Validation Metric: {val_metric:.4f}")
            self.save_checkpoint(epoch, val_metric)

    def validate(self):
        logging.info('Training of this epoch is complete, now validating')
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                self.val_metric_test.update(outputs, labels)

        val_loss = running_loss / len(self.val_loader)
        val_metric = self.val_metric_test.compute()
        self.val_metric_test.reset()
        return val_loss, val_metric


# class Trainer:
#     def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, val_metric, checkpoint_dir='checkpoints1'):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.device = device
#         if not os.path.exists(checkpoint_dir):
#             os.makedirs(checkpoint_dir)
#         self.checkpoint_dir = checkpoint_dir
#         self.val_metric_train = val_metric
#         self.val_metric_test = val_metric

#     def save_checkpoint(self, epoch):
#         checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#         }, checkpoint_path)
#         #logging.info(f"Checkpoint saved at {checkpoint_path}")

#     def load_checkpoint(self):
#         checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
#         if not checkpoints:
#             logging.info("No checkpoints found. Starting from scratch.")
#             return 0
#         latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
#         checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
#         logging.info(f"Loaded checkpoint from {checkpoint_path}, starting at epoch {start_epoch}")
#         return start_epoch

#     def train(self, num_epochs):
#         start_epoch = self.load_checkpoint()
#         self.model.to(self.device)
#         for epoch in range(start_epoch, num_epochs):
#             logging.info(f"Starting epoch {epoch} out of {num_epochs}")
#             self.model.train()
#             running_loss = 0.0
#             for inputs, labels in tqdm(self.train_loader):
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item() 
#                 self.val_metric_train.update(outputs, labels)

#             epoch_loss = running_loss / len(self.train_loader)
#             val_metric_train = self.val_metric_test.compute()
#             val_loss, val_metric = self.validate()
#             logging.info(f"Epoch {epoch} out of {num_epochs}, Training Loss: {epoch_loss:.4f}, Training Metric: {val_metric_train:.4f}, \
#                          Validation Loss: {val_loss:.4f}, Validation Metric: {val_metric:.4f}")
#             self.save_checkpoint(epoch)

#     def validate(self):
#         #logging.info('Training of this epoch is complete, now validating')
#         self.model.eval()
#         running_loss = 0.0
#         with torch.no_grad():
#             for inputs, labels in tqdm(self.val_loader):
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 running_loss += loss.item()
#                 self.val_metric_test.update(outputs, labels)

#         val_loss = running_loss / len(self.val_loader)
#         val_metric = self.val_metric_test.compute()
#         self.val_metric_test.reset()
#         return val_loss, val_metric