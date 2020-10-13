import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pathlib import Path
from visual_tactile_dataset import VisualTactileDataset
from grasp_net import GraspNet
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import eval_model, save_model

if __name__ == '__main__':  
    # Log  
    writer = SummaryWriter()
    models_dir = Path(__file__).parent.resolve() / 'Models'

    # Data loader
    loader_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 0}
    train_dir = Path(__file__).parent.resolve() / 'Data/Train'
    train_dataset = VisualTactileDataset(train_dir, data_cache_size = 10)
    train_loader = DataLoader(train_dataset, **loader_params)
    val_dir = Path(__file__).parent.resolve() / 'Data/Validation'
    val_dataset = VisualTactileDataset(val_dir, data_cache_size = 5)
    val_loader = DataLoader(val_dataset, **loader_params)

    # Network
    model = GraspNet()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training parameters
    num_epochs = 20 # Number of epochs to train
    batch = 1 # Current batch
    eval_batches = 20 # Log info every eval_batches
    running_loss = 0.0 # Average running loss
    total = 0 # Total elements evaluated
    correct = 0 # Correct predictions
    best_val_loss = float('inf') # Best validation loss seen during training
    best_val_accur = 0 # Best validation accuracy seen during training

    # Training Routine
    for epoch in range(num_epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data["cam_bef"], data["cam_dur"], data["lgel_bef"], 
                 data["lgel_dur"], data["rgel_bef"], data["rgel_dur"])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Train Accuracy and Loss
            running_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch == 1 or batch % eval_batches == 0:  
                # Train Accuracy and Loss
                train_accuracy = correct / total
                train_loss = running_loss / eval_batches                
                
                # Validation Accuracy and Loss
                val_accuracy, val_loss = eval_model(model, criterion, val_loader)

                #Log info 
                writer.add_scalar('Loss/train', train_loss, batch)
                writer.add_scalar('Accuracy/train', train_accuracy, batch)
                writer.add_scalar('Loss/validation', val_loss, batch)
                writer.add_scalar('Accuracy/validation', val_accuracy, batch)
                print('[%d, %5d] Training - loss: %.3f - accuracy: %.3f' % (epoch + 1, batch, train_loss, train_accuracy))
                print('Validation - loss: %.3f - accuracy: %.3f' % (val_loss, val_accuracy))

                # Save models
                if(val_loss < best_val_loss ):    
                    model_name = "BestLoss_e%d_b%d_l%.3f.pth" % (epoch + 1, batch, val_loss )
                    save_model(epoch, model, optimizer, scheduler, loss, models_dir / model_name)
                    best_val_loss = val_loss  
                if(val_accuracy > best_val_accur ):    
                    model_name = "BestVal_e%d_b%d_a%.3f.pth" % (epoch + 1, batch, val_accuracy )
                    save_model(epoch, model, optimizer, scheduler, loss, models_dir / model_name)
                    best_val_accur = val_accuracy

                # Restart parameters
                running_loss = 0.0
                correct = 0
                total = 0

            batch += 1
        scheduler.step()