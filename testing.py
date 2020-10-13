import torch
import torch.nn as nn
from pathlib import Path
from grasp_net import GraspNet
from visual_tactile_dataset import VisualTactileDataset
from torch.utils.data import DataLoader
from utils import eval_model

if __name__ == '__main__':  
    # Loading model
    model = GraspNet()
    models_dir = Path(__file__).parent.resolve() / 'Models'
    model_path = models_dir / "BestVal_e1_b80_a0.816.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Dataloader
    loader_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 0}
    test_dir = Path(__file__).parent.resolve() / 'Data/Train'
    test_dataset = VisualTactileDataset(test_dir, load_data=True)
    test_loader = DataLoader(test_dataset, **loader_params)

    # Test model
    criterion = nn.CrossEntropyLoss()
    accuracy, loss = eval_model(model, criterion, test_loader)
    print('Test - loss: %.3f - accuracy: %.3f' % (loss, accuracy))

