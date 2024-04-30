import os 
os.chdir(r'D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation')
import json
import torch
import dataloaders
import models
from utils import losses, Logger
from utils.torchsummary import summary
from trainer import Trainer
from datetime import datetime
import torch.nn as nn

train_logger = Logger()
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
folder_path = r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\rough"

# List all files in the folder
files = os.listdir(folder_path)

# Filter JSON files
config_files = [file for file in files if file.endswith('.json')]
for config_file in config_files:
    print(config_file)
    config_path = os.path.join(folder_path, config_file)
    config = json.load(open(config_path))

    def get_instance(module, name, config, *args):
        return getattr(module, config[name]['type'])(*args, **config[name]['args'])

    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes).to(device)
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index']).to(device)

    trainer = Trainer(
        model=model,
        loss=loss,
        resume=None,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger
    )

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    time_taken = end_time - start_time

    time_folder = "Time"
    os.makedirs(time_folder, exist_ok=True)
    time_file_path = os.path.join(time_folder, f"{config['name']}_{config['trainer']['save_dir'].split('/')[-1]}.txt")

    if os.path.exists(time_file_path):
        os.remove(time_file_path)

    with open(time_file_path, 'w') as f:
        f.write(f'Training started at: {start_time}\n')
        f.write(f'Training ended at: {end_time}\n')
        f.write(f'Time taken for training: {time_taken}')

    print(f"Training time has been recorded and saved in '{time_file_path}'.")
