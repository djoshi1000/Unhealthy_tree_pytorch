base= r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation"
import os
os.chdir(base)
import argparse
import scipy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
import glob
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score, confusion_matrix
from tabulate import tabulate
import pandas as pd

availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')


# Define your functions
def load_ground_truth_mask(path, image_file_path):
    mask_file_path = os.path.join(path, os.path.basename(image_file_path))
    ground_truth_mask = Image.open(mask_file_path)
    ground_truth_mask= ground_truth_mask.resize((256,256))
    ground_truth_mask = np.array(ground_truth_mask)
    return ground_truth_mask
def load_checkpoint(checkpoint_path, device, model):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
            
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
# Function to calculate metrics
# def calculate_metrics(labels, predictions, class_weights=None):
#     print(labels.shape, prediction.shape)
#     if len(labels) == 0 or len(predictions) == 0:
#         return 0, 0, 0, 0
#     f1 = f1_score(labels, predictions, average='binary', sample_weight=class_weights)
#     recall = recall_score(labels, predictions, average='binary', sample_weight=class_weights)
#     precision = precision_score(labels, predictions, average='binary', sample_weight=class_weights)
#     accuracy = accuracy_score(labels, predictions, sample_weight=class_weights)
#     return f1, recall, precision, accuracy
def calculate_metrics(labels, predictions, class_weights=None):
    if len(labels) == 0 or len(predictions) == 0:
        return 0, 0, 0, 0
    
    if class_weights is not None:
        sample_weight = [class_weights[label] for label in labels]
    else:
        sample_weight = None

    f1 = f1_score(labels, predictions, average='binary', sample_weight=sample_weight)
    recall = recall_score(labels, predictions, average='binary', sample_weight=sample_weight)
    precision = precision_score(labels, predictions, average='binary', sample_weight=sample_weight)
    accuracy = accuracy_score(labels, predictions, sample_weight=sample_weight)
    return f1, recall, precision, accuracy



# # Define your configurations
# keyword_to_folder = {
#     "Data_RGN": ["data_RGN_new", "config_RGN_"],
#     "Data_PCA": ["data_PCA_new", "config_PCA_"],
#     "Data_NDVI": ["data_VI_new", "config_vi_"],
#     "RGB_PCA": ["Data_RGBPC_new", "config_rgb_pc_"],
#     "Data_RGB": ["Data_RGB_new", "config_rgb_"]
# }

model_names = ["UNetResnet", "DeepLab","SegResNet"]
num_classes = 2
data_path = r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\data"
base_checkpoint_path = r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\saved"
config_base= r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\rough"
class_weights = {0: 0.15, 1: 0.85} 
folders = [folder for folder in os.listdir(base_checkpoint_path) if os.path.isdir(os.path.join(base_checkpoint_path, folder)) and "Data_" in folder]
# Initialize lists to store metrics
all_subdatasets = []
f1_scores = []
recall_scores = []
precision_scores = []
pixel_accuracies = []
for model_name in model_names:
    for folder in folders:        
        images_folder = os.path.join(data_path, f"{'_'.join(folder.split('_')[:2])}", "Images", "test")            
        masks_folder = os.path.join(data_path, f"{'_'.join(folder.split('_')[:2])}", "masks", "test")
        # print(f"{'_'.join(folder.split('_')[:2])}")
        if 'aug' in folder:
            print('This is for augmentation folder')
            name= str(f"{'_'.join(folder.split('_')[:2])}_aug")
            jsons= f"config_{folder.split('_')[1]}_{model_name}_aug.json"
            base_checkpoint_path1= os.path.join(base_checkpoint_path, name)
            training_path = os.path.join(base_checkpoint_path1,model_name)
            
        else: 
            print('This is for non-augmentation folder')
            name= str(f"{'_'.join(folder.split('_')[:2])}_new")
            jsons= f"config_{folder.split('_')[1]}_{model_name}.json"
            base_checkpoint_path1= os.path.join(base_checkpoint_path, name)
            training_path = os.path.join(base_checkpoint_path1, model_name)

        # print(f'the training path is {training_path}')
        config_file_path= os.path.join(config_base, jsons)
        print(config_file_path)
        config= json.load(open(config_file_path))
        loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(loader.MEAN, loader.STD)
        num_classes = loader.dataset.num_classes
        palette = loader.dataset.palette
        if not os.path.exists(training_path): 
            print('Training path doesnot exist. Sorry')
            continue
    
        for i in os.listdir(training_path):
            checkpoint_path = os.path.join(training_path, i, "best_model.pth")
            if not os.path.exists(checkpoint_path):
                print("No checkpoint path was found.")
                continue
            model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
            model = load_checkpoint(checkpoint_path, device, model)
            image_files = sorted(glob.glob(os.path.join(images_folder, '*.png')))
            batch_size = 4
            image_batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]
            all_preds = []
            all_labels = []
            conf_matrices = []
            for batch in image_batches:
                batch_f1_scores = []
                batch_recall_scores = []
                batch_precision_scores = []
                batch_pixel_accuracies = []
                for img_file in batch:
                    image = Image.open(img_file).convert('RGB')
                    image= image.resize((256,256))
                    input = normalize(to_tensor(image)).unsqueeze(0)
                    ground_truth_mask = load_ground_truth_mask(masks_folder, img_file)
                    ground_truth_mask_tensor = torch.from_numpy(ground_truth_mask).to(device).cpu().numpy()
                                        # Assuming ground_truth_mask_tensor is a PyTorch tensor
                    # print(ground_truth_mask_tensor)
                    prediction = model(input.to(device))
                    upsampled_logits = nn.functional.interpolate(
                        prediction, 
                        size=ground_truth_mask.shape[-2:], 
                        mode="bilinear", 
                        align_corners=False
                        )
                    predicted_mask = (upsampled_logits.argmax(dim=1).cpu().numpy())
                    f1, recall, precision, accuracy = calculate_metrics(ground_truth_mask_tensor.flatten(),predicted_mask.flatten(), class_weights)
                    batch_f1_scores.append(f1)
                    batch_recall_scores.append(recall)
                    batch_precision_scores.append(precision)
                    batch_pixel_accuracies.append(accuracy)
            avg_f1_score = np.mean(batch_f1_scores)
            avg_recall_score = np.mean(batch_recall_scores)
            avg_precision_score = np.mean(batch_precision_scores)
            avg_accuracy = np.mean(batch_pixel_accuracies)
            all_subdatasets.append(f'{name}_{model_name}')
            f1_scores.append(avg_f1_score)
            recall_scores.append(avg_recall_score)
            precision_scores.append(avg_precision_score)
            pixel_accuracies.append(avg_accuracy)
metrics_dict = {
    "Subdataset": all_subdatasets,
    "Average F1 Score": f1_scores,
    "Average Recall Score": recall_scores,
    "Average Precision Score": precision_scores,
    "Average Pixel Accuracy": pixel_accuracies
}

metrics_df = pd.DataFrame(metrics_dict)
print(metrics_df)
print(tabulate(metrics_df, headers='keys', tablefmt='grid'))
metrics_df.to_csv(r'D:\Chapter_2\New_idea\CNNS_all_test_metrics.csv')
