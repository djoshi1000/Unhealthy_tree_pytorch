##Load the required Libraries
import pytorch_lightning as pl
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
from torchinfo import summary
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
warnings.filterwarnings("ignore", category=UserWarning)

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, split, feature_extractor):
        """
        Args:
            root_dir (string): Root directory of the dataset containing train/val/test folders.
            split (string): Specifies the data split (train, val, or test).
            feature_extractor (SegFormerFeatureExtractor): Feature extractor for images + segmentation maps.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.feature_extractor = feature_extractor
        self.id2label = {
            0: 'background',    # background pixel
            1: 'Crown',         # Crown
        }
        image_folder = 'images'
        mask_folder = 'masks'
        image_file_names = [f for f in os.listdir(os.path.join(self.root_dir, image_folder)) if '.png' in f]
        mask_file_names = [f for f in os.listdir(os.path.join(self.root_dir, mask_folder)) if '.png' in f]
        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.images[idx])
        mask_path = os.path.join(self.root_dir, 'masks', self.masks[idx])
        image = Image.open(image_path).convert('RGB')
        # print(image.size)
        segmentation_map = Image.open(mask_path)
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs  # Return filename along with encoded inputs
    
class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.train_step_outputs =[]
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            return_dict=False, 
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")
        
    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)
    
    def training_step(self, batch, batch_nb):        
        images, masks = batch['pixel_values'], batch['labels']        
        outputs = self(images, masks) 
        logits = outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        criterion = FocalLoss()
        loss=criterion(upsampled_logits, masks)
        self.train_step_outputs.append(loss)
        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        return(loss)
    
    def on_train_epoch_end(self):
        # if batch_nb % self.metrics_interval == 0:
        metrics = self.train_mean_iou.compute(
            num_labels=self.num_classes, 
            ignore_index=255, 
            reduce_labels=False,
        )
        avg_val_loss = torch.stack(self.train_step_outputs).mean()
        metrics = {'loss': avg_val_loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
        for k,v in metrics.items():
            # self.log(k,v,on_step=False, on_epochc=True, prog_bar=True)
            self.log(k, v, prog_bar=True, logger=True)
            self.log("epoch", self.current_epoch)
        return(metrics)

    
    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks) 
        logits = outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        criterion = FocalLoss()
        loss=criterion(upsampled_logits, masks)
        self.validation_step_outputs.append(loss)
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        return({'val_loss': loss})
    
    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]  
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        for k,v in metrics.items():
            self.log(k,v, on_step=False, on_epoch=True, prog_bar=True)
            self.log("epoch", self.current_epoch)
        return metrics
    
    def test_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        logits = outputs[1]
       
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        criterion =FocalLoss()
        loss=criterion(upsampled_logits, masks)
        self.test_step_outputs.append(loss)
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
            
        return({'test_loss': loss})
    
    def on_test_epoch_end(self):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack(self.validation_step_outputs).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
        for k,v in metrics.items():
            self.log(k,v)
        
        return metrics
    
    # def configure_optimizers(self):
    #     return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, 'AdamW')(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001,
            weight_decay=1e-4,
            amsgrad=True
        )
 
        LR = 0.00001
        WD = 1e-4
 
        # if self.hparams.optimizer_name in ("AdamW", "Adam"):
        #     optimizer = getattr(torch.optim, self.hparams.optimizer_name)(model.parameters(), lr=LR, 
        #                                                                   weight_decay=WD, amsgrad=True)
        # else:
        #     optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WD)
 
        # if self.hparams.use_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
 
            # The lr_scheduler_config is a dictionary that contains the scheduler
            # and its associated configuration.
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch", "name": "multi_step_lr"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
 
        # else:
        #     return optimizer
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl

# Define function to calculate metrics
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


id2color={0: (0, 0, 0), 1: (0, 0, 255)}
img_size = [256,256]
ds_mean = (0.485, 0.456, 0.406)
ds_std = (0.229, 0.224, 0.225)
batch_size = 12
num_workers = 0
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor.do_reduce_labels = False
feature_extractor.size = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define the base directory where the datasets are stored
base_dir = r"D:\Chapter_2\New_idea\NEW"
class_labels = ['Background', 'Unhealthy Tree']
 
datasets = {
    "Data_RGN": ["CIR_N", "CIR_A"],
    "Data_PCA": ["PCA_N", "PCA_A"],
    "Data_NDVI": ["VI_N", "VI_A"],
    "RGB_PCA": ["RGB_PCA_N", "RGB_PCA_A"],
    "Data": ["RGB_N", "RGB_A"]
}
f1_scores = []
all_subdatasets = []
recall_scores = []
precision_scores = []
pixel_accuracies = []
for dataset, subdatasets in datasets.items():
    dataset_dir = os.path.join(base_dir, dataset)
    # print(dataset_dir)
    train_dataset = SemanticSegmentationDataset(dataset_dir, 'train', feature_extractor)
    val_dataset = SemanticSegmentationDataset(dataset_dir, 'valid', feature_extractor)  # Corrected split
    test_dataset = SemanticSegmentationDataset(dataset_dir, 'test', feature_extractor)  # Corrected split

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    segformer_finetuner = SegformerFinetuner(
        train_dataset.id2label, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        test_dataloader=test_dataloader, 
        metrics_interval=5,
    )
    for subdataset in subdatasets:
        checkpoints_folder = os.path.join(r'D:\Chapter_2\New_idea\ipynb\New_100', subdataset, 'Tree_crown_segm_wtd_100', '*' ,'checkpoints')
        ckpt_files = glob(os.path.join(checkpoints_folder, '*.ckpt'))
        # print(ckpt_files)
        for CKPT_PATH in ckpt_files:
            CKPT_PATH
            checkpoint = torch.load(CKPT_PATH)
            segformer_finetuner.load_state_dict(checkpoint['state_dict'])
            # Set the model to evaluation mode
            segformer_finetuner.eval()
            # Initialize lists to store predictions and labels
            all_preds = []
            all_labels = []
            conf_matrices = []
            # Perform inference on the images
            for batch in test_dataloader:
                images, masks = batch['pixel_values'], batch['labels']
                outputs = segformer_finetuner(images, masks)
                loss, logits = outputs[0], outputs[1]
                upsampled_logits = nn.functional.interpolate(
                    logits, 
                    size=masks.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
                predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()
                masks = masks.cpu().numpy()            
                all_preds.extend(predicted_mask.flatten())
                all_labels.extend(masks.flatten())
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            class_weights = {0: 0.15, 1: 0.85}
            f1, recall, precision, accuracy = calculate_metrics(all_labels, all_preds, class_weights)
            print(f1, "this is f1",recall, "this is recall" ,precision, "this is precision" ,accuracy, "this is accuracy" ,subdataset, "this is subdayaxry"  )
            conf_matrix = confusion_matrix(all_labels, all_preds)
            # # Normalize confusion matrix
            conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Overall Confusion Matrix {subdataset}')
            plt.show()
        
            all_subdatasets.append(subdataset)
            f1_scores.append(f1)
            recall_scores.append(recall)
            precision_scores.append(precision)
            pixel_accuracies.append(accuracy)

# Create a dictionary to hold the metrics
metrics_dict = {
    "Subdataset": all_subdatasets,
    "Average F1 Score": f1_scores,
    "Average Recall Score": recall_scores,
    "Average Precision Score": precision_scores,
    "Average Pixel Accuracy": pixel_accuracies
}
