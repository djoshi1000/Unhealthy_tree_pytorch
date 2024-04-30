import json, os

# Load the original config.json
with open(r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\Config\config_pca_deeplab.json", 'r') as f:
    config = json.load(f)

# Define configurations for different scenarios
data_directories = os.listdir(r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\data")[1:]
save_base_directory = r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\rough"

# Modify and save configurations
for data_dir in data_directories:
    new_config = config.copy()
    models= ['UNetResnet', 'DeepLab','SegResNet']
    for model in models:
        new_config['arch']['type'] = model
        new_config['arch']['args']['freeze_bn'] = False
        new_config['name'] = model
        if model =="DeepLab":
            new_config['optimizer']['differential_lr']= True
        else:
            
            new_config['optimizer']['differential_lr']= True
        new_config['train_loader']['args']['data_dir'] = f'./data/{data_dir}'
        new_config['val_loader']['args']['data_dir'] = f'./data/{data_dir}'
        folder_name = data_dir.split('/')[-1]
        print(folder_name)
        new_config['trainer']['save_dir'] =  f'saved/{folder_name}_new' if any(folder_name in data_dir for folder_name in data_directories) else ''
        filename = f"config_{folder_name.split('_')[-1]}_{new_config['name']}.json"
        save_path = os.path.join(save_base_directory, filename)
        print(save_path)
        with open(save_path, 'w') as f:
            json.dump(new_config, f, indent=4)

print("Configs saved successfully.")



with open(r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\config_aug_seg.json", 'r') as f:
    config = json.load(f)

# Define configurations for different scenarios
data_directories = os.listdir(r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\data")[1:]
save_base_directory = r"D:\Chapter_2\New_idea\Unet_plu\UNetPlusPlus\pytorch-segmentation\rough"

# Modify and save configurations
for data_dir in data_directories:
    new_config = config.copy()
    models= ['UNetResnet', 'DeepLab','SegResNet']
    for model in models:
        new_config['arch']['type'] = model
        new_config['arch']['args']['freeze_bn'] = False
        new_config['name'] = model
        if model =="DeepLab":
            new_config['optimizer']['differential_lr']= True
        else:
            
            new_config['optimizer']['differential_lr']= True
        new_config['train_loader']['args']['data_dir'] = f'./data/{data_dir}'
        new_config['val_loader']['args']['data_dir'] = f'./data/{data_dir}'
        folder_name = data_dir.split('/')[-1]
        print(folder_name)
        new_config['trainer']['save_dir'] =  f'saved/{folder_name}_aug' if any(folder_name in data_dir for folder_name in data_directories) else ''
        filename = f"config_{folder_name.split('_')[-1]}_{new_config['name']}_aug.json"
        save_path = os.path.join(save_base_directory, filename)
        print(save_path)
        with open(save_path, 'w') as f:
            json.dump(new_config, f, indent=4)

print("Configs saved successfully.")
