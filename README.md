# Medical Image Segmentation Kaggle Competition
## Overview
This is project aims to delineate regions of interest within CT scans based on the ["A02025-Medical-Image-Segmentation" kaggle competition](https://www.kaggle.com/competitions/a0-2025-medical-image-segmentation/overview). We augment(rotations, flips) the data and use to train a 4-layer U-Net optimized with a 50/50 combined BCE/DICE loss. A final post-processing step is used to keep only the largest connected region and to fill holes. The final submitted version performs with a DICE score of 0.8702, while the top contributor on Kaggle achieved 0.9024. If I have more time, some potential improvements include (1) further augmentation of data (random crops, elastic deformation, modifications in contrast), (2) More training with adaptive learning rate, (3) stronger regularization, and (4) preprocessing to enhance channels with the most contrast.

## Project Details

In the first part of the notebook we import the training data and masks to get a sense of the distribution of the image sizes and aspect ratios. 

We then define transformations for the input data to prepare for the UNet, including `ResizeAndPad` which generates images and masks with consistent dimensions, keeping track of metadata to allow transformation back to original dimensions. `TrainTransform` and `EvalTransform` are applied to training and validation sets, respectively, with `Augment` only being applied to the former. 

The `UNet` model follows the standard symmetric encoder/decoder sequence with four layers before/after the bottleneck. Dropout regularization is applied at the bottleneck. A combined `DiceBCELoss` is defined for model optimization and `evaluate_model` is used to track the DICE score on the validation set as a function of epoch and saves representative images compared to the ground truth for reference. `train_model` saves dictionary of training set DICE loss, BCE loss, DICEBCE loss, and validation DICE for each epoch, saving the model weights when a new maximimum validation DICE is achieved. 

`postprocess_mask` is applied to predicted masks to keep only the largest connected region and to fill holes. Finally, test data are evaluated and converted to RLE format for submission to Kaggle. 

## Files

**Medical Image Segmentation.ipynb**: project code

**Dataset**: contains Training and Test data

**best_unet_model.pth**: weights for best performing model (can be loaded in latter part of notebook)

**val_outputs**: Contains images, predicted, and ground truth masks for select validation data for each epoch

**Predicted_Masks**: Predicted masks for test dataset
