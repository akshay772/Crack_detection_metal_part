# Crack_detection_metal_part
  Due to computation limitation, images if trimmed to 512x512 pixels.
## Install requirements
  `pip install -r requirements.txt`

## Run the program
  ### Some preprocessing to image 
  #### Structure of folders : 
  * Dataset Directory containing "normal" and "defect" folders
    * dataset ---> "/YE358311_defects/YE358311_Crack_and_Wrinkle_defect/"
    * dataset ---> "/YE358311_Healthy/"
  * Destination Directory containing "train" and "test" folders 
    * data ----> train ----> {"normal", "defect"} subfolders
    * data ----> test ----> {"normal", "defect"} subfolders
  `python3 image_preprocess.py "Path_to_dataset_directory" "Path_to_destination_directory"`
  
  ### Move some files in train folders to test folder for validation set
  * Run for each folder in train ie, normal & defect ( -n 30 denotes random 30 files)
    `find ./data/train/normal -type f -name "*.jpg" -print0 | xargs -0 shuf -e -n 30 -z | xargs -0 cp -vt ./data/test/normal`
    `find ./data/train/defect -type f -name "*.jpg" -print0 | xargs -0 shuf -e -n 30 -z | xargs -0 cp -vt ./data/test/defect`
  ### Training a simple CNN classifier (3 Conv + 1 FC)
  `python3 CNNclassifier.py`
  
## Accuracy Metrics 
  ## Validation accuracy and loss : *val_loss: 1.7570 - val_acc: 0.4286*

## Need for Improvements
* Image Preprocssing/Data Preparation
  * Since cracks are of less area as compared to image and noise, will introduce dropouts to improve accuracy
  * Preparing a object extraction module (For removing the rest of noise to improve accuracy)
  * Preparing a background color update module (For easy extraction of metal part after grayscale conversion)
* Trainng Improvements
  * Using a pretrained model such as VGG16 trained on ImageNet
  * Using model ensembles such as CNN+SVM (rbf kernel), Gauss filter+LBP+SVM(rbf kernel) etc that have proved improving accuracy
