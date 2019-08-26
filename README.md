# Crack_detection_metal_part

## Install requirements
  `pip install -r requirements.txt`

## Run the program
  ### Some preprocessing to image 
  #### Structure of folders : 
  * Dataset Directory containing "normal" and "defect" folders
    `dataset ---> normal
             ---> defect'
  * Destination Directory containing "train" and "test" folders 
  `python3 image_preprocess.py "Path_to_dataset_directory" "Path_to_destination_directory"`
  ### Training a simple CNN classifier (3 Conv + 1 FC)
  `python3 CNNclassifier.py`
  
## Accuracy Metrics 
  ## Validation accuracy and loss : *val_loss: 1.7570 - val_acc: 0.4286*

## Need of Improvements
* Image Preprocssing/Data Preparation
  * Preparing a object extraction module (For removing the rest of noise to improve accuracy)
  * Preparing a background color update module (For easy extraction of metal part after grayscale conversion)
* Trainng Improvements
  * Using a pretrained model such as VGG16 trained on ImageNet
  * Using model ensembles such as CNN+SVM (rbf kernel), Gauss filter+LBP+SVM(rbf kernel) etc that have proved improving accuracy
