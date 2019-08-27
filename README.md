# Crack_detection_metal_part
  A python flask app running on local host. 
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
    
  ### Training a simple CNN classifier (3 Conv + 1 FC)
  * `python3 main.py` --- Start the application server(local)
  * `http://127.0.0.1:5000/crack_detection_train` --- Train the model
  * `http://127.0.0.1:5000/crack_detection_test` --- Opens an hmtl to upload the image and test for cracks
  
## Accuracy Metrics 
  Validation accuracy and loss
  * Training loss : 0.1695
  * **Training accuracy** : 100.0%
  * Validation/Test loss : 0.3975 
  * **Validation/Test accuracy** : 81.1%
  
## Need for Improvements
* Image Preprocssing/Data Preparation
  * Since cracks are of less area as compared to image and noise, will introduce dropouts to improve accuracy
  * Preparing a object extraction module (For removing the rest of noise to improve accuracy)
  * Preparing a background color update module (For easy extraction of metal part after grayscale conversion)
* Trainng Improvements
  * Using a pretrained model such as VGG16 trained on ImageNet
  * Using model ensembles such as CNN+SVM (rbf kernel), Gauss filter+LBP+SVM(rbf kernel) etc that have proved improving accuracy
