🛰️ Land Cover Classification using SVM in R

This repository provides a complete workflow for classifying land cover using Support Vector Machine (SVM) in R. The project processes satellite imagery from LISS-IV and applies machine learning techniques to classify various land cover types.

📖 Overview:
  Land cover classification is essential for monitoring environmental changes, urban planning, and agricultural assessments. This project leverages machine learning (SVM) to classify land cover types using high-resolution LISS-IV satellite imagery. The workflow includes data preprocessing, model training, and classification, ensuring accuracy and efficiency.

🔹 Features:
  1. Automated data extraction: Extracts pixel values from training shapefiles.
  2. Machine Learning with SVM: Applies Support Vector Machine classification.
  3. Hyperparameter tuning: Optimizes the model using cross-validation.
  4. Raster Classification: Assigns land cover labels to satellite imagery.
  5. Easy visualization: Saves classified images for GIS analysis.

🔹 Step 1: Prepare Training Data
  1.1 Create Training Shapefiles: To classify land cover, first, create training samples as shapefiles using GIS software like QGIS or ArcGIS. Each shapefile should contain representative points or polygons for different land cover classes:
    1. Built-up (Urban areas, roads, buildings)
    2. Trees (Forests, vegetation)
    3. Waterbody (Lakes, rivers, reservoirs)
    4. Fallow Land (Unused land)
    5. Crop Land (Agricultural fields)

Ensure that all shapefiles are in the same coordinate reference system (CRS) as the input raster.
  
  1.2 Extract Training Data from Raster: Using the script, extract pixel values from the input raster corresponding to training sample locations. These values are stored in a CSV file (Extracted_Training_Data.csv), which will be used for model training.
  
  I have done some indices like "NIR", "RED", "GREEN", "NG", "NR", "NNIR", "VIgreen", "DVI", "NDVI", "GNDVI", "NDWI", "OSAVI", "MSAVI2", "GEMI" for better results when converting LISS4 image into TOA product.

🔹 Step 2: Train the SVM Model
  2.1 Load and Preprocess Data
      The script reads the extracted training data from the CSV file.
      It removes missing values and converts the class labels into factors for classification.
  
  2.2 Split Data for Training and Testing
      The dataset is split into 80% training and 20% testing.
      The caret package is used to create a balanced split.

  2.3 Train SVM Model
      The script trains multiple SVM models with different kernels (linear, radial, polynomial, sigmoid).
      Hyperparameter tuning is performed using cross-validation.
      The best-performing model is selected and saved as best_svm_model.rds.
      
🔹 Step 3: Apply SVM Model to Classify New Image
  3.1 Load the Trained SVM Model
      The trained model is loaded from best_svm_model.rds.

  3.2 Predict Land Cover for a New Image
      The LISS-IV raster image is loaded.
      Pixel values are extracted and passed through the trained SVM model.
      Predictions are assigned to raster pixels, creating a classified image.

  3.3 Save Classified Image
      The classified raster is saved as Classified_Image.tif, displaying different land cover classes.
      
📦 Dependencies: 
      Ensure you have the following R packages installed:
        install.packages(c("terra", "sf", "lubridate", "dplyr", "readr", "e1071", "caret", "ggplot2", "RColorBrewer"))

🚀 Running the Classification: 
      This will process the input image, train the SVM model, classify the image, and save the output.



🎯 Results:
      The final classified image Classified_Image.tif contains land cover classes assigned to each pixel. The output can be visualized in QGIS or R using:

🏁 Conclusion:
      This project demonstrates the effectiveness of Support Vector Machines (SVM) in classifying land cover from high-resolution satellite imagery. By following this workflow, users can generate accurate land cover maps and improve environmental monitoring efforts. Future improvements may include experimenting with different spectral indices, advanced feature selection, and deep learning-based classification.

📌 Notes:
      Ensure the training shapefiles are well-distributed and representative.
      Modify hyperparameters for better classification accuracy.
      Experiment with different spectral indices for improved classification.
