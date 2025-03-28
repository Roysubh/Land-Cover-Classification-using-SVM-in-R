# Load Required Libraries
library(terra)
library(lubridate)
library(sf)
library(dplyr)
library(readr)
library(e1071)
library(caret)
library(ggplot2)
library(RColorBrewer)

# --------------------------
# 🔹 Step 1: Load Raster and Training Data
# --------------------------

# File paths
fcc_path <- "D:/RStudio_3/SVM/AOI/LISS4_With_Indices.tif"
training_paths <- c("D:/RStudio_3/SVM/Traning_Sample/Builtup.shp",
                    "D:/RStudio_3/SVM/Traning_Sample/Trees.shp",
                    "D:/RStudio_3/SVM/Traning_Sample/Waterbody.shp",
                    "D:/RStudio_3/SVM/Traning_Sample/FallowLand.shp",
                    "D:/RStudio_3/SVM/Traning_Sample/CropLand.shp")

# Load raster
fcc_raster <- rast(fcc_path)

# Load training shapefiles
training_shp <- lapply(training_paths, vect)

# Plot RGB image with training data
plotRGB(fcc_raster, r = 1, g = 2, b = 3, stretch = "lin", main = "FCC with Training Data")

# Define colors
colors <- c("red", "gray", "green", "blue", "yellow")
for (i in seq_along(training_shp)) {
  plot(training_shp[[i]], add = TRUE, col = colors[i], lwd = 2)
}

# Add legend
legend("topright", legend = c("Builtup", "Trees", "Waterbody", "FallowLand", "CropLand"), 
       fill = colors, border = "black")

# --------------------------
# 🔹 Step 2: Extract Training Data from Raster
# --------------------------

# Output CSV path
output_csv <- "D:/RStudio_3/SVM/Traning_Sample_CSV/Extracted_Training_Data.csv"

# Class names
class_names <- c("Builtup", "Trees", "Waterbody", "FallowLand", "CropLand")
extracted_data_list <- list()

for (i in seq_along(training_paths)) {
  training_samples <- vect(training_paths[i])
  
  # Extract raster values for training points
  extracted_values <- extract(fcc_raster, training_samples)
  
  if (!is.null(extracted_values) && nrow(extracted_values) > 0) {
    df <- as.data.frame(extracted_values)
    
    # Check the number of columns before renaming
    if (ncol(df) == 15) {  
      colnames(df) <- c("ID", "NIR", "RED", "GREEN", "NG", "NR", "NNIR", 
                        "VIgreen", "DVI", "NDVI", "GNDVI", "NDWI", 
                        "OSAVI", "MSAVI2", "GEMI")
      df <- df %>% select(-ID)
    } else {
      print(paste("Warning: Unexpected number of columns in extracted data for class", class_names[i]))
      print(colnames(df))
      next
    }

    # Add serial number and class label
    df <- df %>%
      mutate(Sl_No = row_number(), Class = class_names[i]) %>%
      relocate(Sl_No)
    
    extracted_data_list[[i]] <- df
  } else {
    print(paste("No valid data extracted for class:", class_names[i]))
  }
}

# Combine extracted data
final_df <- bind_rows(extracted_data_list)

# Save to CSV
write_csv(final_df, output_csv)

print(paste("Training sample extraction complete. Data saved at", output_csv))

# Step 1: Load the training dataset
training_data <- read.csv("D:/RStudio_3/SVM/Traning_Sample_CSV/Extracted_Training_Data.csv")
training_data <- na.omit(training_data) 

# Convert Class to factor for classification
training_data$Class <- as.factor(training_data$Class)

# Step 2: Prepare training and testing datasets
set.seed(42)
trainIndex <- createDataPartition(training_data$Class, p = 0.8, list = FALSE)
trainSet <- training_data[trainIndex, ]
testSet <- training_data[-trainIndex, ]

# Step 3: SVM Model Training and Selection
kernels <- c("linear", "radial", "polynomial", "sigmoid")
svm_models <- list()
conf_matrices <- list()

best_model <- NULL
best_kernel <- NULL
best_accuracy <- 0

batch_size <- 2000
epochs <- 5
cv_folds <- 5

for (kernel in kernels) {
  cat("\n🔹 Training SVM with", kernel, "kernel...\n")
  for (epoch in 1:epochs) {
    cat("🔄 Epoch", epoch, "for", kernel, "kernel...\n")
    set.seed(epoch)
    trainSet <- trainSet[sample(nrow(trainSet)), ]
    num_batches <- ceiling(nrow(trainSet) / batch_size)
    for (batch in 1:num_batches) {
      batch_data <- trainSet[((batch - 1) * batch_size + 1):min(batch * batch_size, nrow(trainSet)), ]

      # Hyperparameter tuning
      tune_result <- tune.svm(Class ~ ., data = batch_data, 
                              kernel = kernel, 
                              cost = 2^(0:5), 
                              gamma = if (kernel != "linear") 2^(-5:0) else NULL,
                              tunecontrol = tune.control(cross = cv_folds))

      best_cost <- tune_result$best.parameters$cost
      best_gamma <- if ("gamma" %in% colnames(tune_result$best.parameters)) tune_result$best.parameters$gamma else NULL

      # Train SVM with best parameters
      svm_model <- if (kernel == "linear") {
        svm(Class ~ ., data = batch_data, kernel = kernel, cost = best_cost)
      } else {
        svm(Class ~ ., data = batch_data, kernel = kernel, cost = best_cost, gamma = best_gamma)
      }

      # Evaluate model
      svm_predictions <- predict(svm_model, testSet)
      conf_matrix <- confusionMatrix(svm_predictions, testSet$Class)
      accuracy <- conf_matrix$overall["Accuracy"]

      if (accuracy > best_accuracy) {
        best_accuracy <- accuracy
        best_model <- svm_model
        best_kernel <- kernel
      }
    }
  }
  svm_models[[kernel]] <- best_model
  conf_matrices[[kernel]] <- confusionMatrix(predict(best_model, testSet), testSet$Class)
  cat("✅ Best SVM with", kernel, "kernel trained. Accuracy:", round(best_accuracy, 4), "\n")
}

# Save the best SVM model
saveRDS(best_model, "D:/RStudio_3/SVM/SVM_Model/best_svm_model.rds")

cat("✅ Best SVM model saved successfully at: D:/RStudio_3/SVM/best_svm_model.rds\n")

# Set file paths
image_path <- "D:/RStudio_3/SVM/AOI/LISS4_With_Indices.tif"
svm_model_path <- "D:/RStudio_3/SVM/SVM_Model/best_svm_model.rds"
output_raster_path <- "D:/RStudio_3/SVM/Classified_Output/Classified_Image.tif"

# Load trained SVM model
load(svm_model_path)
cat("✅ SVM Model Loaded Successfully\n")

# Load raster image
raster_img <- rast(image_path)

# Convert raster to dataframe for prediction
raster_df <- as.data.frame(raster_img, xy = FALSE, na.rm = TRUE)

# Predict land cover classes using the trained SVM model
predicted_classes <- predict(best_model, raster_df)

# Convert predictions to raster
classified_raster <- raster_img[[1]] 
values(classified_raster) <- as.numeric(predicted_classes)

# Save classified raster
writeRaster(classified_raster, output_raster_path, overwrite = TRUE)
cat("✅ Classified image saved at:", output_raster_path, "\n")
