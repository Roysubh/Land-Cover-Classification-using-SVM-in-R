library(terra)
library(lubridate)
library(sf)
library(dplyr)
library(readr)

input_path <- c("D:/RStudio_3/RF_1/Input/BAND2.tif",
                "D:/RStudio_3/RF_1/Input/BAND3.tif",
                "D:/RStudio_3/RF_1/Input/BAND4.tif")  
output_path <- "D:/RStudio_3/RF_1/Reflected_Input_Image/"  

metadata <- list(
  DateOfPass = "09-MAR-2024",
  SunElevationAtCenter = 49.942622,
  B2_Lmax = 52.0,
  B3_Lmax = 47.0,
  B4_Lmax = 31.5
)

acq_date <- dmy(metadata$DateOfPass)

sun_elevation_angle <- metadata$SunElevationAtCenter
sun_zenith_angle <- 90 - sun_elevation_angle
sun_zenith_angle_rad <- sun_zenith_angle * pi / 180  

d <- 1 + 0.01672 * cos((2 * pi / 365.25) * (yday(acq_date) - 4))

b2_sr <- metadata$B2_Lmax
b3_sr <- metadata$B3_Lmax
b4_sr <- metadata$B4_Lmax

b2_esun <- 181.89
b3_esun <- 156.96
b4_esun <- 110.48

scene <- rast(input_path)

b2_rad <- (scene[[1]] * b2_sr) / 1024
b3_rad <- (scene[[2]] * b3_sr) / 1024
b4_rad <- (scene[[3]] * b4_sr) / 1024

b2_ref <- (pi * b2_rad * d^2) / (b2_esun * cos(sun_zenith_angle_rad))
b3_ref <- (pi * b3_rad * d^2) / (b3_esun * cos(sun_zenith_angle_rad))
b4_ref <- (pi * b4_rad * d^2) / (b4_esun * cos(sun_zenith_angle_rad))

writeRaster(b2_ref, paste0(output_path, "B2_TOA.tif"), overwrite = TRUE)
writeRaster(b3_ref, paste0(output_path, "B3_TOA.tif"), overwrite = TRUE)
writeRaster(b4_ref, paste0(output_path, "B4_TOA.tif"), overwrite = TRUE)

print(paste("TOA reflectance files saved in", output_path))

toa_stack <- c(b4_ref, b3_ref, b2_ref)

fcc_output <- paste0(output_path, "FCC_Composite_TOA.tif")
writeRaster(toa_stack, fcc_output, overwrite = TRUE)

print(paste("FCC image saved at", fcc_output))

plotRGB(toa_stack, r = 1, g = 2, b = 3, stretch = "lin", main = "False Color Composite (FCC)")

aoi_path <- "D:/RStudio_3/RF/AOI/AOI.shp" 

aoi <- vect(aoi_path)

clipped_fcc <- crop(toa_stack, aoi, mask = TRUE)

clipped_output <- paste0("D:/RStudio_3/RF/AOI/", "Clipped_FCC.tif")
writeRaster(clipped_fcc, clipped_output, overwrite = TRUE)

print(paste("Clipping complete. Clipped FCC saved at", clipped_output))

plotRGB(clipped_fcc, r = 1, g = 2, b = 3, stretch = "lin", main = "Clipped False Color Composite (FCC)")

plotRGB(clipped_fcc, r = 1, g = 2, b = 3, stretch = "lin", main = "Clipped False Color Composite (FCC)")

b4_clipped <- clipped_fcc[[1]]  # NIR (Band 4)
b3_clipped <- clipped_fcc[[2]]  # Red (Band 3)

ndvi <- (b4_clipped - b3_clipped) / (b4_clipped + b3_clipped)

names(ndvi) <- "NDVI"

ndvi_output <- paste0("D:/RStudio_3/RF/AOI/", "Clipped_NDVI.tif")
writeRaster(ndvi, ndvi_output, overwrite = TRUE)

print(paste("NDVI computation complete. NDVI saved at", ndvi_output))

fcc_ndvi_stack <- rast(list(clipped_fcc, ndvi))

names(fcc_ndvi_stack) <- c("NIR", "Red", "Green", "NDVI")

stacked_output <- paste0("D:/RStudio_3/RF/AOI/", "Clipped_FCC_NDVI.tif")
writeRaster(fcc_ndvi_stack, stacked_output, overwrite = TRUE)

print(paste("FCC with NDVI as Band 5 saved at", stacked_output))

plot(ndvi, col = terrain.colors(100), main = "NDVI")

print("NDVI visualization complete.")