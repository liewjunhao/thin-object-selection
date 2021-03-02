#!/bin/bash

# Download COIFT
gdown https://drive.google.com/uc?id=1SapWK3yX2utscH8e-cJzQivp_Qp7IBQf
# Download HRSOD
gdown https://drive.google.com/uc?id=1dLZTzQuDEuBICKtaRB03i1Mx9hOssxC4
# Download ThinObject-5K
gdown https://drive.google.com/uc?id=1yUSNCOPwbkEyrQYr5d2OtNjRMEbuQp7O
# Download pre-computed thin regions
gdown https://drive.google.com/uc?id=1EJS63xPOk04ZOihcWCIM80JulHZTTzLY

# Unzip 
unzip COIFT.zip
unzip HRSOD.zip
unzip ThinObject5K.zip
unzip thin_regions.zip

# Clean
rm COIFT.zip
rm HRSOD.zip
rm ThinObject5K.zip
rm thin_regions.zip