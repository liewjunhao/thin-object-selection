#!/bin/bash

# Download COIFT results
gdown https://drive.google.com/uc?id=1Hgj4814fo1EGzpfXbPD9gJQI-YEQiFtS
# Download HRSOD results
gdown https://drive.google.com/uc?id=17NBG0BWOuK1674afTG09G5FBv0IMeflu
# Download ThinObject-5K results
gdown https://drive.google.com/uc?id=1Hiwi7hXFIifMis4q9YOwj1fa2K6OAHZk

# Unzip 
unzip results_coift.zip
unzip results_hrsod.zip
unzip results_thinobject5k_test.zip

# Clean
rm results_coift.zip
rm results_hrsod.zip
rm results_thinobject5k_test.zip
