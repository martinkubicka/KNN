# Dataset preparation 

To prepare the dataset use script `create_dataset.py` in the `src/dataset_preparation` folder.
## Overview
The script takes the following arguments:

- **Source Database (`--source_db`)**: Defines the path to the original LMDB image database.  
  Default: `lmdb.hwr_40-1.0`
- **Target Database (`--target_db`)**: Specifies the path to the augmented LMDB image database ready for training.  
  Default: `lmdb_processed`
- **ID CSV (`--id_csv`)**: Points to a CSV file containing all image IDs in original database.  
  Default: `lines.filtered_max_width.all`
- **Target CSV (`--target_csv`)**: Filename for the CSV file that will store updated/processed image IDs. It will serve as a base name for the csv file train and test.
  Default: `after_processing.csv`

Furthermore, you can adjust default arguments in the class `ImageProcessor` in `image_processor.py.`:
## Parameters

- **`target_width`** *(default: 1024)*  
  The target width for the processed images.  

- **`target_height`** *(default: 48)*  
  The target height for the processed images.  

- **`grayscale`** *(default: True)*  
  Specifies whether the train images should be converted to grayscale during processing.  

- **`min_width`** *(default: 256)*  
  The minimum width for images; if the original width is below this value, and its the last image from unique author, it will be omitted.  The reason it to prevent small images with large zero padding.

## Specifying test sample size 

- **`test_sample_max_img`** *(default: 20000)*  
  Defines the maximum number of images to include in the test sample for evaluation.  

- **`test_sample_max_authors`** *(default: None)*  
  Limits the number of authors to include in the test sample if specified. 