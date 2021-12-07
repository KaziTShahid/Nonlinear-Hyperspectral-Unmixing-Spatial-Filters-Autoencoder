# Nonlinear-Hyperspectral-Unmixing-Spatial-Filters-Autoencoder

This is the code for an autoencoder that performs nonlinear pixel unmixing on hyperspectral images (based on the Fan, Bilinear, PPNM models, and also higher order nonlinear terms), that can have corrupted pixels that have 0 value. This utilizes spatial information by implementing a weighted averaging filter based on RBF distances.

The three mixing models used here are the Fan, Bilinear and PPNM models.

This script can also estimate the number of endmembers, which occurs prior to the implementation of the autoencoder network.

Also, you can have higher order nonlinear terms, like 3rd or 4th degree cross-products instead of only upto the 2nd degree cross products. For this, change the "upto_how_many_degrees" parameter.

# Main Files

"autoencoder_main.py" is the main file to run. It uses one dataset currently, the "PaviaU" dataset. If you want to add more datasets, download more from the link below
http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
Once you download other datasets, make sure to include the filenames in the "dataset_choices" variable.

"rbf_filter_kazi.py" is the first layer, creating mixed pixels that are a weighted average of itself and its surrounding neighborhood.

"rbf_kazi.py" is the second layer, finding abundances.

"nonlin_layer_kazi.py" unmixes according to the Fan or the Bilinear model.

"ppnm_layer_kazi.py" unmixes according to the PPNM model. The model to choose will come from the "mixing_models" variable.

# Citation

If you wish to use this code, please cite the link where the datasets come from.

http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

Also, cite the link where this code was from

https://github.com/KaziTShahid/Nonlinear-Hyperspectral-Unmixing-Spatial-Filters-Autoencoder

Also, please cite the paper published, which can be found in this link

https://ieeexplore.ieee.org/abstract/document/9633169
