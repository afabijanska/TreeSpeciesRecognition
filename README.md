# Tree Species Recognition

## Intro

This crepository contains source code of a convolutional neural network for wood species recognition from wood core images.

## Dataset 

To download dataset of wood core images please follow: http://an-fab.kis.p.lodz.pl/dendrodataset/DendroDataset.7z (425 MB). The dataset contains following tree species:

    Brzoza / Betula sp. / Birch
    Buk zwyczajny / Fagus sylvatica L. / European beech
    Dab / Quercus sp. / Oak
    Grab pospolity / Carpinus betulus L. / European hornbeam
    Klon jawor / Acer pseudoplatanus L. / Sycamore maple
    Jesion wyniosły / Fraxinus excelsior L. / English ash
    Jodla / Abies alba 
    Lipa / Tilia sp. / Lime
    Modrzew europejski / Larix decidua L. / European larch
    Olsza / Alnus sp. / Alder
    Sosna zwyczajna / Pinus sylvestris L. / Scots pine
    Swierk / Picea abies
    Wiąz / Salix sp. / Willow 
    Wierzba / Salix sp. / Willow
    
# Running the code

## Prerequisites

Python 3.6, Tensorflow, Keras, Anaconda3

## Repository content

<ul>
  <li> <b>configuration.txt</b><br> -- file to be edited; <br> -- contains data paths and train/test setings;   
  <li> <b>GetFolds.py</b><br> -- script for dividing dataset into a number of folds given in the configuration code; <br> -- to be run 1st; <br> -- expects that image dataset is stored in a directory with subdirectory for each species<br> -- generates (number of folds) directories, each containing wood core images selected to a given fold; <br> -- copies subdirectory structure from the directory containing original dataset;  
  <li> <b>DGenerateData2.py</b><br> -- script for preparing train data; <br> -- to be run 2nd; <br> -- samples a number of patches from train folds (test fold is given in the config file, remaining ones are considered train folds); <br> -- saves sampled train patches as hdf5 file (defined in config file) 
  <li> <b>TrainNetwork.py</b><br> -- script for training the model with the train patches and the corresponding labels; <br> -- to be run 3rd; <br> -- contains model definition;
  <li> <b>PredictEvaluate.py</b><br> -- script for testing the model at the patch level; <br> -- to be run 4th;
  <li> <b>PredictEvaluate2.py</b><br> -- script for testing the model at the core level; <br> -- to be run 5th;
  <li> <b>helpers.py</b><br> -- some helper functions;
</ul>

# Contact

<b>Anna Fabijańska</b><br>
Institute of Applied Computer Science<br>
Lodz University of Technology<br>
e-mail: anna.fabijanska@p.lodz.pl<br>
WWW: http://an-fab.iis.p.lodz.pl<br>
