# Classification, weight sharing, auxiliary losses
Project 1 of Deep Learning course (EE559) at EPFL,

_by Jalel Zgonda, Jonathan Labhard, Robin Zbinden_


The goal of this project is to test different architectures to compare two digits visible in a two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an auxiliary loss to help the training of the main objective. More about this project can be read in the `report.pdf` file.

## Usage

Run the script `test.py` to test our model with:
```
python test.py
```   
    
## Detailed file description

`EnhancedSiamese.py` defines the enhanced model

`NaiveSiamese.py` defines the naive model

`Nets.py` contains the two networks, i.e., `CompNet` and `DigitNet`

`dlc_practical_prologue.py` contains the functions to generate the data

`helpers.py` contains useful helpers functions

`test.ipynb` is a script to test our model
