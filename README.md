# LakePIAB
Authors: Robert Ladwig, Arka Daw, Cal Buelo, Abhilash Neog

LakePIAB (Lake Physics in a Box) - a Modular Compositional Learning framework combining process-based model formulations and deep learning to simulate 1D vertical water temperature dynamics. Preprint highlighting the Modular Compositional Learning methodology and its performance is available [here](https://doi.org/10.22541/essoar.169143862.25982294/v1).

$A \frac{\partial T}{\partial t}=\frac{\partial}{\partial z}(A K_z \frac{\partial T}{\partial z}) + \frac{1}{{\rho_w c_p}}\frac{\partial H(z)}{\partial z}  + \frac{\partial A}{\partial z}\frac{H_{geo}}{\rho_w c_p}$

See also the software and data release at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10063835.svg)](https://doi.org/10.5281/zenodo.10063835).

## File structure
\src includes scripts for running the models:
- run_ProcessModel.py runs the process-based model for generating training data
- run_DeepModel_noModule.py run the pretrained deep learning model (no modularisation)
- run_DeepModel_noProcess.py runs the deep learning model (no process)
- run_HybridModel.py runs the hybrid MCL models
- run_***_collapse.py runs the models to explore stability perturbations
- processBased_lakeModel_functions.py includes model source code
- oneD_HeatMixing_Functions.py incldues ancillary model source code

Additionally, this folder includes R-scripts to analyse the model outputs
- physicalCalculations.R includes code to explore the performance of the hybrid MCL model
- stability.R includes code to highlight perturbations by the hybrid MCL model

\output and \stability include model output files

\figs includes final manuscript figures

\input includes input data

\MCL includes the Jupyter notebooks for model pretraining and fine-tuning

## Important
Note that for running the scripts you need to decompress the zip files in \input and \MCL\02_training


<a href="url"><img src="logo.png" width=70% height=70% ></a>
