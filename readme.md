
#  Predicting-asthma-from-Early-life-Cumulative-indoor-and-outdoor-exposures
Pachage and code for the paper Predicting-asthma-from-Early-life-Cumulative-indoor-and-outdoor-exposures.
There is implimentation of nested cross validation for calibrated random forest and a hererical bayesian logistic regression useing sklearn, numpyro  and jax.
To install the needed eviroment the yml file is there, the jax can both run on cpu and gpu so install the pachage after your needs.
```
conda env create -f clinical_regression.yml
```
## Data
The data is a synthetic version of the Original COPSAC data(does not give same result). It has the household ID and the measurement of the house where it was taken.
There are multiple indoor and outdoor sources and sinks. The indoor is the number of days the kids are exposed to the source.
The outdoor sources are the total area of the outdoor source as an approximation for the total exposure rate.
The synthetic data was created since it is clinical data, which doesn't allow us to open-source it. 
The data is mostly there to showcase the application and that the code is working.

Example code for the paper is in Examples folder

Plots from running the code get send to the Plots folder 

Models is in the Models folder

Performence of the Models and the pickle files are in Performence  folder.
The pickle files structure can easierly be seen after running one of the example data.



## Pipeline 
![plot](https://github.com/MichaelForsmann/Predicting-asthma-from-Early-life-Cumulative-indoor-and-outdoor-exposures/blob/main/Plots/Untitled-Diagram.drawio(1).png)
## Plots and performence to estimate general function for seasonality of the gases and particles
![plot](https://github.com/MichaelForsmann/Predicting-asthma-from-Early-life-Cumulative-indoor-and-outdoor-exposures/blob/main/Plots/arviz_adjusted.png)
![plot](https://github.com/MichaelForsmann/Predicting-asthma-from-Early-life-Cumulative-indoor-and-outdoor-exposures/blob/main/Plots/shap_12_adjusted.png)
### References 
- [Pyro: bingham2019pyro:](https://arxiv.org/abs/1810.09538) 
- [Scipy: 2020SciPy-NMeth:](https://www.nature.com/articles/s41592-019-0686-2)
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library]( https://openreview.net/forum?id=BJJsrmfCZ)
- [arviz: arviz_2019:](https://joss.theoj.org/papers/10.21105/joss.01143)
- [Kennard-stone](https://www.researchgate.net/publication/357491012_Kennard-Stone_method_outperforms_the_Random_Sampling_in_the_selection_of_calibration_samples_in_SNPs_and_NIR_data)
