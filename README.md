# arabian-anisotropy
Scripts I wrote for my masters project investigating seismic anisotropy of the Arabian Plate.
My scripts prepare the data for input into the tomographic code of [Debayle and Sambridge](http://perso.ens-lyon.fr/eric.debayle/doc_DS2004.html), and plot the results.
## 1. formatter.py
- functions for generating station files from raw data
- generating .rayl files for tomo code
- plotting histograms of group velocities before and after tomographic inversion
## 2. plot.py
- plots the arabian plate with contours of isotropic velocity, and overlain with a quiver plot of anisotropy
## 3. plot_stations.py
- plots the distribution of stations
