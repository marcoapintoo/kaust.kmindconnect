
# kMindConnect

kMindConnect is a software package developed with the aim of offering seamless integration between different the Brain dynamic connectivity modelling and the visualization results. The toolbox provides an easy way to present the dynamic changes in the estimated Brain states (or connectivity.) The default model to perform this estimation inside the application is the switching-regime vector autoregressive model (S-VAR) that is fitted over clustered time-varying vector autoregressive (TV-VAR) coefficients from the data input [Ombao et al., 2018] [Ting et al., 2018] [Pinto et al., 2018][Ting et al., 2016].

![](ui/docs/gui/screen.png)

> Ombao H, Fiecas M, Ting CM and Low YF. (2018). Statistical Models for Brain Signals with Properties that Evolve Across Trials. NeuroImage, In Press.

> Ting CM, Ombao H and Sh-Hussein. (2018). Estimating Dynamic Connectivity States in fMRI Using Regime-Switching Factor Models. IEEE Transactions on Medical Imaging, Accepted for Publication.

> Pinto M, Ting CM and Ombao H. (2018). KMindConnect: A software for modeling brain dynamic connectivity. Journal of Statistical Software, In Preparation.

> Ting, C. M., Samdin, S. B., Ombao, H., & Salleh, S. H. (2016). A Unified Estimation Framework for State-Related Changes in Effective Brain Connectivity.


The source code is available in two repositories under the specified licenses:

* [github.com/marcoapintoo/kaust.kmindconnect](https://github.com/marcoapintoo/kaust.kmindconnect/): For major versions and the binary installers;

* [bitbucket.org/tonitruum/kaust.kmindconnect](https://bitbucket.org/tonitruum/kaust.kmindconnect/): For continuous development.

# Features

## Timeline of states

![](ui/docs/gui/states.png)

## Visual animation

![](ui/docs/gui/animation.png)

## Simulation timing

![](ui/docs/gui/tracking.png)


# Requirements

## Preparing the environment on Ubuntu/Debian


* Step 1: **Install Octave**
```
    sudo apt install octave octave-signal octave-statistics
```
* Step 2: **Install Anaconda/Python3**
* Step 3: **Install additional Python packages**
```
    pip install oct2py 
    conda install plotly
    conda install docopt
```
* Step 4: **Install node and electron** (*Only for building the software*)
```
    sudo apt install node
    sudo npm install -g electron --unsafe-perm=true --allow-root
```

# Credits

A project of the KAUST Biostatistics Group ([biostats.kaust.edu.sa](https://biostats.kaust.edu.sa/Pages/Home.aspx)).

* **Lead Software Developer**: Marco Pinto

* **PI**: Hernando Ombao 

* **Members**: Chee-Ming Ting and Marco Pinto

# License

Licensed under either

* GPL license, Version 3.0, (GPL-LICENSE or https://opensource.org/licenses/GPL-3.0)
* MIT license (MIT-LICENSE or http://opensource.org/licenses/MIT)

