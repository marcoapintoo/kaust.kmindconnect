<template>
    <div>
        <markdown-element>

# Dynamic Brain Connectivity
# SVAR/TVVAR Model

## Introduction

The typical approach (sliding-window correlation) does not describe the dynamic behavior due to non-stationary of the brain signals appropriately.

There is a need to develop interactive packages that implement dimension reduction and state-of-the-art statistical modeling of fMRI time series data.

## Contributions of the method

1. To study the feasibility of using a _regime-SVAR model_ for identification of dynamic connectivity states on mental activities that involves several stimuli while watching a movie.
1. To _recognize brain states_ based on the fMRI connectivity patterns and test the efficiency of _connectivity-state_ detection.
1. To create a software package (with a graphical user interface) to find connectivity networks and estimate brain states in stimuli-complex experiments.

## Model description

### Functional description

> ![](../docs/models/svar/method.png)

### Time-varying VAR model

Generalization of the Vector Autoregressive (VAR) model where $Y_t$ could be expressed in terms of past data $Y_{t-1},\ldots Y_{t-P}$ :

$$
Y_{t}=\sum_{\ell=1}^{P}\Phi_{\ell t}Y_{t-\ell}+v_{t}
$$

* $\Phi_{\ell t}$: $N\times N$ matrix of VAR coefficients 
at a lag $\ell$ and time $t$.\\\vspace{1mm}

## Coefficients clusterization

K-means was used to find clusters of time points (related with the TV-VAR coefficients)
in a $L_1$ neighborhood.
Also to reduce the influence of the initialization values, it was also added a VNS algorithm (variable search neighborhood) to find stable groups.

### Switching VAR model
Quasi-stationary model composed of a
set of $K$ independent VAR processes:

$$
Y_{t}=\sum_{\ell=1}^{P}\Phi_{\ell\left[S_{t}\right]}Y_{t-\ell}+v_{t}
$$

* $\Phi_{\ell\left[S_{t}\right]}$: VAR coefficient matrix
for the lag $\ell$ and the state $j$,
* $S_{t}$: sequence of indicators defined over $j=1,\ldots,K$ for each time $t$.

### Kalman filtering and smoothing
Method to identify and estimate hidden, non-measurable, brain $X_{t}$ and the subsequent state sequence $S_{t}$ using a Bayesian inference.
The signal is filtered through iterative calculus of the mean and covariance of $X_{t}|Y_{1:t},S_{t}=j$ (for the filtering process), and $X_{t}|Y_{1:T},S_{t}=j$ (for the smoothing process.)

### Expectation-Maximization Algorithm
An iterative EM algorithm is used to estimate the MLE of the parameters $\Theta$ of the model:

$$
\Theta = \left\{\Theta_{[j]}\right\}\qquad\Theta_{[j]}=\left(A_{[j]},Q_{[j]},R_{[j]}\right)
$$

* $A_{[S_{t}]}$: describes the effective connectivity across states,\\\vspace{-1.2em}
* $R_{[S_{t}]}, Q_{[S_{t}]}$: covariance in the canonical linear gaussian state-space.

# References

  [1] Ting, C. M., Ombao, H., Balqis Samdin, S., and Salleh, S. H. (2017). Estimating Time-Varying Effective Connectivity in High-Dimensional fMRI Data Using Regime-Switching Factor Models. arXiv preprint arXiv:1701.06754.

  [2] Wang, Y., Ting, C. M., and Ombao, H. (2016). Modeling effective connectivity in high-dimensional cortical source signals. IEEE Journal of Selected Topics in Signal Processing, 10(7), 1315-1325.

  [3] Eavani, H., Satterthwaite, T. D., Filipovych, R., Gur, R. E., Gur, R. C., and Davatzikos, C. (2015). Identifying sparse connectivity patterns in the brain using resting-state fMRI. Neuroimage, 105, 286-299.

        </markdown-element>
    </div>
</template>

<script>

var ApplicationSVARModelDocumentation = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    data(){
        return {
        }
    },
    methods: {
        /*compileDocumentMarkdown: _.debounce( (e) => {
            this.helpDocument = marked(e.target.value, { sanitize: true })
        }, 300),*/
    },
    computed: {
    }
}
Vue.component('app-svar-model-documentation', ApplicationSVARModelDocumentation)

</script>
