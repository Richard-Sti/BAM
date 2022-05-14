# empiricalgalo: Empirical models of the galaxy-halo connection

Collection of empirical models of the galaxy-halo connection, primarily in based in Python.

If you use any of the code implemented in this repository please consider citing either [1] for abundance matching or [2] for machine learning the scatter.

## Models
1. Subhalo abundance matching (SHAM) based on Yao-Yuan Mao's [implementation](https://github.com/yymao/abundancematching) with Peter Behroozi's fiducial deconvolution implementation based on the Richardson-Lucy algorithm. Assumes a constant (log)-normal scatter in the galaxy proxy conditioned on the halo proxy. For more information see the following example notebook: [``./tutorial/tutorial_SHAM.ipynb``](https://github.com/Richard-Sti/empiricalgalo/blob/master/tutorials/tutorial_SHAM.ipynb). Used and described in [1].

2. Ensemble of neural networks with a Gaussian loss function that predict both the mean predicted value and its standard deviation. See [``./tutorial/tutorial_GaussianLossNN.ipynb``](https://github.com/Richard-Sti/empiricalgalo/blob/master/tutorials/tutorial_GaussianLossNN.ipynb) for more information. Used and described in [2].


## Installation
```bash
pip install empiricalgalo
```


## References
[1] Richard Stiskalek; Harry Desmond; Thomas Holvey; and Michael G. Jones. "The dependence of subhalo abundance matching on galaxy photometry and selection criteria." Monthly Notices of the Royal Astronomical Society 506, no. 3 (2021): 3205-3223. [arXiv:2101.02765](https://arxiv.org/abs/2101.02765)
[2] Richard Stiskalek; Deaglan J. Bartlett; Harry Harry; Dhayaa Anbajagane. "The scatter in the galaxy-halo connection: a machine learning analysis" [arXiv:2202.14006](https://arxiv.org/abs/2202.14006)

## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
