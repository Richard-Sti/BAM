# BAM: Baryonic Abundance Matching


BAM implements a subhalo abundance matching model based on Yao-Yuan Mao's [implementation](https://github.com/yymao/abundancematching) with Peter Behroozi's fiducial deconvolution via the Richardson-Lucy algorithm.

Assumes a constant lofnormal scatter in the galaxy proxy conditioned on the halo proxy. See the following example notebook: [``./tutorial/tutorial_SHAM.ipynb``](https://github.com/Richard-Sti/empiricalgalo/blob/master/tutorials/tutorial_SHAM.ipynb).


If you use or find helpful any of the code implemented inthis repository please cite [1].

## Installation
```bash
pip install empiricalgalo
```

## References
[1] Richard Stiskalek; Harry Desmond; Thomas Holvey; and Michael G. Jones. "The dependence of subhalo abundance matching on galaxy photometry and selection criteria." Monthly Notices of the Royal Astronomical Society 506, no. 3 (2021): 3205-3223. [arXiv:2101.02765](https://arxiv.org/abs/2101.02765)

## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
