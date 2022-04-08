# empiricalgalo: Empirical models of the galaxy-halo connection

Collection of empirical models of the galaxy-halo connection. Primarily in based in Python and Julia. If you use any of the code implemented in this repository please consider citing [1] where the code was used for subhalo abundance matching.

## Models
1. Subhalo abundance matching (SHAM) based on Yao-Yuan Mao's [implementation](https://github.com/yymao/abundancematching) with Peter Behroozi's fiducial deconvolution implementation based on the Richardson-Lucy algorithm. Assumes a constant (log)-normal scatter in the galaxy proxy conditioned on the halo proxy. For more information see the following example notebook: [``./tutorial/tutorial_model01_SHAM.ipynb``](https://github.com/Richard-Sti/empiricalgalo/blob/master/tutorials/tutorial_model01_SHAM.ipynb). This is a straightforward model which trades a small number of parameters for strong, although well physically justified assumptions of matching abundances. The major shortcoming of this model is that it assumes a constant scatter and relies on a well-defined halo proxy (or at least its functional relation).


## Installlation 
```bash
pip install .
```


## References
[1] Stiskalek, Richard, Harry Desmond, Thomas Holvey, and Michael G. Jones. "The dependence of subhalo abundance matching on galaxy photometry and selection criteria." Monthly Notices of the Royal Astronomical Society 506, no. 3 (2021): 3205-3223. [arXiv:2101.02765](https://arxiv.org/abs/2101.02765)

## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
