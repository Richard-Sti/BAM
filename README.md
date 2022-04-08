# ClusterSHAM - Clustering-calibrated subhalo abundance matching models

ClusterSHAM is a collection of tools to explore the parameter space of subhalo abundance matching (AM) models.

## Generating galaxy mocks

A deconvoluted and zero scatter mock galaxy catalogs are calculated via `ClusterSHAM.mocks.AbundanceMatch.deconvoluted_catalogs`. Scatter can be added using `ClusterSHAM.mocks.AbundanceMatch.add_scatter`. To generate `N` independent mocks it is sufficient to use the same pair of deconvoluted and zero scatter mock galaxy catalogs.

`ClusterSHAM.mocks.AbundanceMatch` requires a halo proxy object. Currently supported proxies can be found [here](https://github.com/Richard-Sti/ClusterSHAM/blob/master/clustersham/mocks/proxy.py).

Based on https://github.com/yymao/abundancematching.


## 2-point correlation function in mocks

Can be calculated via `ClusterSHAM.mocks.Correlator.mock_wp`, for the jackknife covariance matrix see `ClusterSHAM.mocks.Correlator.mock_jackknife_wp`.

Based on https://github.com/manodeep/Corrfunc.


## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
