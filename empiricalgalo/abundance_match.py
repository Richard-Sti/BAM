# Copyright (C) 2020  Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""A class for abundance matching without too much strain on the memory."""

import numpy
from AbundanceMatching import (AbundanceFunction, rematch, add_scatter,
                               calc_number_densities)
from .proxy import proxies


class AbundanceMatch:
    r"""
    A wrapper around Yao-Yuan Mao's Subhalo Abundance Matching (SHAM)
    Python package [1]. Performs abundance matching on a list of halos.

    Parameters
    ----------
    x : 1-dimensional array
        The galaxy proxy.
    phi : 1-dimensional array
        The abundance values at `x` in units of :math:`x^{-1} L^{-3}` where
        :math:`L` is `boxsize` and :math:`x` is the galaxy proxy.
    halo_proxy : :py:class:`empiricalgalo.proxy`
        A halo proxy object.
    ext_range : tuple of length 2
        Range of `x` over which to perform AM. Values outside `x` are
        extrapolated. Recommended to be sufficiently wider than ``cut_range``
        in ``self.add_scatter``. For more information see
        py:class:`AbundanceMatching.AbundanceFunction`.
    boxsize : int
        Length of a side of the cubical simulation box.
    faint_end_first : bool
        Whether in `x` the faint end is listed first. Typically true for
        galaxy masses and false for magnitudes.
    scatter_mult : float
        A scatter multiplicative factor. Typically 1 for stellar mass and
        2.5 for magnitudes.
    **kwargs :
        Optional arguments passed into
        py:class:`AbundanceMatching.AbundanceFunction`.

    References
    ----------
    .. [1] https://github.com/yymao/abundancematching
    """
    name = "AbundanceMatch"
    _boxsize = None
    _scatter_mult = None
    _halo_proxy = None

    def __init__(self, x, phi, halo_proxy, ext_range, boxsize, faint_end_first,
                 scatter_mult, **kwargs):
        # Initialise the abundance function
        self.af = AbundanceFunction(x, phi, ext_range,
                                    faint_end_first=faint_end_first, **kwargs)
        self.boxsize = boxsize
        self.scatter_mult = scatter_mult
        self.halo_proxy = halo_proxy

    @property
    def boxsize(self):
        """Simulation box side :math:`L`, the volume is :math:`L^3`."""
        return self._boxsize

    @boxsize.setter
    def boxsize(self, boxsize):
        """Set `boxsize` and ensures it is positive."""
        if boxsize <= 0:
            raise ValueError("``boxsize`` must be positive.")
        self._boxsize = boxsize

    @property
    def scatter_mult(self):
        """The scatter multiplicative factor."""
        return self._scatter_mult

    @scatter_mult.setter
    def scatter_mult(self, scatter_mult):
        """Set `scatter_mult`, checks it is positive."""
        if scatter_mult <= 0:
            raise ValueError("``scatter_mult`` must positive.")
        self._scatter_mult = scatter_mult

    @property
    def halo_proxy(self):
        """The halo proxy."""
        return self._halo_proxy

    @halo_proxy.setter
    def halo_proxy(self, halo_proxy):
        """Sets the halo proxy."""
        if halo_proxy.name not in proxies.keys():
            raise ValueError("Unrecognised proxy '{}'. Supported proxies: {}"
                             .format(halo_proxy, [k for k in proxies.keys()]))
        self._halo_proxy = halo_proxy

    def deconvoluted_catalogs(self, theta, halos, nrepeats=20,
                              return_remainder=False):
        """
        Calculate a deconvoluted catalog, to calculate catalogs with scatter
        will add Gaussian scatter to it in ``self.add_scatter``.

        Parameters
        ----------
        theta : dict
            Halo proxy parameters. Must include parameters required by
            `halo_proxy` and can include `scatter`. If the scatter is not
            specified it is set to 0.
        halos : structured numpy.ndarray
            Halos array with named fields containing the required parameters
            to calculate or extract the halo proxy.
        n_repeats : int, optional
            Number of times to repeat fiducial deconvolute process. By
            default 20.
        return_remainder : bool, optional
            Whether to return the remainder of the convolution. This should be
            sufficiently close to 0.

        Returns
        -------
        result : dict
            Keys include:
                cat0 : 1-dimensional array
                    A catalog without scatter.
                cat_deconv : 1-dimensional array
                    A deconvoluted catalog. Scatter is not added yet.
                scatter : float
                    Gaussian scatter to be added to the deconvoluted catalog.
                preselect_mask : 1-dimensional array
                    A preselection mask.
                mask_nans : 1-dimensional array
                    A mask of which halos had a galaxy assigned.
        remainder : 1-dimensional array
            Optionally if `return_remainder` returns the deconvolution's
            remainder.
        """
        # Pop the scatter from theta, apply any multiplicatory factor
        scatter = theta.pop('scatter', None)
        if scatter is None:
            scatter = 0
        scatter *= self.scatter_mult
        # Calculate the AM proxy
        proxy = self.halo_proxy(halos, theta)
        res = {}
        # Some proxies preselect halos that are to be abundance matched
        if len(proxy) == 2:
            plist, preselect_mask = proxy
            res.update({'preselect_mask': preselect_mask})
        else:
            plist = proxy

        # Check all params have been used
        if len(theta) > 0:
            raise ValueError("Unrecognised remaining parameters: ``{}``"
                             .format(list(theta.keys())))

        nd_halos = calc_number_densities(plist, self.boxsize)
        # AbundanceFunction stores deconvoluted results by scatter, so no
        # need to reset it when calling it with a different scatter.
        if return_remainder:
            remainder = self.af.deconvolute(scatter, repeat=nrepeats)
        else:
            try:
                self.af._x_deconv[scatter]
            except KeyError:
                self.af.deconvolute(scatter, repeat=nrepeats,
                                    return_remainder=False)

        # Catalog with 0 scatter
        cat0 = self.af.match(nd_halos, scatter=0, do_add_scatter=True)
        # Deconvoluted catalog. Without adding the scatter
        cat_deconv = self.af.match(nd_halos, scatter, do_add_scatter=False)
        # Figure out which galaxies in the catalog are not NaNs nad have a
        # galaxy matched
        mask_nans = ~numpy.isnan(cat_deconv)

        res.update({'cat0': cat0[mask_nans],
                    'cat_deconv': cat_deconv[mask_nans],
                    'scatter': scatter,
                    'mask_nans': mask_nans})

        if return_remainder:
            return res, remainder
        return res

    def add_scatter(self, catalogs, cut_range, return_catalog=True):
        """
        Adds scatter to a previously deconvoluted catalog from
        ``self.deconvoluted_catalogs`` and selects galaxies within
        ``cut_range``.

        Parameters
        ----------
        catalogs : dict
            Keys must include:
                cat0 : numpy.ndarray
                    A catalog without scatter.
                cat_deconv : numpy.ndarray
                    A deconvoluted catalog. Scatter is not added yet.
                scatter : float
                    Gaussian scatter used to deconvolute this catalog.
                mask_nans : 1-dimensional array
                    A mask of which halos had a galaxy assigned.
            Optionally if preselection:
                preselect_mask : 1-dimensional array
                    A preselection mask.
        cut_range : tuple of length 2
            Lower and upper cut offs on the abundance matched galaxies.
        return_catalog : bool, optional
            Whether to return the matched catalog. By default not returned.

        Returns
        -------
        mask : 1-dimensional array
            Mask corresponding to the `halos` object passed into
            `self.deconvoluted_catalogs`. Determines which halos were assigned
            a galaxy with ``cut_range``.
        catalog : 1-dimensional array
            Returned if `return_catalog`. Matched galaxy proxies.
        """
        # Check ordering
        if cut_range[0] > cut_range[1]:
            cut_range = cut_range[::-1]

        cat_scatter = add_scatter(catalogs['cat_deconv'], catalogs['scatter'])
        cat_scatter = rematch(cat_scatter, catalogs['cat0'],
                              self.af._x_flipped)

        # Select halos that make the cut
        mask_cut = numpy.logical_and(cat_scatter > cut_range[0],
                                     cat_scatter < cut_range[1])
        catalog = cat_scatter[mask_cut]
        # Final mask denoting which halos made it
        mask = numpy.where(catalogs['mask_nans'])[0][mask_cut]
        # Apply preselection, if any
        try:
            mask = numpy.where(catalogs['preselect_mask'])[0][mask]
        except KeyError:
            pass

        if return_catalog:
            return mask, catalog
        return mask
