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
    x : numpy.ndarray (1D)
        The galaxy proxy.
    phi : numpy.ndarray (1D)
        The abundance values at `x` in units of :math:`x^{-1} L^{-3}` where
        :math:`L` is `boxsize` and :math:`x` is the galaxy proxy.
    ext_range : len-2 tuple
        Range of `x` over which to perform AM. Values outside `x` are
        extrapolated. For more information see
        `AbundanceMatching.AbundanceFunction`.
    boxsize : int
        Length of a side of the cubical simulation box.
    faint_end_first : bool
        Whether in `x` the faint end is listed first. Typically true for
        galaxy masses and false for magnitudes.
    scatter_mult : float
        A scatter multiplicative factor. Typically 1 for stellar mass and
        2.5 for magnitudes.
    **kwargs :
        Optional arguments passed into `AbundanceMatching.AbundanceFunction`.

    References
    ----------
    .. [1] https://bitbucket.org/yymao/abundancematching/

    """
    name = "AbundanceMatch"
    _boxsize = None
    _scatter_mult = None

    def __init__(self, x, phi, ext_range, boxsize, faint_end_first,
                 scatter_mult, **kwargs):
        # Initialise the abundance function
        self.af = AbundanceFunction(x, phi, ext_range,
                                    faint_end_first=faint_end_first, **kwargs)
        self.boxsize = boxsize
        self.scatter_mult = scatter_mult

    @property
    def boxsize(self):
        """
        Simulation box side :math:`L`, the volume is :math:`L^3`.
        """
        return self._boxsize

    @boxsize.setter
    def boxsize(self, boxsize):
        """
        Sets `boxsize` and ensures it is positive.
        """
        if boxsize <= 0:
            raise ValueError("'boxsize' must be positive.")
        self._boxsize = boxsize

    @property
    def scatter_mult(self):
        """
        The scatter multiplicative factor.
        """
        return self._scatter_mult

    @scatter_mult.setter
    def scatter_mult(self, scatter_mult):
        """
        Sets `scatter_mult`.
        """
        if scatter_mult <= 0:
            raise ValueError("'scatter_mult' must positive.")
        self._scatter_mult = scatter_mult

#    @halo_proxy.setter
#    def halo_proxy(self, halo_proxy):
#        """Sets the halo proxy."""
#        if halo_proxy.name not in proxies.keys():
#            raise ValueError("Unrecognised proxy '{}'. Supported proxies: {}"
#                             .format(halo_proxy, [k for k in proxies.keys()]))
#        self._halo_proxy = halo_proxy

    def deconvoluted_catalogs(self, theta, halos, nrepeats=20,
                              return_remainder=False):
        """
        Returns a catalog with no scatter and a deconvoluted scatter. To
        calculate catalogs with scatter see `self.add_scatter`.

        Parameters
        ----------
        theta : dict
            Halo proxy parameters. Must include parameters required by
            `halo_proxy` and can include `scatter`.
        halos : structured numpy.ndarray
            Halos array with named fields containing the required parameters
            to calculate the halo proxy.
        n_repeats : int, optional
            Number of times to repeat fiducial deconvolute process. By
            default 20.
        return_remainder : bool, optional
            Whether to return the remainder of the convolution.

        Returns
        -------
        result : dict
            Keys include:
                cat0 : numpy.ndarray
                    A catalog without scatter.
                cat_deconv : numpy.ndarray
                    A deconvoluted catalog. Scatter is not added yet.
                scatter : float
                    Gaussian scatter.
                preselect_mask : numpy.ndarray
                    A preselection mask.
                mask_nans : numpy.ndarray
                    A mask of which halos had a galaxy assigned.
        remainder : numpy.ndarray
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
        # Some proxies preselect halos
        if len(proxy) == 2:
            plist, preselect_mask = proxy
            res.update({'preselect_mask': preselect_mask})
        else:
            plist = proxy

        # Check all params have been used
        if len(theta) != 0:
            raise ValueError("Unrecognised parameters '{}'"
                             .format(theta.keys()))

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
        cat0 = self.af.match(nd_halos)
        # Deconvoluted catalog. Without adding the scatter
        cat_deconv = self.af.match(nd_halos, scatter, False)

        # Figure out which galaxies in the catalog are not NaNs
        mask_nans = ~numpy.isnan(cat_deconv)

        res.update({'cat0': cat0[mask_nans],
                    'cat_deconv': cat_deconv[mask_nans],
                    'scatter': scatter,
                    'mask_nans': mask_nans})

        if return_remainder:
            return res, remainder
        return res

    def add_scatter(self, catalogs, cut_range, return_catalog=False):
        """
        Adds scatter to a previously deconvoluted catalog from
        `self.deconvoluted_catalogs` and selects galaxies within `cut_range`.

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
        cut_range : len-2 tuple
            Faint and bright end cut offs.
        return_catalog : bool, optional
            Whether to return the matched catalog. By default not returned.

        Returns
        -------
        mask : numpy.ndarray
            Mask corresponding to the `halos` object passed into
            `self.deconvoluted_catalogs`. Determines which halos were
            assigned a within `cut_range`.
        catalog : numpy.ndarray
            Returned if `return_catalog`. Matched galaxies, typically either
            log stellar mass or absolute magnitude.
        """
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