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

"""Halo proxies to be used for abundance matching."""

from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy

from astropy import constants as const, units as u
from astropy.cosmology import FlatLambdaCDM
from halotools import empirical_models


@add_metaclass(ABCMeta)
class BaseProxy(object):
    """
    Abstract class for handling the abundance matching proxies. All
    proxies must inherit from this.

    Parameters
    ----------
    halo_params: (list of) str
        Names of halo parameters to calculate the proxy.
    """
    _halo_params = None
    _cache = {}

    @property
    def halo_params(self):
        """The halo parameters to calculate the proxy."""
        return self._halo_params

    @halo_params.setter
    def halo_params(self, pars):
        """Sets the halo parameters. Ensures it is a list of strings."""
        if isinstance(pars, str):
            pars = [pars]
        if not isinstance(pars, (list, tuple)):
            raise ValueError("'halo_params' must be a list.")
        pars = list(pars)
        for p in pars:
            if not isinstance(p, str):
                raise ValueError("Halo parameter '{}' is not a string"
                                 .format(p))
        self._halo_params = pars

    def _check_halo_attributes(self, halos):
        """
        Checks whether the halo params required by the proxies are included.
        """
        for attr in self.halo_params:
            if attr not in halos.dtype.names:
                raise ValueError("`halos` missing attribute '{}'".format(attr))

    @abstractmethod
    def __call__(self, halos, theta):
        pass


class VirialMassProxy(BaseProxy):
    r"""
    Peak to present virial mass halo proxy defined as

        .. math::
            m_{\alpha} = M_0 * \left M_{\mathrm{peak}} / M_0\right]^{\alpha},

    where :math:`M_0` and :math:`M_{\mathrm{peak}}` are the present and peak
    virial masses, respectively.
    """
    name = 'mvir_proxy'

    def __init__(self):
        self.halo_params = ['mvir', 'mpeak']

    def __call__(self, halos, theta):
        """
        Calculates the halo proxy.

        Parameters
        ----------
        halos : structured numpy.ndarray
            Array of halos with named fields.
        theta : dict
            A dictionary of proxy parameters.

        Returns
        -------
        proxy : numpy.ndarray
            Halo proxy.
        """
        alpha = theta.pop('alpha', None)
        if alpha is None:
            raise ValueError("'alpha' must be specified.")
        self._check_halo_attributes(halos)

        # Get the cached proxy parameters. If not found calculate
        try:
            logMvir = self._cache['logMvir']
        except KeyError:
            logMvir = numpy.log10(halos['mvir'])
            self._cache.update({'logMvir': logMvir})
        try:
            logMratio = self._cache['logMratio']
        except KeyError:
            logMratio = numpy.log10(halos['mpeak'] / halos['mvir'])
            self._cache.update({'logMratio': logMratio})

        # More efficient than a single line expression
        proxy = numpy.copy(logMvir)
        proxy += alpha * logMratio
        return proxy


class PeakRedshiftProxy(BaseProxy):
    r"""
    A pre-selection proxy that eliminates all halos whose peak mass
    redshift is above ``zcutoff``. The remaining halos are ranked by the
    present virial mass.
    """

    name = 'zmpeak_proxy'

    def __init__(self):
        self.halo_params = ['mvir', 'mpeak_scale']

    def __call__(self, halos, theta):
        """
        Calculates the halo proxy.

        Parameters
        ----------
        halos : structured numpy.ndarray
            Array of halos with named fields.
        theta : dict
            A dictionary of proxy parameters.

        Returns
        -------
        proxy : numpy.ndarray
            Halo proxy.
        """
        zcutoff = theta.pop('zcutoff', None)
        if zcutoff is None:
            raise ValueError("'zcutoff' must be specified.")
        self._check_halo_attributes(halos)

        # Save values that do not change during runtime
        try:
            zmpeak = self._cache['zmpeak']
        except KeyError:
            zmpeak = 1. / halos['mpeak_scale'] - 1
            self._cache.update({'zmpeak': zmpeak})
        try:
            logMvir = self._cache['logMvir']
        except KeyError:
            logMvir = numpy.log10(halos['mvir'])
            self._cache.update({'logMvir': logMvir})

        mask = zmpeak < zcutoff
        proxy = logMvir[mask]
        return proxy, mask


class VirialVelocityProxy(BaseProxy):
    r"""
    Virial velocity abundance matching proxy defined as


        .. math::
            v_{\alpha} = v_0 * \left[ v_{\max} / v_0\right]^{\alpha},


    where :math:`v_0` is the virial velocity evaluated at the peak halo mass
    and :math:`v_{\max}` is the maximum circular velocity.

    Parameters
    ----------
    cosmo : astropy.cosmology
        Cosmology to calculate the the virial velocity at the peak halo mass.
        Must match the simulation cosmology.

    Note
    ----
    This proxy has not been optimised and requires precomputed 'vvir`.

    """
    name = 'vvir_proxy'

    def __init__(self, cosmo):
        self.halo_params = ['mpeak', 'mpeak_scale', 'Vmax@Mpeak']
        self.cosmo = cosmo

    def __call__(self, halos, theta):
        """
        Calculates the halo proxy.

        Parameters
        ----------
        halos : structured numpy.ndarray
            Array of halos with named fields.
        theta : dict
            A dictionary of proxy parameters.

        Returns
        -------
        proxy : numpy.ndarray
            Halo proxy.
        """
        alpha = theta.pop('alpha', None)
        if alpha is None:
            raise ValueError("'alpha' must be specified.")
        self._check_halo_attributes(halos)
        # Get vvir
        try:
            log_vvir = self._cache['log_vvir']
        except KeyError:
            log_vvir = numpy.log10(self.get_vvir(halos))
            self._cache.update({'log_vvir': log_vvir})

        try:
            log_vratio = self._cache['log_vratio']
        except KeyError:
            log_vratio = numpy.log10(halos['Vmax@Mpeak']) - log_vvir
            self._cache.update({'log_vratio': log_vratio})

        proxy = numpy.copy(log_vvir)
        proxy += alpha * log_vratio
        return proxy

    def get_vvir(self, halos):
        """
        Calculates the virial velocity at peak halo mass as defined in [1].

            .. Note:
                The critical density is assumed to be at present time.


        Parameters
        ----------
        halos : structured numpy.ndarray
            Halos array with names fields

        References
        ----------
        .. [1] Lehmann, Benjamin V et al. "The Concentration Dependence of the
               Galaxy-Halo Connection." arXiv:1510.05651

        Returns
        ----------
        vvir : numpy.ndarray
            Virial velocity at peak halo mass in km/s.
        """
        z_mpeak = (1.0 / halos['mpeak_scale']) - 1
        mvir = halos['mpeak']
        OmZ = self.cosmo.Om(z_mpeak)
        Delta_vir = empirical_models.delta_vir(self.cosmo, z_mpeak) / OmZ

        rho_crit = self.cosmo.critical_density0.to(u.kg/u.m**3)
        vvir = ((4 * numpy.pi / 3 * const.G**3 * Delta_vir * rho_crit)**(1/6)
                * (mvir * u.Msun.decompose())**(1/3)).to(u.km/u.s)
        return vvir.value


#
# =============================================================================
#
#                     A dictionaries of proxies
#
# =============================================================================
#


proxies = {VirialMassProxy.name: VirialMassProxy,
           VirialVelocityProxy.name: VirialVelocityProxy,
           PeakRedshiftProxy.name: PeakRedshiftProxy}
