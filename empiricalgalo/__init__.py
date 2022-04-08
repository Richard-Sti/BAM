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

# get version
def _read_version():
    with open("empiricalgalo/_version.py", "r") as fh:
        vstr = fh.read().strip()
    try:
        vstr = vstr.split('=')[1].strip()[1:-1]
    except IndexError:
        raise RuntimeError("version string in empiricalgalo._verion.py not "
                           "formatted correctly; it should be:\n"
                           "__version__ = VERSION")
    return vstr


__version__ = _read_version()
__author__ = "Richard Stiskalek"

from .abundance_match import AbundanceMatch
from .proxy import proxies

