"""Noise-induced barren plateau study: adaptive vs hardware-efficient VQE ansatz."""

from .hamiltonian import TFIM, make_tfim
from .meanfield import meanfield_energy

__all__ = ["TFIM", "make_tfim", "meanfield_energy"]
__version__ = "0.1.0"
