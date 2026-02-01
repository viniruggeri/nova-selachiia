"""
Neural State Space Models for Shark Population Dynamics.

Available Models
================
- NSSM: Nonlinear State Space Model (deterministic baseline)
- DMM: Deep Markov Model (stochastic, future implementation)

Usage
=====
>>> from nova_selachiia.models import NSSM, NSSMConfig
>>> config = NSSMConfig(input_dim=5, hidden_dim=64)
>>> model = NSSM(**config.to_dict())
"""

from .nssm import NSSM, NSSMConfig

__all__ = ["NSSM", "NSSMConfig"]
