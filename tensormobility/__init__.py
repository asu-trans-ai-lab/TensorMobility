"""TensorMobility: mobility systems as
flow-through tensors -- space, time, and behavior in one certified
computational graph. Umbrella package; sub-names:

  tensormobility.core      axes calculus, typed contracts, unified GMNS networks
  tensormobility.dta       the DTA core: certified assignment & column generation
  tensormobility.dynamics  fluid/path/cohort queues (time seam)
  tensormobility.behavior  activity chains, bounded learning
  tensormobility.engines   equilibrium engine escalation ladder
  tensormobility.profiles  passenger-vehicle, mixed-autonomy (MAGE) layers
  tensormobility.harness   experiment harnesses & well-posedness maps

Implements the Space-Time-Behavior Flow-Through Tensor (STB-FTT)
framework."""
from tensormobility.core.unified_networks import load_case, UnifiedCase
from tensormobility.core.axes import CANONICAL_AXES, slice_mage
from tensormobility.dta.sparse_assignment import network_from_case, solve_fw
from tensormobility.engines.equilibrium_engines import solve_fixed_point

__version__ = "0.6.1"
