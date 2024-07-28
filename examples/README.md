- [Replica exchange molecular dynamics](https://github.com/molmod/psiflow/tree/main/examples/alanine_replica_exchange.py) | **alanine dipeptide**: replica exchange molecular dynamics simulation of alanine dipeptide, using the MACE-MP0 universal potential.
  The inclusion of high-temperature replicas allows for fast conformational transitions and improves ergodicity.
- [Geometry optimizations](https://github.com/molmod/psiflow/tree/main/examples/formic_acid_transition.py) | **formic acid dimer**: approximate transition state calculation for the proton exchange reaction in a formic acid dimer,
  using simple bias potentials and a few geometry optimizations.
- [Static and dynamic frequency analysis](https://github.com/molmod/psiflow/tree/main/examples/h2_static_dynamic.py) | **dihydrogen**: Hessian-based estimate of the H-H bond strength and corresponding IR absorption frequency, and a comparison with a dynamical estimate from NVE simulation and Fourier analysis.
  
- [Bulk modulus calculation](https://github.com/molmod/psiflow/tree/main/examples/iron_bulk_modulus.py) | **iron**: estimate of the bulk modulus of fcc iron using a series of NPT simulations at different pressures
  
- [Solid-state phase stabilities](https://github.com/molmod/psiflow/tree/main/examples/iron_harmonic_fcc_bcc.py) | **iron**: estimating the relative stability of fcc and bcc iron with anharmonic corrections using thermodynamic integration (see e.g. [Phys Rev B., 2018](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.054102))

- [DFT singlepoints](https://github.com/molmod/psiflow/tree/main/examples/water_cp2k_noise.py) | **water**: analysis of the numerical noise DFT energy and force evaluations using CP2K and the RPBE(D3) functional, for a collection of water molecules.
  
- [Path-integral molecular dynamics](https://github.com/molmod/psiflow/examples/water_path_integral_md.py) | **water**: demonstration of the impact of nuclear quantum effects on the variance in O-H distance in liquid water. Path-integral molecular dynamics simulations with increasing number of beads (1, 2, 4, 8, 16) approximate the proton delocalization, and lead to systematically larger variance in O-H distance.
  
- [Machine learning potential training](https://github.com/molmod/psiflow/examples/water_train_validate.py) | **water**: simple training and validation script for MACE on a small dataset of water configurations.
