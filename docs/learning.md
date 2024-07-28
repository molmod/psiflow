Psiflow allows for the seamless development and scalable
execution of online learning algorithms for ML potentials.
The `Learning` class provides an interface based on which such
algorithms can be implemented.
They keep track of the generated data, error metrics, optional [Weights &
Biases](https://wandb.ai) logging, and provide basic restart functionalities in case
something goes wrong.
Learning objects are instantiated using the following arguments:

- **reference** (type `Reference`): the `Reference` instance which will be used to
  evaluate ground-truth energy/force labels for each of the samples generated.
- **path_output** (type `str | Path`): the location to a folder in which intermediate
  models, datasets, walker states, and restart files can be saved.
- **train_valid_split** (type `float`): fraction of generated data which should be used
  for the training set (as opposed to validation).
- **error_thresholds_for_reset** (type `list[Optional[float]]`): during online learning,
  it is not uncommon to have walkers explore unphysical regions in phase space due to
  irregularities in the intermediate potential, excessive temperatures/pressures, ...
  In those cases, it is beneficial to reset walkers to their starting configurations, of
  which it is known to be a physically sound starting point. The decision to reset walkers
  is made every time the 'exact' energy and forces have been computed from a sampled
  state. If the error between the corresponding walker's model (i.e. the previous model)
  and the QM-evaluated energy and forces exceeds a certain threshold (both on energies and
  forces), the walker is reset.
  This argument expects a list of length two (threshold on energy error, and threshold on
  force error), with optional `None` values if no reset is desired.
  For example: `[None, 0.1]` indicates to reset whenever the force RMSE exceeds 100 meV/A,
  and ignore any energy discrepancy.
- **error_thresholds_for_discard** (type `list[Optional[float]]`): states which are
  entirely unphysical do not contribute to the accuracy of the model, and sometimes even
  hinder proper training. If these error thresholds are exceeded, the state is discarded and the walker is reset.
- **wandb_group** (type `str`): if specified, the computed dataset metrics will be logged
  to Weights & Biases in the corresponding group of runs for easy visual analysis.
- **wandb_project** (type `str`): if specified, the computed dataset metrics will be logged
  to Weights & Biases in the corresponding project for easy visual analysis.
- **initial_data** (type `Dataset`): existing, labeled data from which the learning can be
  bootstrapped. Note that *all* states in this dataset must be labeled, and that this is
  only sensible if the labeling agrees with the given Reference instance. (Same level of
  theory, same basis set, grid settings, ... ).

