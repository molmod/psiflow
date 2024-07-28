Once we know how to represent datasets of atomic geometries and label with them with
target QM energy and force values, we can start defining and training ML potentials.
Psiflow defines an abstract `Model` interface which each
particular ML potential should subclass, though at the moment only
[MACE](https://github.com/acesuit/mace) is implemented.
In addition, psiflow provides configuration dataclasses for each model with
reasonable defaults.

A `Model` has essentially three methods:

- `initialize`: compute energy shifts and scalings as well as the average number
of neighbors (and any other network normalization metrics) using a given *training* dataset,
and initialize model weights.
- `train`: train the parameters of a model using two separate datasets, one for
actual training and one for validation. The current model parameters are used as
starting parameters for the training
- `create_hamiltonian`: spawn a hamiltonian in order to use the model with its current
  weights in molecular dynamics simulations 

The following is a minimal illustration:
```py
from psiflow.data import Dataset
from psiflow.models import MACE


# load data with energy and force labels included as extxyz
train, valid = Dataset.load('all_data.xyz').split(0.9, shuffle=True)

model = MACE(                   # for full arg list, see psiflow/models/_mace:MACEConfig
    num_channels=16,
    max_L=2,
    max_num_epochs=400,
    batch_size=16,
)

# initialize, train
model.initialize(train)         # this will calculate the scale/shifts, and average number of neighbors
model.train(train, valid)       # train using supplied datasets

model.save('./')                # saves model and config to current working directory!

hamiltonian = model.create_hamiltonian()
forces_pred = hamiltonian.compute(valid, 'forces')
forces_target = valid.get('forces')

rmse = compute_rmse(forces_pred, forces_target)  # this is a Future!
print('forces RMSE: {} eV/A'.format(rmse.result()))

```
Note that `model.save()` will save both a `.yaml` file with all hyperparameters as well as the actual `.pth` model which is needed to reconstruct the corresponding PyTorch module (possibly outside of psiflow if needed).
As such, it expects a directory as argument (which may either already exist or will be
created).

In many cases, it is generally recommended to provide these models with some estimate of the absolute energy of an isolated
atom for the specific level of theory and basis set considered (and this for each element).
Instead of having the model learn the *absolute* total energy of the system, we first subtract these atomic energies in order
to train the model on the *formation* energy of the system instead, as this generally improves the generalization performance
of the model towards unseen stoichiometries.

```py
model.add_atomic_energy('H', -13.7)     # add atomic energy of isolated hydrogen atom
model.initialize(some_training_data)

model.add_atomic_energy('O', -400)      # will raise an exception; model needs to be reinitialized first
model.reset()                           # removes current model, but keeps raw config
model.add_atomic_energy('O', -400)      # OK!
model.initialize(some_training_data)    # offsets total energy with given atomic energy values per atom

```
Whenever atomic energies are available, `Model` instances will automatically offset the potential energy in a (labeled)
`Dataset` by the sum of the energies of the isolated atoms; the underlying PyTorch network is then initialized/trained
on the formation energy of the system instead.
In order to avoid artificially high energy discrepancies between models trained on the formation energy on one hand,
and reference potential energies as obtained from any `BaseReference`,
the `evaluate` method will first perform the converse operation, i.e. add the energies of the isolated atoms
to the model's prediction of the formation energy.
Similarly, `create_hamiltonian()` also passes any atomic energies which were added to the
model.
