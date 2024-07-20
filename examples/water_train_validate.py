import psiflow
from psiflow.data import Dataset, compute_rmse
from psiflow.models import MACE


def main():
    data = Dataset.load("data/water_train.xyz")
    model = MACE(
        batch_size=2,
        lr=0.02,
        max_ell=3,
        r_max=5.5,
        energy_weight=100,
        correlation=3,
        max_L=1,
        num_channels=16,
        max_num_epochs=20,
        swa=False,
    )

    train, valid = data.split(0.9, shuffle=True)
    model.initialize(train)
    model.train(train, valid)
    hamiltonian = model.create_hamiltonian()

    target_e = data.get("per_atom_energy")
    target_f = data.get("forces")

    data_predicted = data.evaluate(hamiltonian)
    predict_e = data_predicted.get("per_atom_energy")
    predict_f = data_predicted.get("forces")

    e_rmse = compute_rmse(target_e, predict_e)
    f_rmse = compute_rmse(target_f, predict_f)

    print("RMSE(energy) [meV/atom]: {}".format(e_rmse.result() * 1000))
    print("RMSE(forces) [meV/angstrom]: {}".format(f_rmse.result() * 1000))


if __name__ == "__main__":
    with psiflow.load():
        main()
