import io
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import py3Dmol
import pymbar
from ase import Atoms
from ase.io import write
from pymbar import timeseries


def visualize_trajectory(trajectory: list[Atoms]):
    # Create a py3Dmol view object
    block = py3Dmol.view(width=800, height=400)
    block.setBackgroundColor("lightgray")
    block.zoom(1.3)

    def show_structure(frame):
        atoms = trajectory[frame].copy()
        cv = atoms.info["CV"]

        # Convert the Atoms object to an XYZ formatted string
        xyz_io = io.StringIO()
        write(xyz_io, atoms, format="xyz")
        xyz_str = xyz_io.getvalue()

        block.removeAllModels()
        block.addModel(xyz_str, format="xyz")
        block.setStyle(
            {},
            {
                "stick": {"colorscheme": "gray", "radius": 0.1},
                "sphere": {"radius": 0.3},
            },
        )
        block.show()
        energy_label.value = f"CV value: {cv:.2f}"

    # Create a slider for navigating the trajectory
    # slider = widgets.IntSlider(min=0, max=len(trajectory) - 1, description="Frame")

    # Create a label to display the energy
    cv = trajectory[0].info["CV"]
    energy_label = widgets.Label(value=f"CV value: {cv:.2f}")

    # display(
    #     widgets.VBox(
    #         [
    #             slider,
    #             energy_label,
    #             widgets.interactive_output(show_structure, {"frame": slider}),
    #         ]
    #     )
    # )


def generate_fes(
    path_umbrellas: Path,
    centers: np.ndarray,
    kappa: float,
    temperature: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    kB = 1.381e-23 * 6.022e23 / 1000.0  # in kJ/mol/K

    trajectories = []
    i = 0
    while True:
        path = path_umbrellas / "{}.txt".format(i)
        if path.exists():
            trajectories.append(np.loadtxt(path))
            i += 1
        else:
            break
    assert i == len(centers)
    N_k = [0] * len(centers)

    # decorrelate
    for i, trajectory in enumerate(trajectories):
        indices = timeseries.subsample_correlated_data(trajectory)
        N_k[i] = len(indices)
        trajectories[i] = trajectory[indices]

    # compute bin centers
    bin_center_i = np.zeros([nbins])
    cv_min = min([np.min(t) for t in trajectories])
    cv_max = max([np.max(t) for t in trajectories])
    bin_edges = np.linspace(cv_min, cv_max, nbins + 1)
    for i in range(nbins):
        bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

    # pad and put in array
    N = max(N_k)
    c_kn = np.zeros((len(centers), N))
    for i, trajectory in enumerate(trajectories):
        c_kn[i, : N_k[i]] = trajectory[:]

    # bias energy from snapshot n in umbrella k evaluated at umbrella l
    beta = 1 / (kB * temperature)
    u_kln = np.zeros((len(centers), len(centers), c_kn.shape[1]))
    for k in range(len(centers)):
        for n in range(N_k[k]):
            for l in range(len(centers)):  # noqa: E741
                u_kln[k, l, n] = beta * kappa / 2 * (c_kn[k, n] - centers[l]) ** 2

    c_n = pymbar.utils.kn_to_n(c_kn, N_k=N_k)
    fes = pymbar.FES(u_kln, N_k, verbose=True)
    u_kn = np.zeros(c_kn.shape)

    # compute free energy profile in unbiased potential (in units of kT).
    histogram_parameters = {}
    histogram_parameters["bin_edges"] = bin_edges
    fes.generate_fes(
        u_kn, c_n, fes_type="histogram", histogram_parameters=histogram_parameters
    )
    results = fes.get_fes(
        bin_center_i, reference_point="from-lowest", uncertainty_method="analytical"
    )
    center_f_i = results["f_i"] * kB * temperature
    center_df_i = results["df_i"] * kB * temperature

    # Write out free energy profile
    print("free energy profile [kJ/mol]")
    print(f"{'bin':>8s} {'f':>8s} {'df':>8s}")
    for i in range(nbins):
        print(f"{bin_center_i[i]:8.1f} {center_f_i[i]:8.3f} {center_df_i[i]:8.3f}")
    return bin_center_i, center_f_i
