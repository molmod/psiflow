"""
TODO: these functions seem to not be used anywhere in the codebase
 probably a good idea to remove these eventually
"""



@typeguard.typechecked
def _check_distances(state: Geometry, threshold: float) -> Geometry:
    """
    Check if all interatomic distances in a Geometry are above a threshold.

    Args:
        state: Geometry instance to check.
        threshold: Minimum allowed interatomic distance.

    Returns:
        Geometry: The input Geometry if all distances are above the threshold, otherwise NullState.

    Note:
        This function is wrapped as a Parsl app and executed using the default_htex executor.
    """
    from ase.geometry.geometry import find_mic

    if state == NullState:
        return NullState
    nrows = int(len(state) * (len(state) - 1) / 2)
    deltas = np.zeros((nrows, 3))
    count = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            deltas[count] = state.per_atom.positions[i] - state.per_atom.positions[j]
            count += 1
    assert count == nrows
    if state.periodic:
        deltas, _ = find_mic(deltas, state.cell)
    check = np.all(np.linalg.norm(deltas, axis=1) > threshold)
    if check:
        return state
    else:
        return NullState


check_distances = python_app(_check_distances, executors=["default_htex"])



def test_metrics(dataset):
    # TODO: extracted from test_data.test_data_extract

    data = dataset[:2] + Dataset([NullState]) + dataset[3:5]
    forces = data.get("forces", elements=["Cu"])
    reference = np.zeros((5, 4, 3))
    reference[2, :] = np.nan  # ensure nan is in same place
    reference[:, 0] = np.nan  # ensure nan is in same place
    value = compute_rmse(forces, reference)

    # last three atoms are Cu
    forces = np.zeros((5, 4, 3))
    for i in range(5):
        forces[i, :] = data[i].result().per_atom.forces
    forces[:, 0] = np.nan
    assert np.allclose(
        value.result(),
        compute_rmse(forces, forces * np.zeros_like(forces)).result(),
    )
    unreduced = compute_rmse(
        forces, forces * np.zeros_like(forces), reduce=False
    ).result()
    assert len(unreduced) == 5
    unreduced_ = unreduced[np.array([0, 1, 3, 4], dtype=int)]
    assert np.allclose(
        np.sqrt(np.mean(np.square(unreduced_))),
        value.result(),
    )