"""
TODO: these functions seem to not be used anywhere in the codebase
 probably a good idea to remove these eventually
"""


def _check_equality(state0: Geometry, state1: Geometry) -> bool:
    """Check if two Geometry instances are equal"""
    return state0 == state1


check_equality = python_app(_check_equality, executors=["default_threads"])


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
