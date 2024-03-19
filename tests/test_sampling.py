from psiflow.hamiltonians import EinsteinCrystal, PlumedHamiltonian
from psiflow.sampling.ipi_utils import template
from psiflow.sampling.walker import Walker, partition


def test_setup_motion():
    pass
    # xml_str = "<data T="200"><test>4</test></data>"
    # assert ET.fromstring(xml_str).tag == 'data'
    # assert ET.fromstring(xml_str).attrib['T'] == '200'

    # element = ET.Element('data', attrib={'T': '200'})
    # element.append(ET.Element('test'))
    # tree = ET.ElementTree(element)


def test_walkers(dataset):
    plumed_str = """
UNITS LENGTH=A ENERGY=kj/mol TIME=fs
CV: DISTANCE ATOMS=1,2 NOPBC
RESTRAINT ARG=CV AT=1 KAPPA=1
"""
    plumed = PlumedHamiltonian(plumed_str)
    einstein = EinsteinCrystal(dataset[0], force_constant=0.1)
    einstein_ = EinsteinCrystal(dataset[0], force_constant=0.2)
    walker = Walker(dataset[0], dataset[1], einstein, temperature=300)
    assert walker.nvt
    assert not walker.npt
    assert not walker.pimd

    walkers = [walker]
    walkers.append(Walker(dataset[0], dataset[1], 0.5 * einstein_, nbeads=4))
    assert not Walker.is_similar(walkers[0], walkers[1])
    assert len(partition(walkers)) == 2
    walkers.append(Walker(dataset[0], dataset[1], einstein + plumed, nbeads=8))
    assert Walker.is_similar(walkers[1], walkers[2])
    assert len(partition(walkers)) == 2
    walkers.append(
        Walker(dataset[0], dataset[1], einstein, pressure=0, temperature=300)
    )
    assert not Walker.is_similar(walkers[0], walkers[-1])
    assert len(partition(walkers)) == 3
    walkers.append(
        Walker(dataset[0], dataset[1], einstein_, pressure=100, temperature=600)
    )
    assert len(partition(walkers)) == 3
    walkers.append(Walker(dataset[0], dataset[1], einstein, temperature=600))
    partitions = partition(walkers)
    assert len(partitions) == 3
    assert len(partitions[0]) == 2
    assert len(partitions[1]) == 2
    assert len(partitions[2]) == 2

    # nvt partition
    hamiltonians, weights_table = template(partitions[0])
    assert partitions[0][0].nvt
    assert len(hamiltonians) == 1
    assert weights_table[0] == ("TEMP", "EinsteinCrystal0")
    assert weights_table[1] == (300, 1.0)
    assert weights_table[2] == (600, 1.0)

    # pimd partition
    hamiltonians, weights_table = template(partitions[1])
    assert partitions[1][0].pimd
    assert len(hamiltonians) == 3
    assert weights_table[0] == (
        "TEMP",
        "EinsteinCrystal0",
        "EinsteinCrystal1",
        "PlumedHamiltonian0",
    )
    assert weights_table[1] == (300, 0.0, 0.5, 0.0)
    assert weights_table[2] == (300, 1.0, 0.0, 1.0)

    # npt partition
    hamiltonians, weights_table = template(partitions[2])
    assert partitions[2][0].npt
    assert len(hamiltonians) == 2
    assert weights_table[0] == (
        "TEMP",
        "PRESSURE",
        "EinsteinCrystal0",
        "EinsteinCrystal1",
    )
    assert weights_table[1] == (300, 0, 0.0, 1.0)
    assert weights_table[2] == (600, 100, 1.0, 0.0)
