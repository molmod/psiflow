from psiflow.hamiltonians import EinsteinCrystal


def test_einstein(context, data):
    EinsteinCrystal(data[0], force_constant=1)
