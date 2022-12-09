import os
import tempfile
import yaff
import molmod
import numpy as np

from parsl.app.app import python_app
from parsl.data_provider.files import File

from autolearn.execution import Container
from autolearn.utils import _new_file, copy_data_future
from autolearn.sampling.utils import set_path_hills_plumed, get_bias_plumed


class Bias(Container):
    """Represents a PLUMED bias potential"""

    def __init__(self, context, plumed_input):
        super().__init__(context)

        assert 'PRINT' not in plumed_input
        self.plumed_input = plumed_input

        if self.kind[0] == 'METAD': # create hills file
            self.hills_future = File(_new_file(context))
        else:
            self.hills_future = None

#    def evaluate(self, dataset):
#        def evaluate_barebones(bias, dataset):
#            values = np.zeros((len(dataset), 2)) # column 0 for CV, 1 for bias
#            system = yaff.System(
#                    numbers=dataset[0].atoms.get_atomic_numbers(),
#                    pos=dataset[0].atoms.get_positions() * molmod.units.angstrom,
#                    rvecs=dataset[0].atoms.get_cell() * molmod.units.angstrom,
#                    )
#            bias.stage()
#            part_plumed = yaff.external.ForcePartPlumed(
#                    system,
#                    timestep=1000*molmod.units.femtosecond, # does this matter?
#                    restart=0,
#                    fn=bias.files['input'],
#                    fn_log=bias.files['log'],
#                    )
#            ff = yaff.pes.ForceField(system, [part_plumed])
#            for i, sample in enumerate(dataset):
#                ff.update_pos(sample.atoms.get_positions() * molmod.units.angstrom)
#                ff.update_rvecs(sample.atoms.get_cell() * molmod.units.angstrom)
#                part_plumed.plumedstep = i
#                values[i, 1] = ff.compute() / molmod.units.kjmol
#                part_plumed.plumed.cmd('update')
#            part_plumed.plumed.cmd('update') # flush last
#            colvar = np.loadtxt(bias.files['cv'])[:, 1]
#            values[:, 0] = colvar
#            bias.close()
#            return values
#        evaluate_electron = ct.electron(
#                evaluate_barebones,
#                executor='local',
#                )
#        return evaluate_electron(self, dataset)
#
#    def stage(self):
#        assert tuple(list(self.files.values())) == (None, None, None, None)
#
#        # generate temporary files
#        for filename in self.files.keys():
#            tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
#            tmp.close()
#            self.files[filename] = tmp.name
#
#        # generate temporary hills file and store its name in plumed input
#        if self.uses_hills:
#            plumed_input = set_path_hills_plumed(
#                    self.plumed_input,
#                    self.files['hills'],
#                    )
#            plumed_input = 'RESTART\n' + plumed_input # ensures hills are read
#        else:
#            plumed_input = self.plumed_input
#
#        # add print colvar
#        plumed_input += '\nPRINT STRIDE=1 ARG={} FILE={}'.format(
#                self.cv,
#                self.files['cv'],
#                )
#        with open(self.files['input'], 'w') as f:
#            f.write(plumed_input)
#
#    def close(self, unsafe=False):
#        if not unsafe: # load contents of HILLS file
#            if self.uses_hills:
#                with open(self.files['hills'], 'r') as f:
#                    self.hills = f.read()
#        for name, filepath in self.files.items():
#            if os.path.exists(filepath):
#                os.unlink(filepath)
#            self.files[name] = None

    def copy(self):
        bias = Bias(self.context, self.plumed_input)
        if bias.hills_future is not None:
            bias.hills_future = copy_data_future(
                    inputs=[bias.hills_future],
                    outputs=[File(_new_file(self.context))],
                    ).outputs[0]

    @property
    def kind(self):
        return get_bias_plumed(self.plumed_input)

    @classmethod
    def create_apps(context):
        pass
