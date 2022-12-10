import os
import tempfile
import yaff
import molmod
import numpy as np

from parsl.app.app import python_app
from parsl.data_provider.files import File

from autolearn.execution import Container, ModelExecutionDefinition
from autolearn.utils import _new_file, copy_data_future
from autolearn.sampling.utils import set_path_hills_plumed, get_bias_plumed



def evaluate_bias(plumed_input, kind, inputs=[]):
    import tempfile
    import os
    import numpy as np
    import yaff
    yaff.log.set_level(yaff.log.silent)
    import molmod
    from autolearn.sampling.utils import ForcePartASE, create_forcefield, \
            ForceThresholdExceededException, try_manual_plumed_linking, \
            set_path_hills_plumed
    from autolearn.dataset import read_dataset
    dataset = read_dataset(slice(None), inputs=[inputs[0]])
    values = np.zeros((len(dataset), 2)) # column 0 for CV, 1 for bias
    system = yaff.System(
            numbers=dataset[0].get_atomic_numbers(),
            pos=dataset[0].get_positions() * molmod.units.angstrom,
            rvecs=dataset[0].get_cell() * molmod.units.angstrom,
            )
    try_manual_plumed_linking()
    if len(inputs) == 2:
        path_hills = inputs[1]
        plumed_input = 'RESTART\n' + plumed_input # ensures hills are read
    else:
        path_hills = None
    if path_hills is not None: # path to hills
        plumed_input = set_path_hills_plumed(plumed_input, path_hills)
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    colvar = tmp.name # dummy log file
    #arg = kind[1] + ',' + kind[2] + '.bias'
    arg = kind[1]
    plumed_input += '\nFLUSH STRIDe=1' # has to come before PRINT?!
    plumed_input += '\nPRINT STRIDE=1 ARG={} FILE={}'.format(arg, colvar)
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
        f.write(plumed_input) # write input
        path_input = f.name
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    tmp.close()
    plumedlog = tmp.name # dummy log file
    part_plumed = yaff.external.ForcePartPlumed(
            system,
            timestep=1000*molmod.units.femtosecond, # does this matter?
            restart=0,
            fn=path_input,
            fn_log=plumedlog,
            )
    ff = yaff.pes.ForceField(system, [part_plumed])
    for i, atoms in enumerate(dataset):
        ff.update_pos(atoms.get_positions() * molmod.units.angstrom)
        ff.update_rvecs(atoms.get_cell() * molmod.units.angstrom)
        part_plumed.plumedstep = i
        values[i, 1] = ff.compute() / molmod.units.kjmol
        part_plumed.plumed.cmd('update')
    part_plumed.plumed.cmd('update') # flush last
    values[:, 0] = np.loadtxt(colvar)[:, 1]
    os.unlink(plumedlog)
    os.unlink(colvar)
    os.unlink(path_input)
    return values


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

    def evaluate(self, dataset):
        inputs = [dataset.data_future]
        if self.hills_future is not None:
            inputs.append(self.hills_future)
        return self.context.apps(Bias, 'evaluate')(
                self.plumed_input,
                self.kind,
                inputs=inputs,
                )

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
    def create_apps(cls, context):
        executor_label = context[ModelExecutionDefinition].executor_label
        app_evaluate = python_app(evaluate_bias, executors=[executor_label])
        context.register_app(cls, 'evaluate', app_evaluate)
