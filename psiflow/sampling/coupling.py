from psiflow.hamiltonians import PlumedHamiltonian


class Coupling:
    pass


class Metadynamics(Coupling, PlumedHamiltonian):
    def prepare(self):
        pass

    # def serialize(self):
    #    inputs = []
    #    for input_file in input_files:
    #        copied = copy_data_future(
    #            inputs=[input_file],
    #            outputs=[psiflow.context().new_file('metad_', '.txt')],
    #            ).outputs[0]
    #        inputs.append(copied)
    #    return dump_json(
    #        hamiltonian='PlumedHamiltonian',
    #        plumed_input=self.plumed_input,
    #        input_files=[f.filepath for f in inputs],
    #        inputs=inputs,  # wait for them to complete
    #        outputs=[psiflow.context().new_file("hamiltonian_", ".json")],
    #    ).outputs[0]

    def reset(self):
        pass
