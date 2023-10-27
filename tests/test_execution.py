import yaml

import psiflow


def test_parsing(tmp_path):
    contents = """
ModelEvaluation:
  simulation_engine: openmm
ModelTraining:
  gpu: true
"""
    yaml_dict = yaml.safe_load(contents)
    psiflow_config, definitions = psiflow.parse_config(yaml_dict)
    assert len(definitions) == 3  # reference inserted by default
