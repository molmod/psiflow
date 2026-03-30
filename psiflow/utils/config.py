PSIFLOW_INTERNAL = "psiflow_internal"
PARSL_LOGFILE = "parsl.log"
PSIFLOW_LOGFILE = "psiflow.log"
CONTEXT_DIR = "context_dir"


DEFAULT_CONFIG = """
parsl_log_level: WARNING
psiflow_log_level: WARNING
usage_tracking: 3
default_threads: 8
tmpdir_root: /tmp
keep_tmpdirs: false

ModelEvaluation:
  executor: threadpool
  max_threads: 2

ModelTraining:
  executor: threadpool
  max_threads: 2
"""

