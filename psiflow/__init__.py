import parsl

from .execution import ExecutionContextLoader

load = ExecutionContextLoader.load
parse_config = ExecutionContextLoader.parse_config
context = ExecutionContextLoader.context
wait = parsl.wait_for_current_tasks
