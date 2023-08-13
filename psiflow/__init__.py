import parsl

from .execution import ExecutionContextLoader


load    = ExecutionContextLoader.load
context = ExecutionContextLoader.context
wait    = parsl.wait_for_current_tasks
