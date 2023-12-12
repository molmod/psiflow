# def wait():
#    print('asdlfkjaslkdf')
#    import time
#    time.sleep(3)
#    #import parsl
#    #parsl.wait_for_current_tasks()
#
# import atexit
# atexit.register(wait)

from .execution import ExecutionContextLoader, load_from_yaml  # noqa: F401

load = ExecutionContextLoader.load
parse_config = ExecutionContextLoader.parse_config
context = ExecutionContextLoader.context
wait = ExecutionContextLoader.wait
