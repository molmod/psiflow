import logging
from pathlib import Path

import parsl


def setup_logging(file: Path, level=logging.INFO) -> None:
    """Setup the Psiflow parent logger"""
    logger = logging.getLogger('psiflow')
    logger.setLevel(level)
    logger.propagate = False  # do not propagate messages to root logger

    fh = logging.FileHandler(file)
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # keep Parsl logs in the parsl.log file
    logging.getLogger("parsl").propagate = False



def log_dfk_tasks(verbose: bool = False):
    """Get an overview of all tasks stored in the parsl DFK. For debugging purposes."""
    dfk = parsl.dfk()
    parsl.wait_for_current_tasks()
    log = ["- Parsl task overview -"]
    if not verbose:
        log += [f"{i}\t{d['func_name']}" for i, d in dfk.tasks.items()]
        log.append("- Parsl task overview -")
        print(*log, sep="\n")
        return

    for i, d in dfk.tasks.items():
        args = [(_.split("/")[-1] if isinstance(_, str) else _) for _ in d["args"]]
        if "inputs" in (kwargs := d["kwargs"]):
            kwargs["inputs"] = [f.filename for f in kwargs["inputs"]]
        if "outputs" in kwargs:
            kwargs["outputs"] = [f.filename for f in kwargs["outputs"]]
        log.append(f"\n{i}\t{d['func_name']:<30}\n{args}\n{kwargs}")
    log.append("- Parsl task overview -")
    print(*log, sep="\n")