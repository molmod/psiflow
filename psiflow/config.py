import math
import subprocess


def get_partitions():
    scontrol_output = subprocess.check_output(
        ["scontrol", "show", "partition"], text=True
    )
    partition_info = {}

    partitions = scontrol_output.strip().split("\n\n")
    for partition in partitions:
        lines = partition.strip().split("\n")
        partition_dict = {}
        for line in lines:
            pairs = line.split()
            for pair in pairs:
                key, value = pair.split("=", 1)
                partition_dict[key] = value
        partition_name = partition_dict.get("PartitionName")
        if partition_name:
            partition_info[partition_name] = partition_dict

    scontrol_output = subprocess.check_output(["scontrol", "show", "node"], text=True)
    node_info = {}

    nodes = scontrol_output.strip().split("\n\n")
    for node in nodes:
        lines = node.strip().split("\n")
        node_dict = {}
        for line in lines:
            pairs = line.split()
            for pair in pairs:
                key_value = pair.split("=", 1)
                if len(key_value) == 1:
                    key = key_value[0]
                    value = ""
                else:
                    assert len(key_value) == 2
                    key = key_value[0]
                    value = key_value[1]
                node_dict[key] = value
        if "Partitions" not in node_dict:
            continue
        partition_names = node_dict.get("Partitions").split(",")
        for partition_name in partition_names:
            if partition_name in partition_info:
                partition_info[partition_name]["node_dict"] = node_dict

    # get core and gpu count per node
    for partition in partition_info:
        cores_per_socket = int(partition_info[partition]["node_dict"]["CoresPerSocket"])
        sockets = int(partition_info[partition]["node_dict"]["Sockets"])
        partition_info[partition]["cores_per_node"] = cores_per_socket * sockets

        gres_list = partition_info[partition]["node_dict"]["CfgTRES"].split(",")

        gpu_count = 0
        gpu_type = "nvidia"
        for gres in gres_list:
            if "gpu" in gres:
                gpu_count = int(gres.split("=")[1])
                if "mi" in gres:
                    gpu_type = "amd"
        partition_info[partition]["gpus_per_node"] = gpu_count
        partition_info[partition]["gpu_type"] = gpu_type
    return partition_info


def prompt(question, options=None):
    if options is not None:
        print(question)
        for i, option in enumerate(options, 1):
            print(f"\t{i}. {option}")
        print()
        while True:
            try:
                choice = int(input("   select an option (number): "))
                if 1 <= choice <= len(options):
                    answer = options[choice - 1]
                    print()
                    break
                else:
                    print("Please select a valid option number.")
            except ValueError:
                print("Please enter a number.")
    else:
        answer = input(question)
        print()
    return answer


def write_yaml_with_comments(filename, data, comments, indent=0):
    def write_dict(file, data, comments, indent):
        for key, value in data.items():
            # Write the actual key-value pair
            if isinstance(value, dict):
                file.write(f"{' ' * indent}{key}:\n")
                if key in comments and isinstance(comments[key], dict):
                    write_dict(file, value, comments[key], indent + 2)
                else:
                    write_dict(file, value, {}, indent + 2)
            else:
                try:
                    value = int(value)
                except ValueError:
                    pass
                if type(value) is not str:
                    file.write(f"{' ' * indent}{key}: {value}\n")
                else:
                    file.write(f"{' ' * indent}{key}:" + f' "{value}"\n')
                # If the key exists in the comments dictionary, write the commented value
                if key in comments:
                    comment = comments[key]
                    file.write(f"{' ' * indent}# {key}: {comment}\n")

    with open(filename, "w") as file:
        write_dict(file, data, comments, indent)


def setup_slurm_config() -> dict:
    config = {}
    partition_info = get_partitions()

    container_engine = None
    for engine in ["singularity", "apptainer"]:
        if subprocess.run(["which", engine], capture_output=True).returncode == 0:
            container_engine = engine
    if container_engine is None:
        print("NO APPTAINER/SINGULARITY CONTAINER RUNTIME IS DETECTED.")
        print(
            "Either contact your administrator for information on how to activate them,"
        )
        print(
            "or set up all dependencies (PyTorch, CP2K/GPAW/ORCA, plumed, i-PI) manually."
        )
        print()
    else:
        print("DETECTED CONTAINER ENGINE: {}".format(container_engine))
        config["container_engine"] = container_engine
        print()

    config_qm = {}
    config_qm["slurm"] = {}
    print("## QM CALCULATIONS (e.g. DFT singlepoints) ##")
    print()
    partition = prompt(
        "   Which partition would you like to use?",
        list(partition_info),
    )
    config_qm["slurm"]["partition"] = partition

    max_cores = partition_info[partition]["cores_per_node"]
    s = f"   How many cores would you like to use for each singlepoint? [1 - {max_cores}] "
    cores_per_worker = prompt(s, options=None)
    config_qm["cores_per_worker"] = cores_per_worker
    config_qm["slurm"]["cores_per_node"] = cores_per_worker

    s = "   What is the largest number of blocks ( = SLURM jobs) that is allowed to run at any given moment?  "
    max_nodes = partition_info[partition]["MaxNodes"]
    if max_nodes != "UNLIMITED":
        s += f"\n   Based on the current SLURM config, it appears that there exists a limit of {max_nodes} jobs for this partition. "
    max_blocks = prompt(s, options=None)
    config_qm["slurm"]["max_blocks"] = max_blocks
    config_qm["slurm"]["min_blocks"] = 0
    config_qm["slurm"]["init_blocks"] = 0
    config_qm["slurm"]["nodes_per_block"] = 1

    max_time = partition_info[partition]["MaxTime"]
    s = "   What is the requested walltime for each block?\n"
    s += "   This should be at least as large as the expected duration of a single QM energy/force evaluation.\n"
    s += "   on the specified number of cores. SLURM reports a max of {}. \n".format(
        max_time
    )
    s += "   Format as HH:MM:SS (e.g. 4 hours would be 04:00:00)\n   "
    max_walltime = prompt(s, options=None)
    config_qm["slurm"]["walltime"] = max_walltime

    s = "   Which SLURM account should be used for these jobs? "
    account = prompt(s, options=None)
    config_qm["slurm"]["account"] = account

    if container_engine is not None:
        launch_cp2k = "{} exec -e --no-init".format(container_engine)
        launch_cp2k += "oras://ghcr.io/molmod/cp2k:2023.2 /opt/entry.sh"
        launch_cp2k += "mpirun -np {}".format(cores_per_worker)
        launch_cp2k += " -x OMP_NUM_THREADS=1 cp2k.psmp -i cp2k.inp"
        launch_gpaw = "{} exec -e --no-init".format(container_engine)
        launch_gpaw += "oras://ghcr.io/molmod/gpaw:24.1 /opt/entry.sh"
        launch_gpaw += "mpirun -np {}".format(cores_per_worker)
        launch_gpaw += " gpaw python /opt/run_gpaw.py input.json"
    else:
        launch_cp2k = None
        launch_gpaw = None
    config_cp2k = dict(config_qm)
    config_cp2k["launch_command"] = ""
    config_gpaw = dict(config_qm)
    config_gpaw["launch_command"] = ""
    config["CP2K"] = config_cp2k
    config["GPAW"] = config_gpaw

    config_training = {}
    config_training["slurm"] = {}
    print()
    print()
    print("## MODEL TRAINING (training ML potential to QM reference data) ##\n")

    names = list(partition_info)
    for i, name in enumerate(names):
        gpu_count = partition_info[name]["gpus_per_node"]
        if gpu_count > 0:
            names[i] = name + " (GPUs per node: {})".format(gpu_count)
        else:
            names[i] = name + " (no GPUs detected)"
    partition = prompt(
        "   Which partition would you like to use? Keep in mind that training requires the availability of a GPU.",
        names,
    ).split()[0]
    gpu_count = partition_info[name]["gpus_per_node"]
    if gpu_count > 0:  # auto setup
        cores_per_worker = math.floor(
            partition_info[partition]["cores_per_node"] / gpu_count
        )
    else:
        cores_per_worker = ""
    config_training["cores_per_worker"] = cores_per_worker
    config_training["slurm"]["scheduler_options"] = "#SBATCH --gres=gpu:1"
    config_training["cores_per_worker"] = cores_per_worker
    config_training["slurm"]["cores_per_node"] = cores_per_worker
    config_training["slurm"]["partition"] = partition
    config_training["gpu"] = True

    s = "   Which command(s) should be executed in order to load CUDA/ROCm libraries at the start of each job?"
    s += ' (e.g. "ml env/release/2023.1 && ml CUDA/11.8")\n   '
    worker_init = prompt(s)
    config_training["slurm"]["worker_init"] = worker_init

    s = "   Which SLURM account should be used for these jobs? "
    account = prompt(s, options=None)
    config_training["slurm"]["account"] = account

    s = "   What is the largest number of blocks ( = SLURM jobs) that is allowed to run at any given moment?  "
    max_nodes = partition_info[partition]["MaxNodes"]
    if max_nodes != "UNLIMITED":
        s += f"\n   Based on the current SLURM config, it appears that there exists a limit of {max_nodes} jobs for this partition. "
    max_blocks = prompt(s, options=None)
    config_training["slurm"]["max_blocks"] = max_blocks
    config_training["slurm"]["min_blocks"] = 0
    config_training["slurm"]["init_blocks"] = 0
    config_training["slurm"]["nodes_per_block"] = 1

    max_time = partition_info[partition]["MaxTime"]
    s = "   What is the requested walltime for each block?\n"
    s += "   This should be at least as large as the expected duration of a single ML potential training run.\n"
    s += "   SLURM reports a max of {}. ".format(max_time)
    s += "Format as HH:MM:SS (e.g. 4 hours would be 04:00:00)\n   "
    max_walltime = prompt(s, options=None)
    config_training["slurm"]["walltime"] = max_walltime
    config["ModelTraining"] = config_training

    config_eval = {}
    config_eval["slurm"] = {}
    print()
    print()
    print("## MODEL EVALUATION  (e.g. molecular dynamics using i-PI) ##\n")

    partition = prompt(
        "   Which partition would you like to use? ",
        list(partition_info),
    )
    config_training["slurm"]["partition"] = partition
    gpu_count = partition_info[partition]["gpus_per_node"]

    gpu = prompt(
        "   Do you want to employ your ML potential on GPUs? ",
        list(["yes", "no"]),
    )
    if gpu == "yes":
        config_eval["gpu"] = True
        cores_per_worker = math.floor(
            partition_info[partition]["cores_per_node"] / gpu_count
        )
        config_eval["cores_per_worker"] = cores_per_worker
        config_eval["slurm"]["cores_per_node"] = partition_info[partition][
            "cores_per_node"
        ]
        config_eval["slurm"]["scheduler_options"] = "#SBATCH --gres=gpu:1"
        config_eval["worker_init"] = config_training["slurm"]["worker_init"]

    s = "   Which SLURM account should be used for these jobs? "
    account = prompt(s, options=None)
    config_eval["slurm"]["account"] = account

    s = "   What is the largest number of blocks ( = SLURM jobs) that is allowed to run at any given moment?  "
    max_nodes = partition_info[partition]["MaxNodes"]
    if max_nodes != "UNLIMITED":
        s += f"\n   Based on the current SLURM config, it appears that there exists a limit of {max_nodes} jobs for this partition. "
    max_blocks = prompt(s, options=None)
    config_eval["slurm"]["max_blocks"] = max_blocks
    config_eval["slurm"]["min_blocks"] = 0
    config_eval["slurm"]["init_blocks"] = 0

    max_time = partition_info[partition]["MaxTime"]
    s = "   What is the requested walltime for each block?\n"
    s += "   This should be at least as large as the expected duration of the longest MD simulation.\n"
    s += "   SLURM reports a max of {}. ".format(max_time)
    s += "Format as HH:MM:SS (e.g. 4 hours would be 04:00:00)\n   "
    max_walltime = prompt(s, options=None)
    config_eval["slurm"]["walltime"] = max_walltime
    config["ModelEvaluation"] = config_eval

    write_yaml_with_comments("config.yaml", config, {})
    return config
