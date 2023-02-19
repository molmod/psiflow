import os
import math
import time
import logging

from typing import Optional

import parsl.providers.slurm.slurm
from parsl.channels import LocalChannel
from parsl.channels.base import Channel
from parsl.launchers import SingleNodeLauncher
from parsl.launchers.launchers import Launcher
from parsl.providers.cluster_provider import ClusterProvider
from parsl.providers.base import JobState, JobStatus
from parsl.providers.slurm.template import template_string
from parsl.utils import RepresentationMixin, wtime_to_minutes


logger = logging.getLogger(__name__)


translate_table = {
    'PD': JobState.PENDING,
    'R': JobState.RUNNING,
    'CA': JobState.CANCELLED,
    'CF': JobState.PENDING,  # (configuring),
    'CG': JobState.RUNNING,  # (completing),
    'CD': JobState.COMPLETED,
    'F': JobState.FAILED,  # (failed),
    'TO': JobState.TIMEOUT,  # (timeout),
    'NF': JobState.FAILED,  # (node failure),
    'RV': JobState.FAILED,  # (revoked) and
    'SE': JobState.FAILED   # (special exit state)
}


class SlurmProviderVSC(parsl.providers.slurm.slurm.SlurmProvider):
    """Specifies cluster and partition for sbatch, scancel, and squeue"""

    def __init__(self, cluster=None, **kwargs):
        super().__init__(**kwargs)
        self.cluster = cluster
        self.scheduler_options += '#SBATCH --export=NONE\n'

        # both cluster and partition need to be specified
        assert self.cluster is not None
        assert self.partition is not None


    def submit(self, command, tasks_per_node, job_name="parsl.slurm"):
        """Submit the command as a slurm job.

        This function differs in its parent in the self.execute_wait()
        call, in which the slurm partition is explicitly passed as a command
        line argument as this is necessary for some SLURM-configered systems
        (notably, Belgium's HPC infrastructure).
        In addition, the way in which the job_id is extracted from the returned
        log after submission is slightly modified, again to account for
        the specific cluster configuration of HPCs in Belgium.

        Parameters
        ----------
        command : str
            Command to be made on the remote side.
        tasks_per_node : int
            Command invocations to be launched per node
        job_name : str
            Name for the job
        Returns
        -------
        None or str
            If at capacity, returns None; otherwise, a string identifier for the job
        """

        scheduler_options = self.scheduler_options
        worker_init = self.worker_init
        if self.mem_per_node is not None:
            scheduler_options += '#SBATCH --mem={}g\n'.format(self.mem_per_node)
            worker_init += 'export PARSL_MEMORY_GB={}\n'.format(self.mem_per_node)
        if self.cores_per_node is not None:
            cpus_per_task = math.floor(self.cores_per_node / tasks_per_node)
            scheduler_options += '#SBATCH --cpus-per-task={}'.format(cpus_per_task)
            worker_init += 'export PARSL_CORES={}\n'.format(cpus_per_task)

        job_name = "{0}.{1}".format(job_name, time.time())

        script_path = "{0}/{1}.submit".format(self.script_dir, job_name)
        script_path = os.path.abspath(script_path)


        job_config = {}
        job_config["submit_script_dir"] = self.channel.script_dir
        job_config["nodes"] = self.nodes_per_block
        job_config["tasks_per_node"] = tasks_per_node
        job_config["walltime"] = wtime_to_minutes(self.walltime)
        job_config["scheduler_options"] = scheduler_options
        job_config["worker_init"] = worker_init
        job_config["user_script"] = command

        # Wrap the command
        job_config["user_script"] = self.launcher(command,
                                                  tasks_per_node,
                                                  self.nodes_per_block)

        self._write_submit_script(template_string, script_path, job_name, job_config)

        if self.move_files:
            channel_script_path = self.channel.push_file(script_path, self.channel.script_dir)
        else:
            channel_script_path = script_path

        submit_cmd = 'sbatch --clusters={2} --partition={1} {0}'.format(
                channel_script_path,
                self.partition,
                self.cluster,
                )
        retcode, stdout, stderr = self.execute_wait(submit_cmd)

        job_id = None
        if retcode == 0:
            for line in stdout.split('\n'):
                if line.startswith("Submitted batch job"):
                    #job_id = line.split("Submitted batch job")[1].strip()
                    job_id = line.split("Submitted batch job")[1].strip().split()[0]
                    self.resources[job_id] = {'job_id': job_id, 'status': JobStatus(JobState.PENDING)}
        else:
            logger.error("Submit command failed")
            logger.error("Retcode:%s STDOUT:%s STDERR:%s", retcode, stdout.strip(), stderr.strip())
        return job_id

    def _status(self):
        '''Returns the status list for a list of job_ids
        Args:
              self
        Returns:
              [status...] : Status list of all jobs
        '''
        job_id_list = ','.join(
            [jid for jid, job in self.resources.items() if not job['status'].terminal]
        )
        if not job_id_list:
            logger.debug('No active jobs, skipping status update')
            return

        cmd = "squeue --clusters={1} --noheader --format='%i %t' --job '{0}'".format(job_id_list, self.cluster)
        logger.debug("Executing %s", cmd)
        retcode, stdout, stderr = self.execute_wait(cmd)
        logger.debug("squeue returned %s %s", stdout, stderr)

        # Execute_wait failed. Do no update
        if retcode != 0:
            logger.warning("squeue failed with non-zero exit code {}".format(retcode))
            return

        jobs_missing = set(self.resources.keys())
        for line in stdout.split('\n'):
            if not line:
                # Blank line
                continue
            job_id, slurm_state = line.split()
            if slurm_state not in translate_table:
                logger.warning(f"Slurm status {slurm_state} is not recognized")
            status = translate_table.get(slurm_state, JobState.UNKNOWN)
            logger.debug("Updating job {} with slurm status {} to parsl state {!s}".format(job_id, slurm_state, status))
            self.resources[job_id]['status'] = JobStatus(status)
            jobs_missing.remove(job_id)

        # squeue does not report on jobs that are not running. So we are filling in the
        # blanks for missing jobs, we might lose some information about why the jobs failed.
        for missing_job in jobs_missing:
            logger.debug("Updating missing job {} to completed status".format(missing_job))
            self.resources[missing_job]['status'] = JobStatus(JobState.COMPLETED)

    def cancel(self, job_ids):
        ''' Cancels the jobs specified by a list of job ids
        Args:
        job_ids : [<job_id> ...]
        Returns :
        [True/False...] : If the cancel operation fails the entire list will be False.
        '''

        job_id_list = ' '.join(job_ids)
        retcode, stdout, stderr = self.execute_wait("scancel --clusters={1} {0}".format(job_id_list, self.cluster))
        rets = None
        if retcode == 0:
            for jid in job_ids:
                self.resources[jid]['status'] = JobStatus(JobState.CANCELLED)  # Setting state to cancelled
            rets = [True for i in job_ids]
        else:
            rets = [False for i in job_ids]

        return rets
