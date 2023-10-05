import os
import sys

def write_script(cmds, duration, num_cpus, num_gpus, job_name, script_filename):
    f = open(script_filename, 'w')
    f.write('#!/bin/bash -l\n')
    f.write('\n')
    f.write('#$ -P ivc-ml\n')
    if num_gpus > 0:
        f.write('#$ -l gpus=%d\n'%(num_gpus))
        f.write('#$ -l gpu_c=5.0\n')

    f.write('#$ -pe omp %d\n'%(num_cpus))
    f.write('#$ -l h_rt=%s\n'%(duration))
    f.write('#$ -N %s\n'%(job_name))
    f.write('#$ -j y\n')
    f.write('#$ -m ea\n')
    f.write('\n')
    f.write('module load miniconda\n')
    f.write('conda activate vislang-domain-exploration\n')
    f.write('cd ~/data/vislang-domain-exploration\n')
    for cmd in cmds:
        f.write(cmd + '\n')

    f.write('\n')
    f.close()
