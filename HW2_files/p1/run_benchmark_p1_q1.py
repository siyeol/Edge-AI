#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sysfs_paths as sysfs
import subprocess
import time


def get_avail_freqs(cluster):
    """
    Obtain the available frequency for a cpu. Return unit in khz by default!
    """
    # Read cpu freq from sysfs_paths.py
    freqs = open(sysfs.fn_cluster_freq_range.format(cluster)).read().strip().split(' ')
    return [int(f.strip()) for f in freqs]

def get_cluster_freq(cluster_num):
    """
    Read the current cluster freq. cluster_num must be 0 (little) or 4 (big)
    """
    with open(sysfs.fn_cluster_freq_read.format(cluster_num), 'r') as f:
        return int(f.read().strip())

def set_user_space(clusters=None):
    """
    Set the system governor as 'userspace'. This is necessary before you can change the
    cluster/cpu freq to customized values
    """
    print("Setting userspace")
    clusters = [0, 4]
    for i in clusters:
        with open(sysfs.fn_cluster_gov.format(i), 'w') as f:
            f.write('userspace')

def set_cluster_freq(cluster_num, frequency):
    """
    Set customized freq for a cluster. Accepts frequency in KHz as int or string.
    cluster_num must be 0 (little) or 4 (big)
    """
    with open(sysfs.fn_cluster_freq_set.format(cluster_num), 'w') as f:
        f.write(str(frequency))
       
print('Available freqs for LITTLE cluster:', get_avail_freqs(0))
print('Available freqs for big cluster:', get_avail_freqs(4))
set_user_space()

set_cluster_freq(4, 2000000)   # big cluster
# Print current freq for the big cluster
print('Current freq for big cluster:', get_cluster_freq(4))

set_cluster_freq(0, 200000)   # small cluster
# Print current freq for the small cluster
print('Current freq for small cluster:', get_cluster_freq(0))


# Execution of your benchmark
start=time.time()
# Run the benchmark
command = "taskset --all-tasks 0x10 /home/student/HW2_files/TPBench.exe"   # 0x10: core 4
proc_ben = subprocess.call(command.split())

total_time = time.time() - start
print("Benchmark runtime:", total_time)
