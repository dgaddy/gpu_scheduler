import subprocess
import os
import fcntl
import sys
import argparse
from collections import namedtuple, defaultdict, Counter
from datetime import datetime
import random
import time
from contextlib import ExitStack
import pprint

lock_base_directory = '/shared/group/gpu_scheduler/locks'

MINIMUM_PRIVELIGED_JOBS = 1
MINIMUM_NON_PRIVELIGED_JOBS = 0

def get_hostname():
    result = subprocess.run(['hostname'], stdout=subprocess.PIPE, encoding='ascii')
    return result.stdout.strip()

def get_username():
    result = subprocess.run(['id','-un'], stdout=subprocess.PIPE, encoding='ascii')
    return result.stdout.strip()

GPUInfo = namedtuple('GPUInfo', ['index', 'name'])
ProcInfo = namedtuple('ProcInfo', ['pid', 'user', 'gpu_index', 'start_time', 'preemption_candidate'])

def get_gpu_infos():
    # use `nvidia-smi --help-query-gpu` to get all query options
    result = subprocess.run(['nvidia-smi','--query-gpu=gpu_uuid,index,name','--format=csv,noheader,nounits'], stdout=subprocess.PIPE, encoding='ascii')
    gpu_infos = {}
    for line in result.stdout.split('\n'):
        if len(line) == 0:
            continue
        gpu_uuid, index, name = [v.strip() for v in line.split(',')]
        gpu_infos[gpu_uuid] = GPUInfo(index, name)
    return gpu_infos

def get_gpu_processes():
    # use `nividi-smi --help-query-compute-apps` to get all query options
    result = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'], stdout=subprocess.PIPE, encoding='ascii')
    gpu_processes = defaultdict(list)
    for line in result.stdout.split('\n'):
        if len(line) == 0:
            continue
        pid, gpu_uuid = [v.strip() for v in line.split(',')]
        gpu_processes[gpu_uuid].append(pid)
    return gpu_processes

def get_process_stats(field):
    result = subprocess.run(['ps','-eo','pid,{}'.format(field),'--no-headers'], stdout=subprocess.PIPE, encoding='ascii')
    stats = {}
    for line in result.stdout.split('\n'):
        if len(line) == 0:
            continue
        toks = line.strip().split()
        pid = toks[0]
        stat = ' '.join(toks[1:])
        stats[pid] = stat
    return stats

def get_process_users():
    return get_process_stats('user')

def get_process_starts():
    lstarts = get_process_stats('lstart')
    return {
        pid: datetime.strptime(start, '%a %b %d %H:%M:%S %Y').timestamp()
        for pid, start in lstarts.items()
    }

def get_descendent_processes(pid):
    result = subprocess.run(['pgrep', '--pgroup', pid], stdout=subprocess.PIPE, encoding='ascii')
    return [line for line in result.stdout.split('\n') if line]

def process_is_running(pid):
    try:
        subprocess.check_call('kill -0 ' + pid, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        return False
    return True

# def kill_process(pid, max_attempts=None):
#     if max_attempts is None:
#         max_attempts = 1
#     assert max_attempts > 0
#     is_running = True
#     attempts = 0
#     while is_running and attempts < max_attempts:
#         subprocess.run('sudo kill ' + pid, shell=True)
#         time.sleep(1)
#         is_running = process_is_running(pid)
#     return not is_running

def kill_process(pid, max_wait_time=0, recursive=True, signal=15):
    assert max_wait_time >= 0
    is_running = True
    wait_time = 0
    if recursive:
        to_kill = "-{}".format(pid)
        to_check = get_descendent_processes(pid)
    else:
        to_kill = "{}".format(pid)
        to_check = [pid]
    subprocess.run('sudo kill -{} {}'.format(signal, to_kill), shell=True)
    still_running = [p for p in to_check if process_is_running(p)]
    while still_running and wait_time < max_wait_time:
        time.sleep(1)
        still_running = [p for p in to_check if process_is_running(p)]
        wait_time += 1
    return still_running

def get_locking_pid(lock_filename):
    # note actually returns process accessing the file, not just locking
    result = subprocess.run('sudo lsof -t '+lock_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='ascii')
    result_list = result.stdout.strip().split()
    if len(result_list) > 1:
        print('Warning: mutliple processes accessing lock file')
    if len(result_list) == 0:
        return None
    else:
        return result_list[0]

def lock_and_run(lock_filenames, command, env={}):
    with ExitStack() as stack:
        for filename in lock_filenames:
            f = stack.enter_context(open(filename, 'wb'))
            try:
                fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
                # the lock is released when the file is closed (or worst case when this process dies)
            except:
                return False, f # failed to aquire lock

        print('Running command:', command)
        if 'CUDA_VISIBLE_DEVICES' in env:
            print('GPU(s):', env['CUDA_VISIBLE_DEVICES'])

        # If we used subprocess.run, Ctrl-C (keyboard interrupt) is not passed to
        # the subprocess, so run in a polling loop and catch the interrupt.
        process = subprocess.Popen(command, shell=True, env=env)
        try:
            while True:
                returned = process.wait()
                if returned is None:
                    time.sleep(1)
                else:
                    break
        except KeyboardInterrupt:
            # hack: for some reason, kill_process(str(process.pid)) does not 
            # actually kill all subprocess, so kill this process instead (in the same
            # way it would be killed by another invocation of reserve.py).
            # The finally exception below will still be invoked, releasing the lock.
            print("Ctrl+C caught, sending kill to this process and its subprocesses")
            kill_process(str(os.getpid()))
        return True, None

def confirm(prompt):
    print(prompt, end='')
    inp = ""
    while inp not in ['y', 'n']:
        print(' y/n')
        inp = input().strip().lower()
    return inp == 'y'

def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--large-mem', action='store_true', help="request gpu with extra memory")
    parser.add_argument('--num-gpus', type=int, default=1, help='number of gpus to reserve')
    parser.add_argument('--no-inherit-environment', action='store_true', help="don't pass the current environment variables to the command")
    parser.add_argument('--preempt-wait-time', type=int, default=10, help='wait this many seconds for a preempted process to exit')
    parser.add_argument('command', nargs='*', help="command to run")
    return parser

def try_launch(args, lock_directory):
    gpus = get_gpu_infos()
    gpu_processes = get_gpu_processes()
    process_users = get_process_users()

    users_with_reservation = []
    all_start_times = get_process_starts()

    reserved_processes_by_user = defaultdict(list)

    available_gpu_locks = {} # lock_file : gpu_index

    for gpu, gpu_info in gpus.items():
        can_run = ((args.large_mem and '8000' in gpu_info.name) or 
                    (not args.large_mem and '8000' not in gpu_info.name))
        fn = os.path.join(lock_directory, 'gpu{}'.format(gpu_info.index))
        used_by = get_locking_pid(fn)
        if used_by is None and len(gpu_processes[gpu]) > 0:
            # process is using without a reservation
            process_infos = ' '.join(['{}:{}'.format(pid, process_users[pid]) for pid in gpu_processes[gpu]])
            print('Warning: processes with no reservation on gpu {} - {}'.format(gpu_info.index, process_infos))
            used_by = gpu_processes[gpu][0]

        if used_by is None:
            if can_run:
                available_gpu_locks[fn] = gpu_info.index
        else:
            reserved_processes_by_user[process_users[used_by]].append(ProcInfo(
                pid=used_by,
                user=process_users[used_by],
                gpu_index=gpu_info.index,
                start_time=all_start_times[used_by],
                preemption_candidate=can_run,
            ))

    while len(available_gpu_locks) >= args.num_gpus:
        if args.no_inherit_environment:
            env = {}
        else:
            env = os.environ

        files_to_lock = list(available_gpu_locks.keys())[:args.num_gpus]
        env['CUDA_VISIBLE_DEVICES'] = ','.join(available_gpu_locks[f] for f in files_to_lock)

        success, blocking_file = lock_and_run(files_to_lock, ' '.join(args.command), env)

        if success:
            sys.exit(0)
        else:
            print('Failed to aquire lock on', blocking_file)
            del available_gpu_locks[blocking_file]
            #TODO: add to reserved_processes_by_user

    return reserved_processes_by_user

def main():
    args = make_arg_parser().parse_args()
    lock_directory = os.path.join(lock_base_directory, get_hostname())
    user = get_username()

    with open(os.path.join(lock_directory, 'privileged_users'), 'r') as f:
        privileged_users = f.read().split()

    reserved_processes_by_user = try_launch(args, lock_directory)
    print('All gpus reserved')
    user_reservation_counts = {
        user: len(procs)
        for user, procs in reserved_processes_by_user.items()
    }
    user_reservation_counts_usable = {
        user: len([p for p in procs if p.preemption_candidate])
        for user, procs in reserved_processes_by_user.items()
    }
    print('Users with a reservation:', user_reservation_counts)
    print('Users with a reservation on usable gpus:', user_reservation_counts_usable)
    # by_gpu = defaultdict(list)
    # for user, procs in reserved_processes_by_user.items():
    #     for proc in procs:
    #         by_gpu[proc.gpu_index].append((user, proc))
    # pprint.pprint(by_gpu)

    if user in privileged_users and user not in user_reservation_counts_usable and args.num_gpus == 1:
        reserved = list(reserved_processes_by_user.items())
        # shuffle before sorting to break ties non-deterministically
        random.shuffle(reserved)
        for user_to_boot, users_processes in sorted(reserved, key=lambda tpl: len(tpl[1]), reverse=True):
            preemption_candidates = [proc_info for proc_info in users_processes if proc_info.preemption_candidate]
            over_minimum = ((user_to_boot in privileged_users and len(users_processes) > MINIMUM_PRIVELIGED_JOBS) or
                            (user_to_boot not in privileged_users and len(users_processes) > MINIMUM_NON_PRIVELIGED_JOBS))
            if preemption_candidates and over_minimum:
                # preempt the newest preemptable process by this user
                process_to_preempt = max(preemption_candidates, key=lambda pinfo: pinfo.start_time)
                do_kill = confirm('Do you want to preempt {}:{} on gpu {}?'.format(
                    user_to_boot, process_to_preempt.pid, process_to_preempt.gpu_index
                ))
                if do_kill:
                    still_running = kill_process(process_to_preempt.pid, args.preempt_wait_time)
                    if not still_running:
                        print("{}:{} subprocesses no longer running; attempting to launch".format(
                            user_to_boot, process_to_preempt.pid
                        ))
                        try_launch(args, lock_directory)
                    else:
                        print("waited {} seconds after kill signal but {}:{} {} is still running; consider manually preempting".format(
                            args.preempt_wait_time, user_to_boot, process_to_preempt.pid, still_running
                        ))
                break

if __name__ == '__main__':
    main()
