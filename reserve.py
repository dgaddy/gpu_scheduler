import subprocess
import os
import fcntl
import sys
import argparse
from collections import namedtuple, defaultdict, Counter

lock_base_directory = './locks'

def get_hostname():
    result = subprocess.run(['hostname'], stdout=subprocess.PIPE, encoding='ascii')
    return result.stdout.strip()

def get_username():
    result = subprocess.run(['id','-un'], stdout=subprocess.PIPE, encoding='ascii')
    return result.stdout.strip()

GPUInfo = namedtuple('GPUInfo', ['index', 'name'])
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

def get_process_users():
    result = subprocess.run(['ps','-eo','pid,user','--no-headers'], stdout=subprocess.PIPE, encoding='ascii')
    users = {}
    for line in result.stdout.split('\n'):
        if len(line) == 0:
            continue
        pid, user = line.strip().split()
        users[pid] = user
    return users

def kill_process(pid):
    subprocess.run('sudo kill ' + pid, shell=True)

def get_locking_pid(lock_filename):
    # note actually returns process accessing the file, not just locking
    result = subprocess.run(['lsof','-t',lock_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='ascii')
    result_list = result.stdout.strip().split()
    if len(result_list) > 1:
        print('Warning: mutliple processes accessing lock file')
    if len(result_list) == 0:
        return None
    else:
        return result_list[0]

def lock_and_run(lock_filename, command, env={}):
    with open(lock_filename, 'wb') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except:
            return False # failed to aquire lock

        try:
            print('Running command:', command)
            if len(env) > 0:
                print('Env:', env)
            result = subprocess.run(command, shell=True, env=env)
            return True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            # note that worst case the lock is released when this process dies

def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--large-mem', action='store_true', help="request gpu with extra memory")
    parser.add_argument('command', nargs='*', help="command to run")
    return parser

def main():
    args = make_arg_parser().parse_args()
    print(args, args.command)
    lock_directory = os.path.join(lock_base_directory, get_hostname())
    def get_lock_filename(index):
        return os.path.join(lock_directory, 'gpu{}'.format(index))
    user = get_username()

    with open(os.path.join(lock_directory, 'privileged_users'), 'r') as f:
        privileged_users = f.read().split()

    gpus = get_gpu_infos()
    gpu_processes = get_gpu_processes()
    process_users = get_process_users()

    users_with_reservation = []
    for gpu, gpu_info in gpus.items():
        if args.large_mem and '8000' not in gpu_info.name:
            continue
        elif not args.large_mem and '8000' in gpu_info.name:
            continue
        fn = get_lock_filename(gpu_info.index)
        used_by = get_locking_pid(fn)
        while used_by is None:
            if len(gpu_processes[gpu]) > 0:
                process_infos = ' '.join(['{}:{}'.format(pid, process_users[pid]) for pid in gpu_processes[gpu]])
                print('Warning: processes with no reservation on gpu {} - {}'.format(gpu_info.index, process_infos))

            env = {'CUDA_VISIBLE_DEVICES':gpu_info.index}

            success = lock_and_run(fn, ' '.join(args.command), env)

            if success:
                sys.exit(0)
            else:
                print('Failed to aquire lock on gpu', gpu_info.index)
            used_by = get_locking_pid(fn)

        users_with_reservation.append(process_users[used_by])

    print('All gpus reserved')
    counts = Counter(users_with_reservation)
    print('Users with a reservation:', dict(counts))
    if user in privileged_users and user not in users_with_reservation:
        user_with_most = counts.most_common(1)[0][0]
        print('Suggest to preempt', user_with_most)

if __name__ == '__main__':
    main()
