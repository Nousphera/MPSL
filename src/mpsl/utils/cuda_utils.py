# As per https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/12 last reply by Seonjun_Kim
import os
import subprocess
import sys


def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode('utf-8')[:-1]

    return out


def get_free_gpu_indices():
    out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
    out = (out.split('\n'))[1:]
    out = [l for l in out if '--' not in l]

    total_gpu_num = int(len(out)/5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])

    out = run_cmd('nvidia-smi --query-compute-apps=gpu_bus_id --format=csv')
    gpu_bus_ids_in_use = (out.split('\n'))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]


def get_free_cuda_device_name(global_args):
    chosen_gpu_id = global_args.gpu_id

    if chosen_gpu_id is not None:
        return f'cuda:{chosen_gpu_id}'

    is_windows_os = sys.platform.startswith('win')

    if is_windows_os:
        return 'cuda'

    free_gpu_indices = get_free_gpu_indices()

    if len(free_gpu_indices) == 0:
        raise Exception('No CUDA devices available.')

    gpu_idx = free_gpu_indices[0]
    # Ensuring that only the necessary GPU is allocated
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

    return f'cuda:{gpu_idx}'
