import datetime
import errno
import os
import time
from collections import defaultdict, deque
import torch
import numpy as np

if torch.distributed.is_available() and torch.distributed.is_initialized():
    import torch.distributed as dist
else: 
    class MockDist:
        def is_available(self): return False
        def is_initialized(self): return False
        def get_world_size(self): return 1
        def get_rank(self): return 0
        def barrier(self): pass
        def all_reduce(self, tensor, op=None): pass
        def all_gather(self, tensor_list_or_data, tensor=None, async_op=False):
            if tensor is None:
                return [tensor_list_or_data]
            if isinstance(tensor_list_or_data, list) and len(tensor_list_or_data) > 0:
                 tensor_list_or_data[0].copy_(tensor) 
            return None
        
        class MockPickle:
            def dumps(self, obj): import pickle; return pickle.dumps(obj) 
            def loads(self, obj): import pickle; return pickle.loads(obj)
        _pickle = MockPickle()

    dist = MockDist()
    def is_dist_avail_and_initialized():
        if not torch.distributed.is_available(): return False
        if not torch.distributed.is_initialized(): return False
        return True
    def get_world_size():
        if not is_dist_avail_and_initialized(): return 1
        return torch.distributed.get_world_size()
    def get_rank():
        if not is_dist_avail_and_initialized(): return 0
        return torch.distributed.get_rank()


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if not self.deque: return float('nan')
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque: return float('nan')
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0: return float('nan')
        return self.total / self.count

    @property
    def max(self):
        if not self.deque: return float('-inf')
        return max(self.deque)

    @property
    def value(self):
        if not self.deque: return float('nan')
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )

def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try: serialized_data = torch.distributed._pickle.dumps(data)
    except AttributeError: import pickle; serialized_data = pickle.dumps(data)
    buffer = torch.tensor(bytearray(serialized_data), dtype=torch.uint8, device=device)
    local_size = torch.tensor([buffer.numel()], device=device)
    size_list = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = max(s.item() for s in size_list)
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=device) for _ in size_list]
    if local_size.item() != max_size:
        padding = torch.empty(size=(max_size - local_size.item(),), dtype=torch.uint8, device=device)
        buffer = torch.cat((buffer, padding), dim=0)
    dist.all_gather(tensor_list, buffer)
    result = []
    for size, tensor in zip(size_list, tensor_list):
        bytes_data = bytes(tensor[:size.item()].tolist())
        try: loaded_data = torch.distributed._pickle.loads(bytes_data)
        except AttributeError: import pickle; loaded_data = pickle.loads(bytes_data)
        result.append(loaded_data)
    return result

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2: return input_dict
    with torch.inference_mode():
        names = []; values = []
        for k in sorted(input_dict.keys()): names.append(k); values.append(input_dict[k])
        values = torch.stack(values, dim=0); dist.all_reduce(values)
        if average: values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor): v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    def __getattr__(self, attr):
        if attr in self.meters: return self.meters[attr]
        if attr in self.__dict__: return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    def __str__(self):
        return self.delimiter.join(f"{name}: {str(meter)}" for name, meter in self.meters.items())
    def synchronize_between_processes(self):
        for meter in self.meters.values(): meter.synchronize_between_processes()
    def add_meter(self, name, meter): self.meters[name] = meter
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header: header = ""
        start_time = time.time(); end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}"); data_time = SmoothedValue(fmt="{avg:.4f}")
        try: num_iterations = len(iterable)
        except TypeError: num_iterations = -1
        space_fmt = ":" + str(len(str(num_iterations))) if num_iterations > 0 else ""

        log_msg_parts = [header]
        if num_iterations > 0: log_msg_parts.append("[{iter_count" + space_fmt + "}/{total_iters}]")
        else: log_msg_parts.append("[{iter_count" + space_fmt + "}]")
        log_msg_parts.extend(["eta: {eta}", "{meters}", "time: {time}", "data: {data}"])
        if torch.cuda.is_available(): log_msg_parts.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg_parts)
        
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or (num_iterations > 0 and i == num_iterations - 1):
                eta_seconds = iter_time.global_avg * (num_iterations - i) if num_iterations > 0 and iter_time.count > 0 else 0
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                current_iter_display = str(num_iterations) if num_iterations > 0 else "?"
                
                format_args = {
                    "iter_count": i,
                    "total_iters": current_iter_display,
                    "eta": eta_string, "meters": str(self),
                    "time": str(iter_time), "data": str(data_time),
                }
                if torch.cuda.is_available(): format_args["memory"] = torch.cuda.max_memory_allocated() / MB
                print(log_msg.format(**format_args))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        avg_iter_time_str = f"({total_time / num_iterations:.4f} s / it)" if num_iterations > 0 and num_iterations > 0 else ""
        print(f"{header} Total time: {total_time_str} {avg_iter_time_str}")

def mkdir(path):
    try: os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST: raise

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print_dist(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force: builtin_print(*args, **kwargs)
    __builtin__.print = print_dist

def is_dist_avail_and_initialized():
    if not torch.distributed.is_available(): return False
    if not torch.distributed.is_initialized(): return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized(): return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized(): return 0
    return torch.distributed.get_rank()

def is_main_process(): return get_rank() == 0
def save_on_master(*args, **kwargs):
    if is_main_process(): torch.save(*args, **kwargs)

def init_distributed_mode(args): 
    if isinstance(args, dict):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args['rank']=int(os.environ["RANK"]); args['world_size']=int(os.environ["WORLD_SIZE"]); args['gpu']=int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ:
            args['rank']=int(os.environ["SLURM_PROCID"]); args['gpu']=args['rank']%torch.cuda.device_count()
        else: print("Not using distributed mode"); args['distributed']=False; return
        args['distributed']=True; torch.cuda.set_device(args['gpu']); dist_url=args.get('dist_url','env://') 
        print(f"| distributed init (rank {args['rank']}): {dist_url}", flush=True)
        dist.init_process_group(backend=args.get('dist_backend','nccl'), init_method=dist_url, world_size=args['world_size'], rank=args['rank'])
    else: 
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank=int(os.environ["RANK"]); args.world_size=int(os.environ["WORLD_SIZE"]); args.gpu=int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ:
            args.rank=int(os.environ["SLURM_PROCID"]); args.gpu=args.rank%torch.cuda.device_count()
        else: print("Not using distributed mode"); args.distributed=False; return
        args.distributed=True; torch.cuda.set_device(args.gpu)
        args.dist_backend=getattr(args,'dist_backend',"nccl"); dist_url=getattr(args,'dist_url','env://')
        print(f"| distributed init (rank {args.rank}): {dist_url}", flush=True)
        dist.init_process_group(backend=args.dist_backend, init_method=dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier(); setup_for_distributed(get_rank()==0)

