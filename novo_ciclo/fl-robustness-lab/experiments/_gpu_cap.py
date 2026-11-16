"""Lançador com teto de memória de GPU (fila v9, GPUs emprestadas 0/1/2/6/7).
Uso: python -u experiments/_gpu_cap.py experiments/<script>.py [args...]
Teto em GB pela variável FL_GPU_MEM_GB (padrão 4). Acima dele o PyTorch dá OOM no NOSSO processo,
deixando o resto da GPU livre para outros usuários."""
import os, runpy, sys
import torch
gb = float(os.environ.get("FL_GPU_MEM_GB", "4"))
if torch.cuda.is_available():
    tot = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / tot), 0)
    print(f"[_gpu_cap] teto de memória da GPU: {gb:.1f} GB de {tot:.1f} GB", flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
