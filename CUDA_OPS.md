# CUDA operators

Only **MMMamba** needs custom CUDA kernels: upstream `mamba_ssm` 1.2.0.post1 with `causal_conv1d` 1.2.0.post2, built from
source against the environment of `environment.yml` (Python 3.9, torch 2.7.1+cu126, CUDA 12.6 toolkit):

```bash
bash scripts/build_mamba_ssm.sh
```

**TS-Pan** compiles RWKV's `wkv` CUDA kernel on first use (`torch.utils.cpp_extension.load`, patched to an absolute
source path). This needs `ninja` (in `environment.yml`) and a CUDA toolkit with `nvcc` (found through `CUDA_HOME` or
`/usr/local/cuda`). The kernel is built once per machine into the PyTorch extensions cache (`$TORCH_EXTENSIONS_DIR`, default
`~/.cache/torch_extensions`). The released TS-Pan numbers were produced with `nvcc` 11.8 under torch 2.7.1+cu126.

All other backbones are plain PyTorch.
