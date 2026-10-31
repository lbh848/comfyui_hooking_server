# SOYA Modal runtime image

This directory builds the public `linux/amd64` base image used by Modal. CUDA
kernels are compiled locally without a GPU for the supported GPU architectures:

- A100 40/80 GB: `sm_80`
- A10: `sm_86`
- L4 and L40S: `sm_89`
- RTX PRO 6000: `sm_120`

Build from this directory with Docker Desktop in Linux-container mode:

```powershell
docker buildx build `
  --platform linux/amd64 `
  --load `
  --build-arg VCS_REF=(git rev-parse HEAD) `
  --tag soya-comfy-runtime:cu128-torch2.11-sage2.2.0-r1 `
  docker/modal-runtime
```

The build runs `cuobjdump` and fails unless all four required cubin targets are
present. GPU execution is intentionally validated later with the Modal probe on
each supported GPU type.
