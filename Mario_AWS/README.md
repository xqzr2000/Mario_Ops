# Mario_AWS

## Repo layout

```text
mario_aws/
├─ train.py                  # Parallel cloud-aware training loop (async vec envs,
│                            #   resume, lineage guards, S3 sync, CW metrics)
├─ play.py                   # Records a gameplay clip, uploads it, prints a URL
├─ config.py                 # All settings, overridable via env vars; lineage-aware paths
├─ requirements.txt          # Pinned deps, shared by BOTH images
│                            #   (the numpy 1.26.4 pin is load-bearing)
├─ mario_agent/              # Double DQN package (device-aware: CPU or CUDA)
│   ├─ __init__.py
│   ├─ config.py             # IMAGE_SIZE / STACK_SIZE / FRAME_SKIP (single source of truth)
│   ├─ dqn_model.py          # MarioNet: online + frozen target networks
│   ├─ mario_agent.py        # act(_batch) / cache(_batch) / recall / learn(_once) / save / load
│   ├─ data_pipeline.py      # build_env: skip -> gray -> resize -> stack [-> no-op starts]
│   └─ vector_env.py         # gym RNG patch, info/terminal-frame parsing, vec env factory
├─ cloud/
│   ├─ storage.py            # S3 persistence + presigned/public video URLs
│   └─ monitoring.py         # CloudWatch custom metrics (incl. flag rate + eval score)
├─ tools/
│   └─ smoke_test.py         # Environment checks; run automatically on Codespace create
├─ Dockerfile.develop        # Dev image: Python 3.10, CPU torch, editor tooling
├─ Dockerfile.deploy         # Production image: CUDA 12.1, runs on g4dn/g5
└─ .dockerignore
```
