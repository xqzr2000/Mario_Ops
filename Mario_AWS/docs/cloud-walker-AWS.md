# Cloud walker: AWS

*A dated learning journal about building the Mario reinforcement-learning deployment path on Amazon Web Services — and what that experience taught us before moving to serverless H100 training with Modal.*

**Companion journal:** [Cloud walker: Modal](https://github.com/xqzr2000/cmu-11-768-assignment-3/blob/main/docs/cloud-walker-Modal.md) · **Mario project:** [Mario_Ops](https://github.com/xqzr2000/Mario_Ops) · **Source notebook:** `understanding_mario61.ipynb`, especially Sections **8–10** and the **August 5, 2026 session record**.

> **Status as of November 10, 2026:** We completed the **AWS CPU builder → Docker smoke test → ECR push → EC2 termination** path. We **did not** complete AWS GPU training. S3/CloudWatch integrations exist in the Mario code, but the AWS deployment session did not enable or verify them. The ECR *pull* path on a new instance has not been tested either. This journal records completed steps as facts and marks everything else as planned.
>
> This is a **learning log**, not an instruction to start billable infrastructure. Commands that create or run resources are explicitly labeled. Prices and service-quota details mentioned from the notebook are historical observations, not current AWS quotes.

---

## 1. Where this fits in my cloud-learning journey

My earlier Mario experiment began as a notebook-trained **Double DQN** agent. In the `Mario_Colab` track, Google Colab supplied a T4 GPU. We eventually obtained a Mario flag-reaching checkpoint and a gameplay video. But a notebook and an interactive runtime were not the infrastructure I wanted to learn to manage.

For **Mario_Ops**, the next question was: *Can I package my own RL training code, launch a machine in AWS, operate it remotely, and move the same container between compute instances?* This taught me the fundamentals of **general-purpose cloud infrastructure**: identities, virtual machines, networks, container registries, remote shells, persistent artifacts, and cost cleanup.

Later, CMU 11-768 Assignment 3 asked a different question: *Can I post-train an existing open-weight LLM using rented H100s without administrating the machine?* That is the story in [Cloud walker: Modal](https://github.com/xqzr2000/cmu-11-768-assignment-3/blob/main/docs/cloud-walker-Modal.md).

**The main evolution:**

```text
Mario_Colab                Mario_AWS / AWS                 ScottyRL / Modal
------------               ----------------                 ----------------
Notebook + managed T4  ->  Codespaces control terminal ->  Codespaces RL orchestrator
                              |
                           |
                         AWS CLI + IAM                  Modal CLI + API token
                              |
                           |
                         EC2 machine I manage          Modal GPU function I declare
                              |
                           |
                         Docker + ECR                  modal.Image + Volumes
                              |
                           |
                         Mario Double DQN              Qwen3.5-4B + LoRA + GRPO

    Colab Mario training     AWS image/deployment path       First Modal RL probe
        completed                  validated                    completed
```

The last line matters: completing training in Colab does **not** mean we completed training on AWS.

---

## 2. The architecture: what we proved versus what we designed

### What we actually ran on AWS — August 5, 2026

```text
[GitHub Codespaces: control machine, no GPU]
       | AWS CLI, using scoped IAM user `marioops-cli`
       | create/inspect EC2 and ECR; start an SSM session
       v
[AWS EC2 c5.xlarge in us-west-2: CPU builder]
       | Amazon Linux 2023, 4 vCPU, 100 GB gp3 root disk
       | SSM Agent <- Session Manager <- Codespaces
       | install Git, Docker, tmux
       | clone Mario_Ops/Mario_AWS
       | build CUDA-capable Docker image
       | run 20 episodes with MARIOOPS_DEVICE=cpu
       | authenticate to ECR using EC2 IAM role
       v
[Amazon ECR: repository `marioops`, tag `cpu-test`]
       | image pushed: 3,941,067,456 bytes (~3.94 GB decimal)
       v
[Inspect image from Codespaces]
       |
       v
[Terminate builder; verify no orphaned EBS volumes]
```

**This was a successful *deployment-path* test**, not a cloud GPU training result. It proved the build and push halves of our container-delivery pipeline.

### The intended next architecture — **not yet run**

```text
[ECR: tested image]
       |
       | pull (not yet validated)
       v
[EC2 g4dn GPU trainer: planned]
       | CPU workers: parallel NES emulators / experiences
       | GPU: Double DQN inference + gradient updates
       | Docker --gpus all; MARIOOPS_DEVICE=cuda
       | tmux + time limit + checkpointing
       +------> [S3: checkpoints and gameplay video]     NOT YET TESTED
       +------> [CloudWatch: training metrics]           NOT YET TESTED
       |
       v
[Save/verify outputs; terminate instance]           NOT YET EXECUTED
```

The notebook also discussed larger `g6` instances with more vCPUs and an L4 GPU. Those were capacity-design considerations, **not** proof that a `g6` instance was launched.

---

## 3. Journal — what we learned, in order

### Before August 5 — Turning a notebook into cloud software

**What we changed**

Instead of copying the notebook straight onto a server, we refactored `Mario_Ops` into an importable and deployable Python project. The most important separation was:

| Component | Its job | What I learned |
|---|---|---|
| `.devcontainer/devcontainer.json` | Describes a reproducible Codespaces development environment | The development machine can be defined in version-controlled files |
| `Mario_AWS/Dockerfile.develop` | Python 3.10, CPU PyTorch, editor/developer tooling | Development does not need an expensive GPU image |
| `Mario_AWS/Dockerfile.deploy` | CUDA-capable production container, default command `python train.py` | The deployable artifact should be independent of the development container |
| `Mario_AWS/requirements.txt` | Shared Python pins, including `numpy==1.26.4` | Dependency compatibility is part of infrastructure correctness |
| `Mario_AWS/mario_agent/` | Double DQN model, agent, preprocessing, vector environments | Training logic belongs in reusable Python modules |
| `Mario_AWS/tools/smoke_test.py` | Verifies environment, vector stepping, agent, checkpointing, and optionally short training | Cheap tests should precede paid cloud tests |
| `Mario_AWS/cloud/` | Optional S3 and CloudWatch adapters | Cloud dependencies can be disabled during local/CPU verification |

The development container installed **Docker-in-Docker** and the **AWS CLI** via Dev Container features. The **Session Manager plugin** required an additional installation step: the AWS CLI alone was not enough for `aws ssm start-session`.

**Why this mattered:** Codespaces became the **administration workstation**. It was not our expensive training machine. The same codebase could be tested there, cloned onto EC2, and eventually run on a GPU host.

### August 5 — AWS identity became real

The notebook distinguished four identities:

1. **AWS root**: emergency/account ownership; not everyday work.
2. **`IAM-admin`**: privileged console setup, budgets, quotas, and role/policy configuration.
3. **`marioops-cli`**: constrained programmatic IAM user used by the **Codespaces AWS CLI**; its access-key pair was stored as **Codespaces secrets**, not committed to the repository.
4. **`MarioOpsInstanceRole`**: an **EC2 instance role**, supplying temporary credentials to the machine for ECR and other allowed AWS operations.

That split is fundamental: the human-facing control plane and the machine executing a job **should not share long-term credentials**. The AWS CLI's `aws sts get-caller-identity` told us which identity was active in each shell.

An earlier preparatory section of the notebook calls a proposed role `EC2-MarioApp-Role`; the **actual August 5 builder session** uses the instance-profile name **`MarioOpsInstanceRole`**. Follow the observed session when reconstructing what happened.

**Specific lesson:** A failure like `AccessDenied` is not automatically broken setup. First read the ARN in the error and ask *which identity* is trying the operation. The instance was allowed to push ECR images but not to create an ECR repository. That was least privilege working as designed.

### August 5 — Networking and remote access without SSH

We used the `us-west-2` (Oregon) region, a default VPC, a security group named `marioops-no-inbound`, and AWS Systems Manager **Session Manager**.

- **VPC**: the network boundary in which EC2 and its security group live.
- **Security group**: a stateful virtual firewall; ours intentionally had **no inbound rules**.
- **SSM Agent + Session Manager**: let Codespaces open a shell on EC2 over the AWS management path without exposing port 22 or distributing SSH private keys.
- **Instance profile**: attached the EC2 role, giving the instance short-lived AWS credentials through instance metadata; the launch required IMDSv2 (`HttpTokens=required`).

We learned the difference between two milestones: **EC2 is `running`** and **the SSM Agent is `Online`**. Reaching the first does not guarantee the second.

We also learned that `aws ssm start-session` changes the meaning of the shell: after connecting, commands execute **on EC2**, not in Codespaces. `sudo -i` changes identity again, to a root shell on that same EC2 host.

### August 5 — Launching the CPU builder and learning what costs money

The validated builder was `c5.xlarge` — **4 vCPUs, no GPU** — rather than the GPU instance discussed in the earlier plan. It ran **Amazon Linux 2023**, with a **100 GB gp3** root volume configured with `DeleteOnTermination=true`.

We used an EC2 **dry run** first. A successful authorization check produces the unintuitive `DryRunOperation` response; **it does not start an instance**. The actual `run-instances` operation starts billable EC2 compute.

Two problems made this lesson memorable:

- Passing the wrong **instance-profile name** (`MarioOpsInstanceProfile`) failed because the actual profile used the role name, `MarioOpsInstanceRole`.
- The account initially rejected the instance as **not eligible for Free Tier**. It needed a paid-plan upgrade even though dry-run authorization had succeeded.

The notebook records an estimated **~$0.17/hour** for this CPU instance at the time. That is a historical figure, not a quote for a future launch.

### August 5 — Managing a real remote machine

Inside EC2, we installed Git, Docker, and `tmux`, then started Docker. We cloned the public [Mario_Ops repository](https://github.com/xqzr2000/Mario_Ops) into `/opt` and entered `Mario_AWS/`.

`tmux` taught us a subtle but important cloud principle: **a remote process should not depend on an interactive browser tab or SSM connection**. The `build` session lived on EC2 even when the observation window disconnected.

The notebook also records practical mistakes: entering a second `tmux` session when already attached; confusing `duplicate session` with an error; and pasting many commands immediately after `sudo -i`, which caused some commands to be consumed by the wrong shell. The recovery routine was *reconnect → sudo → attach existing tmux session*.

### August 5 — Building the deployment Docker image

We ran the image build **on the EC2 CPU builder**, not on Codespaces:

```bash
# HISTORICAL COMMAND — executed inside EC2, from Mario_Ops/Mario_AWS
# Building again would consume EC2 time if an instance is running.
docker build -f Dockerfile.deploy -t marioops:cpu-test .
```

**Recorded observations:** nine build stages, **169.9 seconds** total, including a large CUDA base-image download/extraction. The Docker image was built successfully on a CPU-only host. The `numpy==1.26.4` pin avoided the `nes-py`/Gym incompatibility that had been important in development.

A CUDA-based container does **not** require a GPU just to be built. The GPU matters when actually executing CUDA computations. The tag `cpu-test` describes the **test**, not the image's ultimate hardware capability.

### August 5 — Running a deliberately small CPU training smoke test

We ran **20 episodes**, **two parallel NES environments**, and explicit CPU mode:

```bash
# HISTORICAL COMMAND — inside EC2. Safe only in the sense that it runs on
# the already-provisioned CPU machine; EC2 still bills while running.
docker run --rm \
  -e MARIOOPS_DEVICE=cpu \
  -e MARIOOPS_RUN_ID=ec2-cpu-test \
  -e AWS_REGION=us-west-2 \
  -e N_ENVS=2 \
  -e NUM_EPISODES=20 \
  -e BURNIN=5 \
  -e REPLAY_CAPACITY=20000 \
  -e LOG_EVERY_EPISODES=5 \
  -e EVAL_EVERY_EPISODES=10 \
  -e EVAL_EPISODES=5 \
  marioops:cpu-test
```

We **intentionally left `MARIOOPS_S3_BUCKET` unset**, so S3 and CloudWatch integration remained inactive (`s3=off cloudwatch=off` in the recorded diagnostics). This isolated Docker/training correctness from AWS-storage permissions.

The log demonstrated the intended DQN mechanics:

- Two emulator workers started **before** replay-buffer allocation (important for multiprocessing memory behavior).
- The replay buffer was preallocated at approximately **1.13 GB**.
- The experiment used a **gamma=0.99, seven-action** checkpoint lineage (`checkpoints_7_action_g99`).
- Best-checkpoint saving worked locally, and the experiment ran at roughly **40 steps/second** on four vCPUs.
- Short-run evaluation-score changes were not treated as evidence of model improvement or regression.

**What this proved:** the container could run the real Mario program and save a checkpoint on the CPU builder. **What it did not prove:** CUDA GPU operation, long-duration training, checkpoint transfer to S3, or CloudWatch publishing.

### August 5 — ECR: separating control-plane and machine permissions

We used **Amazon Elastic Container Registry (ECR)** as the persistent home for the Docker artifact. The division of work was intentional:

| Action | Executed from | AWS identity |
|---|---|---|
| Create the ECR repository `marioops` | Codespaces | `marioops-cli` |
| Log in to ECR, tag and push the image | EC2 | `MarioOpsInstanceRole` |
| Verify the uploaded tag and size | Codespaces | `marioops-cli` |

The EC2 role received `AccessDenied` for `ecr:CreateRepository`. That was expected: **pushing to a repository is not the same permission as creating one**. An attempt to create the repository with tags also uncovered the separate `ecr:TagResource` permission requirement. We removed the tags rather than granting extra rights just to finish the test.

**Result recorded:** ECR image tag **`cpu-test`**, reported image size **3,941,067,456 bytes** (approximately **3.94 GB**), with a recorded image digest. The push of **13 layers** succeeded. We do **not** claim that a new GPU machine has successfully pulled and executed it.

### August 5 — Termination was part of the experiment

When the image had been verified in ECR, we terminated the builder and checked the EC2 state transition from `shutting-down` to `terminated`. We also queried for detached, available EBS volumes, because a forgotten volume can continue billing after the VM is gone.

**Reported spend for this session:** approximately **$1**. This is the notebook's rough session estimate, not a bill audit.

**Learning:** A cloud experiment is not finished when the program finishes. It is finished when its **artifacts are preserved and its billable resources are accounted for**.

---

## 4. Understanding the AWS control path — an annotated reference

The following commands document the recorded workflow. **Read-only commands can be used to inspect an existing account; launch/build/push commands should not be rerun just for learning.** Commands are separated by the machine where they belong.

### A. Codespaces — check identity and installed tools (read-only)

```bash
aws --version
session-manager-plugin --version
aws sts get-caller-identity         # expect an IAM user such as marioops-cli
aws ec2 describe-instances \
  --filters Name=tag:Project,Values=marioops \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType]' \
  --output table
```

The Codespace used GitHub-encrypted secrets for `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION`; they should never appear in the repository or in Docker image layers. The Dev Container installed AWS CLI and Docker-in-Docker, while the Session Manager plugin was added separately.

### B. Codespaces — resolve the region's resources (read-only)

```bash
export AWS_REGION=us-west-2
export VPC_ID=$(aws ec2 describe-vpcs \
  --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
export AMI_ID=$(aws ssm get-parameters \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64 \
  --query 'Parameters[0].Value' --output text)
export SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values=marioops-no-inbound \
  --query 'SecurityGroups[0].GroupId' --output text)
printf 'VPC=%s\nAMI=%s\nSG=%s\n' "$VPC_ID" "$AMI_ID" "$SG_ID"
```

This reveals why hardcoding AWS resource IDs is fragile: AMI and security-group IDs are specific to a region and account. **Creating** the security group was a one-time operation in the original session; afterward, we **looked it up by name** instead of trying to create it twice. We checked that its inbound-permission list was empty.

### C. Codespaces — dry-run versus actual EC2 launch

```bash
# READ-ONLY AUTHORIZATION TEST — does NOT create a VM:
aws ec2 run-instances --dry-run \
  --image-id "$AMI_ID" \
  --instance-type c5.xlarge \
  --iam-instance-profile Name=MarioOpsInstanceRole \
  --security-group-ids "$SG_ID"
```

A successful dry run responds with `DryRunOperation`; the launch itself would be a **separate, billable** `aws ec2 run-instances` call. The recorded full launch also set a 100 GB gp3 volume with `DeleteOnTermination`, `HttpTokens=required`, a project tag, and `--instance-initiated-shutdown-behavior terminate`.

**Do not launch anything from this learning log unless you intentionally want to rent EC2 compute.** The full original command is preserved in notebook Section 10.3.4.

### D. Codespaces → SSM → EC2 — recognize which shell you are in

```bash
# Codespaces; requires an actual running, SSM-Online EC2 instance:
aws ec2 wait instance-running --instance-ids "$IID"
aws ssm describe-instance-information \
  --filters "Key=InstanceIds,Values=$IID" \
  --query 'InstanceInformationList[0].PingStatus' --output text
aws ssm start-session --target "$IID"
```

Then on **EC2**, not Codespaces:

```bash
sudo -i
# EC2 root shell:
dnf install -y git docker tmux
systemctl enable --now docker
tmux new -s build
cd /opt
git clone https://github.com/xqzr2000/Mario_Ops.git
cd Mario_Ops/Mario_AWS
df -h /
```

If reconnecting, use `tmux attach -t build`; do not blindly start a new session. `Ctrl-b`, then `d`, detaches the terminal while leaving processes running on the host.

### E. Codespaces creates ECR; EC2 pushes the image

```bash
# HISTORICAL ACTION — Codespaces, creates a persistent AWS resource:
aws ecr create-repository --repository-name marioops \
  --region us-west-2 --image-scanning-configuration scanOnPush=true
```

```bash
# HISTORICAL ACTION — EC2, after a successful local Docker build:
export AWS_REGION=us-west-2
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$REGISTRY"
docker tag marioops:cpu-test "$REGISTRY/marioops:cpu-test"
docker push "$REGISTRY/marioops:cpu-test"
```

```bash
# Codespaces; read-only verification:
aws ecr describe-images --repository-name marioops --region us-west-2 \
  --query 'imageDetails[].[imageTags[0],imageSizeInBytes,imagePushedAt]' \
  --output table
```

### F. Codespaces — clean up EC2, then verify (billable action followed by read-only checks)

```bash
# DESTRUCTIVE ACTION — terminates the named instance, so confirm the ID first:
aws ec2 terminate-instances --instance-ids "$IID"

# READ-ONLY checks:
aws ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text
aws ec2 describe-volumes --filters Name=status,Values=available \
  --query 'Volumes[].[VolumeId,Size,CreateTime]' --output table
```

An empty list of *available* volumes was the expected cleanup outcome, but also inspect other associated resources and your billing dashboard. Terminating EC2 does **not** automatically delete ECR images, S3 objects, or all possible EBS volumes.

---

## 5. What was inside the Mario container?

We were deploying **our own Double DQN**, not hosting an LLM. The Mario training code had these roles:

```text
N_ENVS CPU processes (NES game emulation)
      |
      | observations / rewards / terminal transitions
      v
Parent process
      | batched MarioNet action selection
      | replay buffer (transitions, not LLM tokens)
      | sample mini-batches for Double DQN learning
      v
PyTorch model learner (CPU during smoke test; GPU planned)
      |
      +--> online and target-network training logic
      +--> local periodic/best/final checkpoints
      +--> optional S3 uploads + CloudWatch metrics
```

**Why the project emphasizes CPUs:** NES game simulation runs on CPU processes; a small DQN can perform its gradient work relatively quickly on a GPU. More GPU power cannot automatically accelerate the CPU emulator. Increasing `N_ENVS` needs enough **vCPU**, RAM, and effective environment throughput. This is why the notebook compared larger `g4dn`/`g6` instances on their CPU counts rather than assuming more expensive GPUs always win.

`Mario_AWS/train.py` was designed for vectorized environments and overlap between actor stepping and learning. It also addressed a few hard-earned RL engineering problems:

- **Gym/NumPy interoperability**, including a pinned `numpy==1.26.4` and an RNG compatibility patch.
- **Memory discipline**: fork emulator workers before allocating the large replay buffer.
- **Correct terminal observations**: retain final frames through vector-environment resets.
- **Lineage isolation**: prevent incompatible action spaces or discount factors (`gamma`) from sharing checkpoints.
- **Resume/termination safety**: checkpoint before stopping and preserve outputs via optional cloud adapters.

These are capabilities documented in the notebook and code design. The CPU smoke test exercised a subset; the long GPU/CloudWatch/S3 behavior was **not** thereby verified.

---

## 6. AWS storage and monitoring: designed, but not yet demonstrated in this deployment

| Resource | Intended role | Evidence from August 5 |
|---|---|---|
| **ECR** | Store versioned Docker images | **Confirmed:** repository and `cpu-test` image pushed |
| **EBS** | Root disk for the builder VM, Docker layers, and temporary local checkpoints | **Confirmed:** 100 GB gp3 launch configuration; deletion/cleanup checked |
| **S3** | Durable checkpoints, gameplay recordings, and restore between machines | **Not exercised:** `MARIOOPS_S3_BUCKET` intentionally unset |
| **CloudWatch** | Optional custom metrics such as rewards, loss, and flag rate | **Not exercised:** monitoring disabled for the CPU run |
| **SSM** | Manage EC2 shells without inbound SSH | **Confirmed:** session workflow used |
| **ECR pull on GPU EC2** | Reuse image rather than rebuild | **Not exercised:** next planned step |

Important distinctions:

- **ECR stores a container image**, not a training checkpoint or live instance.
- **EBS is block storage** attached to the VM; our root volume was configured for deletion on termination.
- **S3 is object storage** appropriate for durable output and resuming from a different machine; an S3 integration *existing in source code* is not proof that a cloud upload succeeded.
- **CloudWatch custom metrics** were coded as an optional integration, but did not provide demonstrated cloud dashboards or GPU telemetry in this smoke test. Do not equate code-level metric publication with automatic collection of every GPU/CPU log; additional configuration may be required.

The next safe validation proposed in the notebook was to launch a small **CPU** instance again, *pull* the image from ECR, then validate S3 permissions and CloudWatch independently before committing to a more expensive GPU run. `MARIOOPS_RUN_ID` must remain consistent across checkpoint-writing and restore runs.

---

## 7. Deep comparison: AWS for Mario versus Modal for ScottyRL

This table distinguishes **our executed AWS experience** from the **documented Modal deployment and probe**. “Planned” is used explicitly where the AWS notebook has no execution evidence.

| Question | Mario_Ops on AWS | ScottyRL on Modal |
|---|---|---|
| **What are we trying to train?** | Our own small **Double DQN** to play Mario | **Qwen3.5-4B**, post-trained with RL using a rank-32 **LoRA adapter** |
| **What was actually validated?** | CPU EC2 Docker build, 20-episode smoke test, ECR push, EC2 termination | H100 image build and a **10-step synchronous GRPO probe** on 2 H100s, as recorded in [Modal journal](https://github.com/xqzr2000/cmu-11-768-assignment-3/blob/main/docs/cloud-walker-Modal.md) |
| **Cloud abstraction** | Rent and administer an **EC2 VM** | Deploy a **Python-defined GPU function/web server**; no VM shell to maintain |
| **Control environment** | GitHub Codespaces with **AWS CLI** | GitHub Codespaces with **Modal CLI**, invoked via `uv run modal` |
| **Authentication** | IAM user access keys as Codespaces secrets; EC2 receives an **IAM role** | Modal token ID and secret as Codespaces secrets; simpler but different permissions model |
| **Network & access** | VPC + security group with no inbound; **SSM** for remote shell | HTTPS Tinker endpoint printed by `modal deploy`; keep its unauthenticated URL private |
| **Compute provisioning** | `aws ec2 run-instances` using AMI, instance type, instance profile, security group, EBS | `gpu="H100:2"` or `gpu="H100:4"`, memory/CPU and scale-to-zero declared in `@app.function` |
| **GPU count/placement** | **0 GPUs in completed AWS session**. GPU trainer was only planned (`g4dn`/`g6`) | **2 H100s** for Number Search; **4 H100s** for TextCraft; split vs colocated by run |
| **Where the training loop runs** | Intended to run **inside Docker on EC2**; CPU smoke run did so | Agent/environment/GRPO trainer orchestration in **Codespaces**; sampling and parameter updates on Modal |
| **How code arrives** | Clone repository on EC2, build Docker image, push to ECR; pull later on new VM (planned) | `modal.Image` builds a CUDA/SkyRL environment on Modal; reusable cached layers; no separate ECR push by us |
| **Where base-model weights come from** | MarioNet is our network initialized/trained by project code; we are not downloading a foundation model | Modal downloads **Qwen/Qwen3.5-4B** open weights from Hugging Face on first service start; caches on Modal Volume |
| **How inference runs** | MarioNet selects discrete controller actions for parallel emulators; CPU during smoke test | **vLLM** generates tool-call tokens/log-probabilities; GPUs also support **Megatron** training |
| **Learning algorithm** | **Double DQN**, replay transitions, online and target networks | **REINFORCE/GRPO**, group rewards/advantages, token masks, PPO-style loss; train LoRA parameters |
| **Training interface** | Direct Python/PyTorch code in the EC2 Docker process | Remote **Tinker-compatible API**, e.g. `sample`, `forward_backward`, `optim_step` via SkyRL |
| **Persistent image artifact** | **ECR** `marioops:cpu-test` (confirmed) | Modal-built and cached image (confirmed) |
| **Training artifacts** | Local checkpoints tested; **S3 upload planned** | **Modal Volumes** for model cache and saved state; first probe checkpoint recorded |
| **Metrics** | Local stdout confirmed; **CloudWatch not enabled during smoke test** | Local `runs/*.jsonl` plus Modal dashboard; optional W&B |
| **Long-running process** | `tmux` on the EC2 machine, plus planned time limit/SIGTERM recovery | Trainer in Codespaces (`tmux` recommended); separate Modal service whose container scales to zero |
| **Recovery** | Planned checkpoint-to-S3 and a new EC2 instance to restore | Resumable state on `<app>-state` Volume; `--set train.resume=true`; tested probe saved checkpoint, full crash recovery not proven |
| **When billing begins** | Actual EC2 **launch** (and storage persists as applicable) | GPU container activation after a **request**; deployment registration alone does not allocate GPUs |
| **How to stop spending** | Terminate EC2 and inspect orphaned EBS, ECR/S3 storage | `uv run modal app stop <app>` and check dashboard; idle window can still bill before scale-down |
| **Best lesson** | Understand and control the **machine, identity, network, and artifact** | Define the **workload/service**, while the platform handles machine lifecycle |

### The biggest conceptual difference

On AWS Mario, we **moved the executable code to the machine**:

```text
Codespaces --AWS CLI/SSM--> EC2 --Docker--> train.py + MarioNet
```

On Modal ScottyRL, the **orchestration remains in Codespaces**, but the GPU-bound model operations live behind an API:

```text
Codespaces: agent + env + reward + GRPO trainer
              |
              | Tinker requests over HTTPS
              v
Modal: SkyRL server + vLLM sampling + Megatron LoRA updates
```

This distinction is about **our chosen architectures**, not a claim that AWS cannot run APIs or that Modal cannot run code remotely. Both clouds can support other designs.

### An illuminating similarity

Both projects use **reinforcement learning**, but the experience-generating environment is different:

- Mario: CPU emulators produce **state–action–reward transitions** for a DQN replay buffer.
- ScottyRL: vLLM produces **text/tool-call trajectories with token log-probabilities**, while Python environments compute rewards; GRPO transforms trajectories into masked token-training examples.

The expensive bottleneck moves too: Mario's workload can be **CPU-emulation-bound**, while ScottyRL's large-model inference and backward passes are emphatically **GPU-heavy**. That is why comparing instances by *GPU name alone* is a poor way to plan cloud ML spending.

---

## 8. Cost and risk: lessons to carry into both clouds

### What the journals actually record

| Event | Historical record | Interpretation |
|---|---|---|
| AWS builder session, Aug 5 | Approximately **$1** | Rough estimate for the CPU build/smoke/push session, not a final AWS statement |
| AWS GPU Mario training | **Not run** | No valid AWS GPU-training cost or throughput result |
| Modal first image build + 10-step GRPO probe, Nov 9–10 | Approximately **$5** | Estimate from the [Modal journal](https://github.com/xqzr2000/cmu-11-768-assignment-3/blob/main/docs/cloud-walker-Modal.md), not an invoice |
| Modal Number Search deployment | Around **$8.50/hour** for 2 H100s in the journal | Historic planning estimate; check current Modal rates |
| Modal TextCraft deployment | Around **$16.40/hour** for 4 H100s in the journal | Historic planning estimate, not evidence the 4-H100 job ran |

An instance terminated after an experiment can still leave **billable persistent storage**. A Modal app stopped after training can leave **Volumes**; a successful test does not itself prove every artifact was cleaned up. Record both usage and persistent resources.

### Spending discipline adopted from AWS, then reused on Modal

1. Run unit tests and inexpensive **CPU smoke tests** before renting GPUs.
2. Separate **image-build validation** from the **full training experiment**.
3. Confirm **credentials, quota, region, and resource IDs** before launch.
4. Name runs and artifacts predictably so a resume cannot silently pick up the wrong checkpoint.
5. Verify an initial small end-to-end run (including metrics and checkpoints), not merely successful startup.
6. Check the cloud provider's billing dashboard and configure budget/spend alerts or limits where available.
7. Stop/terminate compute immediately afterward, then inspect persistent resources.
8. Distinguish **planned architecture**, **working source code**, **successful smoke test**, and **validated long training** in the learning log.

**Security reminder:** The August session also tightened `.dockerignore` for `.env`, key files, and `.aws/`; and `.gitignore` for checkpoints/media. `gitignore` does not remove files already tracked, and deleting a secret from a later Docker layer does **not** erase an earlier layer that contained it. Secret hygiene must come *before* `docker build` and `docker push`.

---

## 9. Outstanding AWS milestones — explicit TODOs, not achievements

- [ ] **Validate ECR pull** on a fresh CPU EC2 instance (image should be reusable without building).
- [ ] **Validate S3** bucket access with private/public settings reviewed, then confirm checkpoint save and recovery using a consistent `MARIOOPS_RUN_ID`.
- [ ] **Validate CloudWatch** permissions and observed metrics/logs; do not assume optional code implies working dashboards.
- [ ] **Obtain appropriate G/VT GPU quota** in the chosen region. The notebook records an earlier **32-vCPU request denied**, with a future suggestion to ask for **16 vCPUs** tied to one `g4dn.4xlarge`. Quotas are dynamic; inspect the current account first.
- [ ] **Check GPU host readiness** (driver, NVIDIA container runtime, disk, and Docker `--gpus all`). This was not verified by the CPU build.
- [ ] **Run a small GPU probe** using `MARIOOPS_DEVICE=cuda` and a bounded training duration, inspect CUDA utilization, checkpoints, score, and cost.
- [ ] **Run longer AWS GPU Mario training** only if the probe demonstrates an acceptable learning signal and throughput.
- [ ] **Prove restart/cleanup automation**, including behavior on interruption and instance-initiated shutdown configured to terminate.
- [ ] **Evaluate OIDC for GitHub Actions** as a future credential strategy; the August session used IAM access keys in Codespaces, **not** OIDC.

The notebook's proposed next GPU configuration used `g4dn.4xlarge` (16 vCPUs and one T4), with a plan for multiple NES workers and a time-bounded container. It also discussed alternatives such as `g6`; none of those plans is an observed AWS GPU run. The notebook separately flags possible DQN quality issues (such as Q-value overestimation) that deserve investigation before paying for a long run.

---

## 10. How I would explain the lesson now

AWS taught me what a cloud deployment is made of:

- **Identity**: a human CLI caller is not the same as the machine's IAM role.
- **Compute**: an EC2 VM is mine to launch, configure, observe, and terminate.
- **Networking**: VPC and security groups define access; SSM can eliminate inbound SSH.
- **Containers**: an image packages code and dependencies, but ECR only stores it; a VM still has to execute it.
- **Storage**: temporary EBS, durable S3, and container ECR solve different problems.
- **Operations**: logs, checkpoints, tmux, service quotas, budgets, and cleanup are part of the application.

Modal then showed me which of these concerns a higher-level platform can absorb. I still need to understand model hosting, sampling, LoRA updates, state, and cost — but I no longer need to choose an AMI, SSH/SSM into a GPU node, prepare a Docker daemon, or manually push an ECR image to conduct the provided ScottyRL experiment.

**My takeaway:** AWS taught me **what is underneath the cloud abstraction**. Modal taught me **what productive ML development can look like when that infrastructure is abstracted away**. They are not mutually exclusive choices; understanding both helps me decide when I need low-level control and when I should prioritize the experiment itself.

---

## 11. Continue the two-journal series

- **This chapter — AWS / Mario:** the foundation: build a deployable container, administer EC2, and use ECR with least-privilege IAM. **Verified outcome:** successful CPU deployment-path proof, not GPU training.
- **[Next chapter — Modal / ScottyRL](https://github.com/xqzr2000/cmu-11-768-assignment-3/blob/main/docs/cloud-walker-Modal.md):** apply that foundation to serverless H100s and post-training an open-weight LLM. **Verified outcome so far:** image build and first 10-step RL probe.

**Suggested backlink to add near the top of `docs/cloud-walker-Modal.md` once this file is committed to the same `docs/` folder:**

```markdown
**Previous chapter:** [Cloud walker: AWS](cloud-walker-AWS.md) — building and testing
Mario_Ops on EC2, packaging Docker images in ECR, and learning IAM/SSM; the AWS GPU
training step remained planned rather than completed.
```

If publishing this file under `cmu-11-768-assignment-3/docs/`, the sibling relative link `[Cloud walker: Modal](cloud-walker-Modal.md)` will also work. The full GitHub URL above works as a standalone downloaded file before that commit.

### Template for the next AWS entry

```markdown
### YYYY-MM-DD — <experiment or infrastructure change>

**Goal:**
**What I actually executed:**
**What succeeded (commands, artifacts, metrics):**
**What failed and how I diagnosed it:**
**AWS identities/resources used:**
**Cost and cleanup verified:**
**What remains untested:**
**What this taught me about Modal (or vice versa):**
```

### Evidence and scope

- **Primary historical source:** uploaded `understanding_mario61.ipynb`, Sections **8** (repository/dev containers/smoke testing), **9** (Mario_AWS code and cloud adapters), **10** (AWS deployment runbook), and the **2026-08-05 session record**.
- **Comparison source:** [`docs/cloud-walker-Modal.md`](https://github.com/xqzr2000/cmu-11-768-assignment-3/blob/main/docs/cloud-walker-Modal.md), retrieved from the `cmu-11-768-assignment-3` repository; the journal's November 2026 entries are treated as its recorded observations.
- **Method:** A command appearing in the runbook is **not** taken as evidence that it was executed. Completed AWS outcomes are anchored to the August 5 record and to the user's explicit confirmation that GPU training was not completed. Estimates, future plans, and general architectural explanations are labeled separately.
