# Mario_Ops

*Russell's favourite, 心头好，so many first-times with this project.*

---

**One Mario game, played by multiple AI approaches. Everything in the cloud.**

`Mario_Ops` is a cloud-native playground for experimenting with different ways of making an AI play **Super Mario Bros. 1-1**.

The project started as a Double DQN reinforcement-learning experiment and has grown into a repository for comparing different approaches to a same problem:

- Train a Double DQN entirely in **Google Colab**.
- Train a Double DQN entirely in **GitHub Codespaces**.
- Train a Double DQN using **parallel simulation environments** (upto 64) on **AWS** infrastructure.
- Or, skip training entirely and use a general-purpose **OpenAI vision model** to play directly from screenshots.

This project takes a simple approach:

> Fork this repo, (give it a Star), and let AI play Mario in your preferred cloud.

- No local GPU, Python installation, or Docker setup is required.
- Click one button, and everything is ready in your browser.

---

## Quick Start

Choose one of the following options depending on how you want to run the project.

### 1. Run `Mario_Colab.ipynb` in Google Colab

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xqzr2000/Mario_Ops/blob/main/Mario_Colab/Mario_Colab.ipynb)
- Go with the flow (*just follow the steps in the notebook*).

### 2. Test Training in GitHub Codespaces

- Fork this repo → Click the green `Code` button → `Codespaces` tab → `Create codespace on main`.
- Paste the following into your Codespace terminal to run a short test training session:

```bash
# 20-episode test run
# takes about 5 minutes to complete

N_ENVS=2 \
NUM_EPISODES=20 \
BURNIN=5 \
LOG_EVERY_EPISODES=5 \
EVAL_EVERY_EPISODES=10 \
EVAL_EPISODES=5 \
python train.py
```

### 3. Train on AWS, using an EC2 instance with 64 vCPUs

- Start GitHub Codespaces normally, the AWS CLI environment is already built in.
- Follow the steps in this [README](https://github.com/xqzr2000/Mario_Ops/blob/main/Mario_AWS/README.md) (*I’m still working on making this README foolproof, but everything is ready to go.*)


### 4. Let OpenAI Play Mario

- Start GitHub Codespaces normally, `Mario_Ops.code-workspace` will take care of you.
- Follow the steps in this [README](https://github.com/xqzr2000/Mario_Ops/blob/main/Mario_OpenAI/README.md) (*I’m still working on making this README foolproof, but everything is ready to go.*)


*The `.devcontainer/devcontainer.json` file automatically builds and configures the entire development environment, including AWS CLI, based on `Dockerfile.develop`.* 

*Codespace opens in `Mario_AWS` by default, making it easy to run a quick training test. The `Mario_Ops.code-workspace` file provides a VS Code multi-root workspace, making it easy to switch between the `Mario_AWS` and `Mario_OpenAI` projects.*

---

## One Mario, Multiple Intelligences

```text
Mario_Ops/
│
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile.develop
│
├── Mario_AWS/
│   ├── cloud/
│   ├── mario_agent/
│   ├── tools/
│   ├── Dockerfile.deploy
│   ├── config.py
│   ├── play.py
│   ├── README.md
│   ├── requirements.txt
│   └── train.py
│
├── Mario_Colab/
│   ├── Mario_Colab.ipynb
│   └── mario_net_best.chkpt.zip
│
├── Mario_OpenAI/
│   ├── tools/
│   ├── .env.example
│   ├── config.py
│   ├── openai_agent.py
│   ├── openai_play.py
│   ├── README.md
│   ├── requirements.txt
│   └── video_utils.py
│
├── Mario_CoreWeave/ (developing a neocloud approach)
│
├── Mario_Ops.code-workspace
├── .gitignore
└── README.md
```

---
