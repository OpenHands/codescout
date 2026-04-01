<h1 align="center"> CodeScout: An Effective Recipe for Reinforcement Learning of Code Search Agents</h1>


This repository contains the source code for the paper **CodeScout: An Effective Recipe for Reinforcement Learning of Code Search Agents**

🏆 CodeScout achieves open-source SOTA code localization performance outperforming 8-18x larger base and post-trained LLMs and narrows the gap with frontier closed-source models.

<div align="center">
	<img src="./docs/verified_file_main.png" alt="CodeScout main figure (verified file-level)" width="49%" />
	<img src="./docs/verified_function_main.png" alt="CodeScout main figure (verified function-level)" width="49%" />
</div>

---

## ✨ Overview

A prerequisite for coding agents to perform tasks on large repositories is code localization - the identification of relevant files, classes, and functions to work on. While repository-level code localization has been performed using embedding-based retrieval approaches such as vector search, recent work has focused on developing agents to localize relevant code either as a standalone precursor to or interleaved with performing actual work. Most prior methods on agentic code search equip the agent with complex, specialized tools, such as repository graphs derived from static analysis. In this paper, we demonstrate that, with an effective reinforcement learning recipe, a coding agent equipped with *nothing more* than a standard Unix terminal can be trained to achieve strong results. Our experiments on three benchmarks (SWE-Bench Verified, Pro, and Lite) reveal that our models consistently achieve superior or competitive performance over **2-18×** larger base and post-trained LLMs and sometimes approach performance provided by closed models like Claude Sonnet, even when using specialized scaffolds. Our work particularly focuses on techniques for re-purposing existing coding agent environments for code search, reward design, and RL optimization. We release the resulting model family, CodeScout, along with all our code and data.

---
## 🧠 Methodology

Given a GitHub issue and a pre-PR repository, CodeScout navigates the repository via a terminal using Unix command-line utilities (for e.g. `rg`, `sed`, and `cat`.), and localizes relevant files, classes, and functions of code. The ground truth location set is determined by parsing the gold patch that fixes the issue.

We train 1.7B, 4B, and 14B models using RL wherein our reward function is the sum of file-level, module-level, and function-level F1 scores. 

![CodeScout system diagram](./docs/recipe.png)

## 🚀 Quick Start

### Environment setup

#### Pre-requisities:
1. uv: [Installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. `ripgrep`: [Installation instructions](https://github.com/burntsushi/ripgrep?tab=readme-ov-file#installation).
   - **Note**: We have used v15.1.0 in our experiments.
   - We have installed ripgrep using cargo:
   ```bash
      # Step 1: Install Rust (if not already installed on the machine)
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
      source $HOME/.cargo/env

      # Step 2: Install ripgrep via cargo
      cargo install ripgrep --version 15.1.0

      # Step 3: Verify if installation completed successfully - this command should execute without errors
      rg --version       
   ```

#### Installing Dependencies:

Training CodeScout models requires access to GPUs. We use 8xH100 GPUs for all our RL runs.
> **IMPORTANT NOTE**: You must ensure that the uv venv is not saved in the repository root since our training backend (SkyRL) uses Ray which copies all the files in this repository to a Ray cluster during RL.

```bash
export VIRTUAL_ENV=<location where uv virtual environment must be installed>
uv sync --all-extras --active
source $VIRTUAL_ENV/bin/activate
```

### Launch training (example)
The below command assumes access to a 8x GPU node and trains a 4B model using the same hyper-parameter settings as that of CodeScout-4B. Make sure to set your Wandb API Key as an environment variable.

```bash
export WANDB_API_KEY=<your_key>
bash scripts/run_async_training_4b.sh -m Qwen/Qwen3-4B-Instruct-2507 -n 8 -b 8 -c 1 -r Qwen3-4b-custom-finish-tool-gspo -w false -s <path to save checkpoints> -i 4 -t 4 -d ./data/swe_smith/ -o "+generator.reward=configs/reward_config_4b.yaml"

```

### 📊 Evaluation Setup

We will release the code for evaluation soon.