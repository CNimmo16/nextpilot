Knowledge distillation to produce a small model tailored for nextjs with app router

## Setup

1. Install uv with `./bin/install_uv.sh` (you may need to add to your path after doing this to allow running the `uv` command, see output of installation script for details)
2. Run uv sync to install dependencies
3. Activate the virtual environment with `source .venv/bin/activate`

## Preprocessing

1. Create a file called `.env`, and fill it with the GITHUB_TOKEN variable - `GITHUB_TOKEN=<your-token>` - replacing your token with a personal access token created through Github.
2. With the virtual env activated, run `python -m bin.preprocess` to run a script which downloads training data via the Github API to the `data/nextjs_repos` folder.

## Training

With the virtual env activated, run `python -m bin.train` to run the training script. See `bin/train.py` to tweak hyperparameters etc.

During training, model checkpoints will be saved to `data/checkpoints/epoch_<n>.pth`, where `n` is the epoch number. Upon completion, the final weights will be saved under a unique readable id within the `data/weights` folder, which will be logged to the console.

IMPORTANT: Upon starting training, the `data/checkpoints` folder will be wiped, so make sure to save any checkpoints you wish to keep from previous training runs before restarting training.

```
If you get an error referring to a missing "python.h installation, run `./bin/install_python_h.sh` to install the missing dependency.
```

## Inference

With the virtual env activated, run `python -m bin.infer`. You can now choose to test the base llama model, or load a model from one of the saved checkpoints. You can then enter some code to test completion with, or use the example code as defined in the `bin/infer.py` file.
