# Hyperparameter Tuning Guide for ParaJEPA

This guide explains how to use Weights & Biases (W&B) for hyperparameter tuning.

## Setup

1. **Install W&B** (if not already installed):
   ```bash
   pip install wandb
   ```

2. **Login to W&B**:
   ```bash
   wandb login
   ```
   Follow the prompts to authenticate.

## Usage Modes

### 1. Single Run with W&B Logging

Run a single training run with W&B logging:

```bash
python hyperparameter_tuning.py
```

This will:
- Use default hyperparameters from `config.py`
- Log all metrics to W&B
- Create a run in the "para-jepa" project

### 2. W&B Sweep (Automated Hyperparameter Search)

#### Step 1: Initialize the Sweep

```bash
wandb sweep sweep_config.yaml
```

This will output a sweep ID like: `your-username/para-jepa/abc123xyz`

#### Step 2: Run the Sweep Agent

On each machine/GPU you want to use:

```bash
wandb agent <sweep_id>
```

For example:
```bash
wandb agent your-username/para-jepa/abc123xyz
```

You can run multiple agents in parallel to speed up the search.

#### Step 3: Monitor Progress

View your sweep in the W&B dashboard:
- Go to https://wandb.ai
- Navigate to your project "para-jepa"
- Click on the sweep to see all runs and results

## Sweep Configuration

The `sweep_config.yaml` file defines:
- **Search method**: Bayesian optimization (recommended)
- **Metric to optimize**: `best_val_loss` (minimize)
- **Hyperparameter ranges**: See the file for all tunable parameters

### Tunable Hyperparameters

- `ema_decay`: 0.99 - 0.999 (EMA update rate)
- `pred_depth`: 2 - 5 (Predictor network depth)
- `batch_size`: [8, 16, 32] (Memory dependent)
- `learning_rate`: 1e-6 - 1e-4 (Log-uniform distribution)
- `weight_decay`: 0.0 - 0.1 (L2 regularization)
- `max_length`: [64, 128, 256] (Sequence length)

### Modifying the Sweep Configuration

Edit `sweep_config.yaml` to:
- Change search method (grid, random, bayes)
- Adjust hyperparameter ranges
- Add/remove hyperparameters
- Change the metric to optimize

## Best Practices

1. **Start Small**: Run a few test runs first to ensure everything works
2. **Monitor Resources**: Watch GPU memory usage, adjust batch_size if needed
3. **Early Stopping**: Consider adding early stopping for long sweeps
4. **Save Checkpoints**: Best models are automatically saved as `para_jepa_best_model.pt`
5. **Compare Runs**: Use W&B dashboard to compare different hyperparameter combinations

## Viewing Results

After training, you can:
1. View metrics in W&B dashboard
2. Compare runs side-by-side
3. Download best model checkpoints
4. Export results as CSV/JSON

## Troubleshooting

- **"wandb not found"**: Install with `pip install wandb`
- **Authentication errors**: Run `wandb login` again
- **Out of memory**: Reduce `batch_size` or `max_length` in sweep config
- **Sweep not starting**: Check that `hyperparameter_tuning.py` is in the same directory

## Example Workflow

```bash
# 1. Initialize sweep
wandb sweep sweep_config.yaml
# Output: your-username/para-jepa/abc123xyz

# 2. Run agent (can run multiple in parallel)
wandb agent your-username/para-jepa/abc123xyz

# 3. Monitor in browser
# Go to https://wandb.ai/your-username/para-jepa

# 4. After sweep completes, analyze results
# Best hyperparameters will be highlighted in W&B dashboard
```

## Next Steps

After finding good hyperparameters:
1. Update `config.py` with best values
2. Train final model with best hyperparameters
3. Run evaluation suite
4. Document results

