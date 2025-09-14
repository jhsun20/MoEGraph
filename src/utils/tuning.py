import os
import yaml
import optuna
from utils.train import train
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
import random


def objective(trial, config_base, phase):
    config = yaml.safe_load(yaml.dump(config_base))  # deep copy

    # --- Suggest hyperparameters by phase ---
    if phase == 1:
        config['dataset']['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128])
        config['training']['lr'] = trial.suggest_categorical("lr", [0.001, 0.0005, 0.0001])
        config['model']['rho_edge'] = trial.suggest_float("rho_edge", 0.2, 0.9)

    # --- Modify config for tuning ---
    config['logging']['wandb']['name'] = f"tune-trial-{trial.number}"
    config['experiment']['name'] = f"tune-phase{phase}"
    config['logging']['save_model'] = False  # no model saving during tuning
    # --- Run and capture output ---
    val_score = train(config, trial)
    return val_score


def run_optuna_tuning(config, phase, output_dir):
    print(f"Starting Optuna hyperparameter tuning - Phase {phase}")

    output_dir = os.path.join(output_dir, f"optuna_phase{phase}")
    os.makedirs(output_dir, exist_ok=True)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=25)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    if phase == 1:
        n_trials = config['experiment']['hyper_search']['n_trials_1']
    elif phase == 2:
        n_trials = config['experiment']['hyper_search']['n_trials_2']
    config['experiment']['seeds'] = [random.randint(0, 10000)]
    study.optimize(lambda trial: objective(trial, config, phase), n_trials=n_trials)

    # Save best config
    best_config = yaml.safe_load(yaml.dump(config))
    best_config = recursive_update(best_config, study.best_params)
    with open(os.path.join(output_dir, "best_config.yaml"), 'w') as f:
        yaml.dump(best_config, f)

    # Save visualizations
    try:
        plot_optimization_history(study).write_html(os.path.join(output_dir, "history.html"))
        plot_param_importances(study).write_html(os.path.join(output_dir, "importance.html"))
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("Tuning completed. Best params:")
    print(study.best_params)
    return best_config


def recursive_update(cfg, updates):
    for k, v in updates.items():
        if k in ["rho_edge"]:
            cfg.setdefault("model", {})[k] = v
        elif k in ["lr"]:
            cfg.setdefault("training", {})[k] = v
        elif k in ["batch_size"]:
            cfg.setdefault("dataset", {})[k] = v
        else:
            cfg[k] = v
    return cfg