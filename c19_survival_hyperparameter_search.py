# -------------------------------------------------------------------
# Import required packages and setup environment
# -------------------------------------------------------------------
from optuna import Trial
from xgboost import XGBRegressor

# -------------------------------------------------------------------
# XGBoost trials
# -------------------------------------------------------------------
def instantiate_xgboost(trial: Trial) -> XGBRegressor:
    """
    Create an XGBRegressor model with hyperparameters suggested by an Optuna trial.

    This function defines the search space for several XGBoost hyperparameters and
    uses an Optuna `Trial` object to sample values during optimization. It is
    intended for use inside an Optuna objective function when tuning
    Cox-based survival models (`survival:cox`).

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial instance used to suggest hyperparameter values.

    Returns
    -------
    XGBRegressor
        An instance of `XGBRegressor` configured with hyperparameters sampled
        from the defined search space.

    Hyperparameters Suggested
    -------------------------
    max_depth : int
        Maximum depth of each tree. Sampled from [2, 16].

    learning_rate : float
        Step size shrinkage used in updates.
        Sampled on a log scale between 0.0001 and 0.02.

    subsample : float
        Fraction of samples to use for each tree, sampled from [0.5, 1.0].

    min_split_loss : float
        Minimum loss reduction required for a node split, sampled from [0, 0.5].

    min_child_weight : int
        Minimum sum of instance weights needed in a child node, sampled from [1, 40].

    reg_lambda : float
        L2 regularization term, sampled from [0, 10].

    reg_alpha : float
        L1 regularization term, sampled from [0, 10].

    Notes
    -----
    - The model is configured for survival analysis using:
      * `objective='survival:cox'`
      * `eval_metric='cox-nloglik'`
    - The `'hist'` tree method is used for efficient training on larger datasets.
    """

    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.02, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'min_split_loss': trial.suggest_float('min_split_loss', 0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 40),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),  # L2
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  # L1        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'tree_method': 'hist'
    }
    return XGBRegressor(**params)