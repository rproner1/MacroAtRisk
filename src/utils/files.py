from datetime import UTC, datetime
from pathlib import Path
import os
import portalocker
import json


def get_latest_file(
    file: Path | None = None,
    prefix: str | None = None,
    extension: str = ".parquet",
    directory: Path = Path("."),
) -> Path | None:
    """
    Returns the latest file matching the prefix and extension in the directory.

    Args:
        file (Path, optional): The file path without the timestamp. Defaults to None.
        prefix (str): The prefix of the file name. Defaults to None.
        extension (str, optional): The extension of the file. Defaults to ".parquet".
        directory (Path, optional): The directory to search in. Defaults to Path(".")

    Returns:
        path: The path to the latest file.
    """
    if file is None and prefix is None:
        raise ValueError("Either file or prefix must be provided.")
    if file:
        prefix = file.stem
        extension = file.suffix
        directory = file.parent

    if files := list(directory.glob(f"{prefix}_2*{extension}")):
        return max(files, key=lambda x: x.stem)
    else:
        return None


def timestamp_file(file: Path) -> Path:
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return file.with_name(f"{file.stem}_{ts}{file.suffix}")

def check_hps_exist(model_name: str, log_path: str) -> bool:
    """
    Checks if hyperparameters for a model already exist in the log file.

    Parameters
    ----------
    model_name: str
        Name of the model, e.g., "model_q5".
    log_path: str
        Path to the log file, e.g., "tuning_log.txt".

    Returns
    -------
    bool:
        True if hyperparameters exist, False otherwise.
    """
    
    if not os.path.exists(log_path):
        return False
    
    try:
        with open(log_path, 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            log = json.load(f)
            portalocker.unlock(f)
        return model_name in log
    
    except Exception:
        return False


def save_hyperparameters(hps: dict, model_name: str, log_path: str, overwrite: bool=False):
    """
    Saves hyperparameters to a log file.

    Parameters
    ----------
    hps: dict
        Hyperparameters to save.
    model_name: str
        Name of the model, e.g., "model_q5".
    log_path: str
        Path to the log file, e.g., "tuning_log.txt".
    """
    # Check if log exists, create log file if not
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump({}, f)
            portalocker.unlock(f)

    # Load log and update
    with open(log_path, 'r+') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        log = json.load(f)

        # If model not in log, always save
        if model_name not in log:
            print('Saving model hyperparameters (first entry)...')
            log[model_name] = hps
            f.seek(0)
            json.dump(log, f)
            f.truncate()
        else:
            if overwrite:
                old_value = log[model_name].get('value', 0.0)
                if hps.get('value', 0.0) > old_value:
                    print('Saving model hyperparameters (better value)...')
                    log[model_name] = hps
                    f.seek(0)
                    json.dump(log, f)
                    f.truncate()
        portalocker.unlock(f)
    return None


def load_hyperparameters(model_name: str, log_path: str) -> dict:
    """
    Loads hyperparameters from a log file.

    Parameters
    ----------
    model_name: str
        Name of the model, e.g., "model_q5".
    log_path: str
        Path to the log file, e.g., "tuning_log.txt".

    Returns
    -------
    dict:
        Hyperparameters for the specified model.
    """
    import portalocker, json

    with open(log_path, 'r') as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        log = json.load(f)
        portalocker.unlock(f)

    return log.get(model_name, {})

def load_tuning_log(log_path):
    if not os.path.exists(log_path):
        return {}
    try:
        with open(log_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}
