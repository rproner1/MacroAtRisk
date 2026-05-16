from datetime import UTC, datetime
from pathlib import Path
import os
import portalocker
import json
import tempfile
import numpy as np


def get_latest_file(
    file: Path | None = None,
    prefix: str | None = None,
    extension: str = ".csv",
    directory: Path = Path("."),
) -> Path | None:
    """
    Returns the latest file matching the prefix and extension in the directory.

    Args:
        file (Path, optional): The file path without the timestamp. Defaults to None.
        prefix (str): The prefix of the file name. Defaults to None.
        extension (str, optional): The extension of the file. Defaults to ".csv".
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


def _log_lock_context(log_path: Path):
    """Return a filesystem lock context for a given JSON log path."""

    lock_path = log_path.with_suffix(f"{log_path.suffix}.lock")
    return portalocker.Lock(str(lock_path), timeout=30)


def _recover_json_object_from_text(raw_text: str, log_path: Path) -> dict:
    """Recover dict content from corrupted or concatenated JSON text."""

    text = raw_text.strip()
    if not text:
        return {}

    # Fast path for valid JSON object.
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            return decoded
        raise ValueError(f"Expected a JSON object in {log_path}, got {type(decoded).__name__}.")
    except json.JSONDecodeError:
        pass

    # Recovery path for concatenated objects: {..}{..} or mixed whitespace between objects.
    decoder = json.JSONDecoder()
    idx = 0
    merged = {}
    parsed_any = False

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break

        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            # Corruption-tolerant mode: skip ahead to the next object start.
            next_obj_start = text.find("{", idx + 1)
            if next_obj_start == -1:
                break
            idx = next_obj_start
            continue

        if not isinstance(obj, dict):
            # Skip non-dict JSON chunks if they exist in a corrupted file.
            idx = end
            continue

        merged.update(obj)
        parsed_any = True
        idx = end

    if not parsed_any:
        raise ValueError(f"Unable to parse JSON content in {log_path}.")

    return merged


def _read_log_dict(log_path: Path) -> dict:
    """Read a hyperparameter log as dict, with recovery for concatenated JSON content."""

    if not log_path.exists():
        return {}

    raw_text = log_path.read_text(encoding="utf-8")
    return _recover_json_object_from_text(raw_text, log_path)


def _atomic_write_json(log_path: Path, payload: dict) -> None:
    """Atomically write JSON payload to disk to avoid partial files."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=log_path.parent,
        delete=False,
        prefix=f"{log_path.stem}.",
        suffix=".tmp",
    ) as tmp_file:
        json.dump(payload, tmp_file, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = Path(tmp_file.name)

    temp_path.replace(log_path)


def repair_hyperparameters_log(log_path: str | Path) -> dict:
    """Repair and rewrite a possibly corrupted hyperparameter log file."""

    log_path = Path(log_path)
    with _log_lock_context(log_path):
        recovered = _read_log_dict(log_path)
        _atomic_write_json(log_path, recovered)
    return recovered

def check_hps_exist(model_name: str, log_path: str | Path) -> bool:
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
    
    log_path = Path(log_path)

    if not log_path.exists():
        return False

    with _log_lock_context(log_path):
        log = _read_log_dict(log_path)

    return model_name in log


def save_hyperparameters(hps: dict, model_name: str, log_path: str | Path, overwrite: bool = False):
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
    log_path = Path(log_path)
    with _log_lock_context(log_path):
        log = _read_log_dict(log_path)

        # If model not in log, always save.
        should_write = model_name not in log

        if not should_write and overwrite:
            old_value = log[model_name].get('value', 0.0)
            should_write = hps.get('value', 0.0) > old_value

        if should_write:
            log[model_name] = hps
            _atomic_write_json(log_path, log)

    return None


def load_hyperparameters(model_name: str, log_path: str | Path) -> dict:
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
    log_path = Path(log_path)

    if not log_path.exists():
        return {}

    with _log_lock_context(log_path):
        log = _read_log_dict(log_path)

    return log.get(model_name, {})


def load_tuning_log(log_path: str | Path) -> dict:
    log_path = Path(log_path)
    if not log_path.exists():
        return {}
    with _log_lock_context(log_path):
        return _read_log_dict(log_path)


def concat_shap_values(
    shap_dir: str | Path,
    output_path: str | Path,
    glob_pattern: str = "*.npy",
) -> np.ndarray:
    shap_dir = Path(shap_dir)
    output_path = Path(output_path)

    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    all_shap_files = list(shap_dir.glob(glob_pattern))
    if not all_shap_files:
        raise ValueError(f"No SHAP files found in {shap_dir} matching {glob_pattern}.")

    arr_list = []
    for file in all_shap_files:
        try:
            arr = np.load(file)
            arr_list.append(arr)
        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")

    if not arr_list:
        raise ValueError(f"No valid SHAP files could be read from {shap_dir}.")

    concatenated_arr = np.concatenate(arr_list, axis=0) # concatenate along the first dimension (samples)
    np.save(output_path, concatenated_arr)

    return concatenated_arr