import json
import yaml
import os
import pickle


def _ensure_extension(file_path, allowed_extensions):
    """
    Ensure the file has one of the allowed extensions.

    Args:
        file_path (str): Path to check
        allowed_extensions (tuple or str): Allowed file extensions

    Raises:
        ValueError: If the file doesn't have an allowed extension
    """
    if isinstance(allowed_extensions, str):
        allowed_extensions = (allowed_extensions,)

    file_path_lower = str(file_path).lower()
    if not any(file_path_lower.endswith(ext) for ext in allowed_extensions):
        extensions_str = " or ".join([f"'{ext}'" for ext in allowed_extensions])
        raise ValueError(f"File must have {extensions_str} extension: {file_path}")


def _ensure_directory(file_path):
    """
    Create the directory for the given file path if it doesn't exist.

    Args:
        file_path (str): Path to the file

    Returns:
        str: The directory that was checked/created
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return directory


def save_pickle(data, file_path):
    """
    Save a Python object to a pickle file.

    Args:
        data: The Python object to save
        file_path (str): Path to the pickle file to save

    Raises:
        ValueError: If the file extension is not .pkl or .pickle
    """
    # Check file extension
    _ensure_extension(file_path, (".pkl", ".pickle"))

    # Create directory if it doesn't exist
    _ensure_directory(file_path)

    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(file_path):
    """
    Read a pickle file and return its contents as a Python object.

    Args:
        file_path (str): Path to the pickle file to read

    Returns:
        The unpickled Python object

    Raises:
        ValueError: If the file extension is not .pkl or .pickle
        FileNotFoundError: If the file does not exist
    """
    # Check file extension
    _ensure_extension(file_path, (".pkl", ".pickle"))

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading pickle file {file_path}: {str(e)}")


def read_json(file_path):
    """
    Read a JSON file and return its contents as a Python object.

    Args:
        file_path (str): Path to the JSON file to read

    Returns:
        dict or list: The contents of the JSON file as a Python object

    Raises:
        ValueError: If the file is not in JSON format
        FileNotFoundError: If the file does not exist
    """
    # Check file extension
    _ensure_extension(file_path, ".json")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")


def save_json(data, file_path, indent=2):
    """
    Save a Python object to a JSON file.

    Args:
        data (dict or list): The Python object to save
        file_path (str): Path to the JSON file to save
        indent (int, optional): Indentation level for JSON. Default is 2.

    Raises:
        ValueError: If the file extension is not .json
    """
    # Check file extension
    _ensure_extension(file_path, ".json")

    # Create directory if it doesn't exist
    _ensure_directory(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_yaml(file_path):
    """
    Read a YAML file and return its contents as a Python object.

    Args:
        file_path (str): Path to the YAML file to read

    Returns:
        dict or list: The contents of the YAML file as a Python object

    Raises:
        ValueError: If the file is not in YAML format
        FileNotFoundError: If the file does not exist
    """
    # Check file extension
    _ensure_extension(file_path, (".yaml", ".yml"))

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
            return data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {file_path}: {str(e)}")


def save_yaml(data, file_path):
    """
    Save a Python object to a YAML file.

    Args:
        data (dict or list): The Python object to save
        file_path (str): Path to the YAML file to save

    Raises:
        ValueError: If the file extension is not .yaml or .yml
    """
    # Check file extension
    _ensure_extension(file_path, (".yaml", ".yml"))

    # Create directory if it doesn't exist
    _ensure_directory(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_jsonl(data, file_path):
    """
    Save a list of dictionaries to a JSONL file.

    Args:
        data (list of dicts): The list of dictionaries to save
        file_path (str): Path to the JSONL file to save

    Raises:
        ValueError: If the file extension is not .jsonl
    """
    # Check file extension
    _ensure_extension(file_path, ".jsonl")

    # Create directory if it doesn't exist
    _ensure_directory(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def read_jsonl(file_path, force=False):
    """
    Read a JSONL file and return its contents as a list of dictionaries.

    Args:
        file_path (str): Path to the JSONL file to read

    Returns:
        list of dicts: The contents of the JSONL file as a list of dictionaries

    Raises:
        ValueError: If the file extension is not .jsonl
        FileNotFoundError: If the file does not exist
    """
    # Check file extension
    _ensure_extension(file_path, ".jsonl")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    output = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                output.append(json.loads(line))
            except:
                if force:
                    continue
                else:
                    raise ValueError(f"Invalid JSONL format in {file_path}: {line}")
    return output
