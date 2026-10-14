import numpy as np
import json
from pathlib import Path


def read_file_as_single_string(file_path, encoding):
    with open(file_path, encoding=encoding) as file:
        return file.read()


def read_file_as_lines(file_path):
    with open(file_path) as file:
        lines = [line.strip() for line in file]

    return lines


def read_file_to_np_array(file_path, dtype):
    with open(file_path) as file:
        lines = [line.strip() for line in file]

    return np.asarray(lines, dtype=dtype)


def save_file(file_path, content, mode='w'):
    with open(file_path, mode, newline='') as file:
        file.write(content)


def save_object_to_json(_object, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filename, 'w') as file:
            json.dump(_object, file)
    except Exception as error:
        print(f'Error occurred in save_object_to_json: {error}')

        raise error


def load_object_from_json(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return dict()
    except Exception as error:
        print(f'Error occurred in load_object_from_json: {error}')

        raise error
