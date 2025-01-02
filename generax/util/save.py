import pickle
from pathlib import Path
from typing import Union
import csv
import numpy as np
import jax.numpy as jnp

suffix = '.pickle'

def save_pytree(data, path: Union[str, Path], overwrite: bool = True):
  path = Path(path)
  if path.suffix != suffix:
    path = path.with_suffix(suffix)
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists():
    if overwrite:
      path.unlink()
    else:
      raise RuntimeError(f'File {path} already exists.')
  with open(path, 'wb') as file:
    pickle.dump(data, file)

def load_pytree(path: Union[str, Path]):
  path = Path(path)
  if not path.is_file():
    raise ValueError(f'Not a file: {path}')
  if path.suffix != suffix:
    raise ValueError(f'Not a {suffix} file: {path}')
  with open(path, 'rb') as file:
    data = pickle.load(file)
  return data


def dict_to_csv(data_dict, filename):
    """
    Saves a dictionary of 1D numpy arrays to a CSV file.

    :param data_dict: Dictionary with metric names as keys and 1D numpy arrays as values.
    :param filename: Name of the CSV file to save.
    """
    # Ensure all arrays are of the same length
    array_length = len(next(iter(data_dict.values())))
    if not all(len(arr) == array_length for arr in data_dict.values()):
        raise ValueError("All arrays must be of the same length")

    # Transpose the data to get rows for each timestep
    transposed_data = np.column_stack(list(data_dict.values()))

    # Write to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(data_dict.keys())
        # Write the rows for each timestep
        writer.writerows(transposed_data)

def csv_to_dict(filename, to_array=False):
    """
    Reads a CSV file and converts it to a dictionary with metric names as keys and 1D numpy arrays as values.

    :param filename: Name of the CSV file to read.
    :return: Dictionary with metric names as keys and 1D numpy arrays as values.
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the first line to get the headers (metric names)

        # Initialize a dictionary with the headers and empty lists
        data_dict = {header: [] for header in headers}

        # Read the rest of the rows
        for row in reader:
            for header, value in zip(headers, row):
                data_dict[header].append(float(value))  # Convert values to float

    # Convert lists to numpy arrays
    if to_array:
      for key in data_dict.keys():
          data_dict[key] = jnp.array(data_dict[key])

    return data_dict
