from os import PathLike, scandir
from pathlib import Path
from typing import List, Any, Iterable, Union, Iterator, Dict, Tuple, Literal
import json
import os
import shutil
from pathlib import Path
import logging


def _walk(	path: PathLike, 
			depth: Union[int, float] = float('inf')) -> Iterator[PathLike]:
	"""Recursively list files and directories up to a certain depth"""
	if depth is None: depth = float('inf')
	depth -= 1
	with scandir(path) as p:
		for entry in p:
			if entry.is_dir() and depth > 0:
				yield from _walk(entry.path, depth)
			else:
				yield entry.path


def list_dir(	dir_path: PathLike, 
				relative: bool = False, 
				depth: Union[int, float] = float('inf')) -> Iterator[Path]:
	dir_path = Path(dir_path).absolute()
	for f in _walk(dir_path, depth):
		if relative: yield Path(f).relative_to(dir_path)
		else: yield Path(f)

def remove_path(path: PathLike) -> None:
	"""
	remove path
	"""
	path = Path(path)
	if path.exists():
		if path.is_symlink():
			path.unlink()
		elif path.is_dir():
			shutil.rmtree(path)
		else:
			os.remove(path)



data_path  = "augmented_rlbench"
save_dir = "playground/racer_llava_data"
remove_path(save_dir)
os.makedirs(save_dir, exist_ok=True)

task_data = {}
for split_path in list_dir(data_path, depth=1):
    for task_path in list_dir(split_path, depth=1):
        task = task_path.name
        if task in ["retry", "log"]: continue
        for ep in sorted(os.listdir(task_path), key=lambda x: int(x.split("_")[0])):
            file_path = os.path.join(task_path, ep, "llava.json")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"llava.json not found: {file_path}")
            if task not in task_data.keys():
                task_data[task] = []
            task_data[task].extend(json.load(open(file_path)))

all_data = []
for task, data in task_data.items():
    with open(os.path.join(save_dir, task + ".json"), "w") as f:
        json.dump(data, f, indent=1)
    print(task, len(data))
    all_data.extend(data)

with open(os.path.join(save_dir, "all_tasks.json"), "w") as f:
    json.dump(all_data, f)
print("total data size: ", len(all_data))