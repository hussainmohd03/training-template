from pathlib import Path
import os
from datasets import Dataset, load_dataset
from enum import Enum


ds_types = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}

class DatasetSplitter:
    def __init__(self, input_file: str, output_dir: str, val_size: float = 0.1,
                 test_size: float = 0.1, seed: int = 42):

        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

    def load(self) -> Dataset: 
        if os.path.exists(self.input_file):
            file_extension = os.path.splitext(self.input_file)[1][1:]
            ds_type = ds_types.get(file_extension)
            if not ds_type: 
                raise ValueError(f"Unsupported file type: {file_extension}")
            return load_dataset(ds_type, data_files=self.input_file, split="train")

        return load_dataset(self.input_file, split="train")

    def split(self, ds: Dataset): 
        first = ds.train_test_split(test_size=self.val_size + self.test_size, seed=self.seed)
        temp_ratio = self.test_size / (self.val_size + self.test_size)
        second = first["test"].train_test_split(test_size=temp_ratio, seed=self.seed)

        splits = {
            'train': first["train"],
            'val': second['train'],
            'test': second['test'] 
        }
        return splits
    
    def save(self, splits): 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, dataset in splits.items():
            filename = self.output_dir / f"{split_name}.jsonl"

            dataset.to_json(filename)
            print(f"Wrote {len(dataset)} rows to {split_name}.jsonl")

            
            

    def run(self): 
        ds = self.load()
        splits = self.split(ds)
        self.save(splits)
        return splits
