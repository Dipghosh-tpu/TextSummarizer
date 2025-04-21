import os
from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict
from src.textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # This method remains the same as it handles the tokenization
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)
            
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def convert(self):
        # Load the pre-split datasets from json files
        train_dataset = load_dataset("json", 
                                   data_files=os.path.join(self.config.data_inp_path, "train.json"))["train"]
        test_dataset = load_dataset("json", 
                                  data_files=os.path.join(self.config.data_inp_path, "test.json"))["train"]
        val_dataset = load_dataset("json", 
                                 data_files=os.path.join(self.config.data_inp_path, "val.json"))["train"]
        
        # Create a DatasetDict with all splits
        dataset_samsum = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
            "validation": val_dataset
        })

        # Apply tokenization to all splits
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features, 
            batched=True,
            remove_columns=dataset_samsum["train"].column_names  # Remove original columns after tokenization
        )
        
        # Save the processed dataset
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.data_path))