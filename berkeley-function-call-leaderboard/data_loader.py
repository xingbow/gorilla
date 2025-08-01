"""
Data loading utilities for BFCL (Berkeley Function Call Leaderboard) datasets.

This module provides functions to load function calling instances and their corresponding
ground truth answers from the BFCL dataset files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from bfcl_eval.constants.eval_config import PROJECT_ROOT
from bfcl_eval.utils import load_file




class BFCLDataLoader:
    """
    A data loader for BFCL (Berkeley Function Call Leaderboard) datasets.
    
    This class provides methods to load function calling instances and their
    corresponding ground truth answers from the dataset files.
    """
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the BFCL data loader.
        
        Args:
            data_dir: Path to the data directory. If None, uses the default data directory.
        """
        if data_dir is None:
            self.data_dir = Path(PROJECT_ROOT) / "bfcl_eval" / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.answer_dir = self.data_dir / "possible_answer"
        
        # Validate directories exist
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.answer_dir.exists():
            raise FileNotFoundError(f"Answer directory not found: {self.answer_dir}")
    
    def get_available_datasets(self) -> List[str]:
        """
        Get a list of available dataset names.
        
        Returns:
            List of dataset names (without .json extension).
        """
        datasets = []
        for json_file in self.data_dir.glob("BFCL_*.json"):
            datasets.append(json_file.stem)
        return sorted(datasets)
    
    def load_questions(self, dataset_name: str, sort_by_id: bool = True) -> List[Dict]:
        """
        Load function calling question instances from a dataset file.
        
        Args:
            dataset_name: Name of the dataset (e.g., "BFCL_v3_simple").
            sort_by_id: Whether to sort the results by ID.
            
        Returns:
            List of question instances, each containing:
            - id: Unique identifier
            - question: Nested array of conversation turns
            - function: Available function definitions (for non-multi-turn datasets)
            - Additional fields for multi-turn datasets (initial_config, path, involved_classes)
        """
        dataset_file = self.data_dir / f"{dataset_name}.json"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        questions = load_file(dataset_file, sort_by_id=sort_by_id)
        
        # For multi-turn datasets, load function definitions separately
        if self.is_multi_turn_dataset(dataset_name):
            questions = self._add_multi_turn_functions(questions)
        
        return questions
    
    def load_ground_truth(self, dataset_name: str, sort_by_id: bool = True) -> List[Dict]:
        """
        Load ground truth answers for a dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "BFCL_v3_simple").
            sort_by_id: Whether to sort the results by ID.
            
        Returns:
            List of ground truth instances, each containing:
            - id: Unique identifier matching the question instance
            - ground_truth: Expected function calls and parameters
        """
        # Some datasets don't have ground truth (e.g., irrelevance tests)
        answer_file = self.answer_dir / f"{dataset_name}.json"
        
        if not answer_file.exists():
            return []
        
        return load_file(answer_file, sort_by_id=sort_by_id)
    
    def load_dataset(self, dataset_name: str, sort_by_id: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Load both questions and ground truth for a dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "BFCL_v3_simple").
            sort_by_id: Whether to sort the results by ID.
            
        Returns:
            Tuple of (questions, ground_truth) lists.
        """
        questions = self.load_questions(dataset_name, sort_by_id)
        ground_truth = self.load_ground_truth(dataset_name, sort_by_id)
        
        return questions, ground_truth
    
    def load_multiple_datasets(self, dataset_names: List[str], sort_by_id: bool = True) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
        """
        Load multiple datasets at once.
        
        Args:
            dataset_names: List of dataset names to load.
            sort_by_id: Whether to sort the results by ID.
            
        Returns:
            Dictionary mapping dataset names to (questions, ground_truth) tuples.
        """
        results = {}
        for dataset_name in dataset_names:
            results[dataset_name] = self.load_dataset(dataset_name, sort_by_id)
        return results
    
    def load_by_category(self, category: str, sort_by_id: bool = True) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
        """
        Load all datasets belonging to a specific category.
        
        Args:
            category: Test category (e.g., "simple", "multi_turn", "live").
            sort_by_id: Whether to sort the results by ID.
            
        Returns:
            Dictionary mapping dataset names to (questions, ground_truth) tuples.
        """
        datasets = []
        for dataset_name in self.get_available_datasets():
            if category in dataset_name:
                datasets.append(dataset_name)
        
        if not datasets:
            raise ValueError(f"No datasets found for category: {category}")
        
        return self.load_multiple_datasets(datasets, sort_by_id)
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Union[str, int, bool]]:
        """
        Get metadata information about a dataset.
        
        Args:
            dataset_name: Name of the dataset.
            
        Returns:
            Dictionary containing dataset metadata:
            - name: Dataset name
            - num_questions: Number of question instances
            - num_ground_truth: Number of ground truth instances
            - has_ground_truth: Whether ground truth exists
            - is_multi_turn: Whether this is a multi-turn dataset
            - category: Extracted category from dataset name
        """
        questions = self.load_questions(dataset_name, sort_by_id=False)
        ground_truth = self.load_ground_truth(dataset_name, sort_by_id=False)
        
        # Extract category from dataset name
        parts = dataset_name.split("_")
        if len(parts) >= 3:
            category = "_".join(parts[2:])  # Remove "BFCL_v3_" prefix
        else:
            category = dataset_name
        
        return {
            "name": dataset_name,
            "num_questions": len(questions),
            "num_ground_truth": len(ground_truth),
            "has_ground_truth": len(ground_truth) > 0,
            "is_multi_turn": "multi_turn" in dataset_name,
            "category": category
        }
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate a dataset for consistency and completeness.
        
        Args:
            dataset_name: Name of the dataset to validate.
            
        Returns:
            Dictionary containing validation results:
            - is_valid: Whether the dataset is valid
            - errors: List of error messages (if any)
            - warnings: List of warning messages (if any)
        """
        errors = []
        warnings = []
        
        try:
            questions = self.load_questions(dataset_name, sort_by_id=False)
            ground_truth = self.load_ground_truth(dataset_name, sort_by_id=False)
        except FileNotFoundError as e:
            errors.append(str(e))
            return {"is_valid": False, "errors": errors, "warnings": warnings}
        
        # Check if all questions have unique IDs
        question_ids = [q["id"] for q in questions]
        if len(question_ids) != len(set(question_ids)):
            errors.append("Duplicate question IDs found")
        
        # Check if ground truth exists and has matching IDs
        if ground_truth:
            ground_truth_ids = [gt["id"] for gt in ground_truth]
            
            # Check for unique ground truth IDs
            if len(ground_truth_ids) != len(set(ground_truth_ids)):
                errors.append("Duplicate ground truth IDs found")
            
            # Check if all ground truth IDs have corresponding questions
            missing_questions = set(ground_truth_ids) - set(question_ids)
            if missing_questions:
                warnings.append(f"Ground truth IDs without corresponding questions: {missing_questions}")
            
            # Check if all questions have ground truth (warn if not)
            missing_ground_truth = set(question_ids) - set(ground_truth_ids)
            if missing_ground_truth:
                warnings.append(f"Questions without ground truth: {len(missing_ground_truth)} instances")
        else:
            warnings.append("No ground truth available for this dataset")
        
        # Validate question structure
        for i, question in enumerate(questions):
            if "id" not in question:
                errors.append(f"Question {i} missing 'id' field")
            if "question" not in question:
                errors.append(f"Question {i} missing 'question' field")
            if "function" not in question:
                errors.append(f"Question {i} missing 'function' field")
            
            # Check question format
            if "question" in question and not isinstance(question["question"], list):
                errors.append(f"Question {i} 'question' field is not a list")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def is_multi_turn_dataset(self, dataset_name: str) -> bool:
        """Check if a dataset is a multi-turn dataset."""
        return "multi_turn" in dataset_name
    
    def _add_multi_turn_functions(self, questions: List[Dict]) -> List[Dict]:
        """
        Add function definitions to multi-turn questions by loading from multi_turn_func_doc.
        
        Args:
            questions: List of question instances without function definitions.
            
        Returns:
            List of question instances with function definitions added.
        """
        # Load function definitions from multi_turn_func_doc directory
        func_doc_dir = self.data_dir / "multi_turn_func_doc"
        
        if not func_doc_dir.exists():
            return questions
        
        # Load all function definitions
        all_functions = {}
        for func_file in func_doc_dir.glob("*.json"):
            with open(func_file, 'r') as f:
                functions = [json.loads(line) for line in f.readlines()]
                all_functions[func_file.stem] = functions
        
        # Add function definitions to questions based on involved_classes
        for question in questions:
            if "involved_classes" in question:
                question_functions = []
                for class_name in question["involved_classes"]:
                    # Map class names to function doc file names
                    func_doc_name = self._get_func_doc_name(class_name)
                    if func_doc_name in all_functions:
                        question_functions.extend(all_functions[func_doc_name])
                question["function"] = question_functions
            else:
                question["function"] = []
        
        return questions
    
    def _get_func_doc_name(self, class_name: str) -> str:
        """Map class name to function doc file name."""
        class_to_doc_mapping = {
            "GorillaFileSystem": "gorilla_file_system",
            "MathAPI": "math_api",
            "MessageAPI": "message_api",
            "TwitterAPI": "posting_api",
            "TicketAPI": "ticket_api",
            "TradingBot": "trading_bot",
            "TravelAPI": "travel_booking",
            "VehicleControlAPI": "vehicle_control"
        }
        return class_to_doc_mapping.get(class_name, class_name.lower())


def load_bfcl_dataset(dataset_name: str, data_dir: Optional[Union[str, Path]] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience function to load a BFCL dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "BFCL_v3_simple").
        data_dir: Path to the data directory. If None, uses the default data directory.
        
    Returns:
        Tuple of (questions, ground_truth) lists.
    """
    loader = BFCLDataLoader(data_dir)
    return loader.load_dataset(dataset_name)


def get_available_datasets(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Get a list of available BFCL dataset names.
    
    Args:
        data_dir: Path to the data directory. If None, uses the default data directory.
        
    Returns:
        List of dataset names.
    """
    loader = BFCLDataLoader(data_dir)
    return loader.get_available_datasets()


if __name__ == "__main__":
    # Example usage
    loader = BFCLDataLoader()
    
    # List available datasets
    print("Available datasets:")
    datasets = loader.get_available_datasets()
    for dataset in datasets:  # Show first 10
        print(f"  - {dataset}")

    dataset_name = datasets[-1]
    
    # Load a simple dataset
    print(f"\nLoading {dataset_name} dataset...")
    questions, ground_truth = loader.load_dataset(dataset_name)
    print(f"Loaded {len(questions)} questions and {len(ground_truth)} ground truth instances")
    
    # Show first question
    if questions:
        print("\nFirst question:")
        print(json.dumps(questions[0], indent=2))
        print(f"ID: {questions[0]['id']}")
        for qid, q in  enumerate(questions[0]['question']):
            print(f"question {qid}: {q[0]['content']}")
        print(f"Available functions: {len(questions[0]['function'])}")
    
    # Show dataset info
    print("\nDataset info:")
    info = loader.get_dataset_info(dataset_name)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Validate dataset
    print("\nValidation results:")
    validation = loader.validate_dataset(dataset_name)
    print(f"  Valid: {validation['is_valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")