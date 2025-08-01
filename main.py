import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from atom.experiment.dataset import load_data
from atom.experiment.module import atom, plugin, set_module
from atom.experiment.utils import (
    duration_formatter,
    get_file_count,
    get_next_log_file,
    load_json,
    save_json,
)
from atom.llm import get_call_count, get_token, set_model

# Configuration constants
LOG_DIR = "log/{dataset}/{size}"


# Dataset configuration
@dataclass
class DatasetConfig:
    question_key: str | list[str]
    answer_key: str
    module_type: str
    scoring_function: str

    def requires_context(self) -> bool:
        return self.module_type == "multi-hop"


# Dataset configuration mapping
DATASET_CONFIGS = {
    "gsm8k": DatasetConfig(
        question_key="question", answer_key="answer", module_type="math", scoring_function="score_math"
    ),
    "math": DatasetConfig(
        question_key="problem", answer_key="solution", module_type="math", scoring_function="score_math"
    ),
    "bbh": DatasetConfig(
        question_key="input", answer_key="target", module_type="multi-choice", scoring_function="score_mc"
    ),
    "mmlu": DatasetConfig(
        question_key=["Question", "A", "B", "C", "D"],
        answer_key="Answer",
        module_type="multi-choice",
        scoring_function="score_mc",
    ),
    "hotpotqa": DatasetConfig(
        question_key="question", answer_key="answer", module_type="multi-hop", scoring_function="score_mh"
    ),
    "longbench": DatasetConfig(
        question_key="input", answer_key="answers", module_type="multi-hop", scoring_function="score_mh"
    ),
}


class ExperimentRunner:
    def __init__(self, dataset: str, model: str, start: int = 0, end: int = -1, mode: str = "atom"):
        # Initialize experiment runner
        self.dataset = dataset
        self.start = start
        self.end = None if end == -1 else end
        self.interval = "full" if self.end is None else f"{start}-{end}"
        self.timestamp = time.time()
        self.mode = mode
        # Validate dataset support
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.config = DATASET_CONFIGS[dataset]
        set_model(model)

    def gather_results(self, testset: list[dict[str, Any]]) -> list[Any]:
        # Collect experiment results
        set_module(self.config.module_type)

        question_key = self.config.question_key
        results = []

        if self.config.requires_context():
            from atom.experiment.prompter.multihop import contexts

            # Handle case where question_key is a list
            if isinstance(question_key, list):
                formatted_questions = [self._format_question_from_keys(item, question_key) for item in testset]
                for question, item in tqdm(
                    zip(formatted_questions, testset), total=len(testset), desc=f"Processing {self.dataset} tasks"
                ):
                    results.append(atom(question, contexts(item, self.dataset)))
            else:
                results.extend(
                    [atom(item[question_key], contexts(item, self.dataset)) for item in tqdm(testset, desc=f"Processing {self.dataset} tasks")]
                )
        else:
            # Handle case where question_key is a list
            if isinstance(question_key, list):
                results.extend(
                    [atom(self._format_question_from_keys(item, question_key)) for item in tqdm(testset, desc=f"Processing {self.dataset} tasks")]
                )
            else:
                results.extend([atom(item[question_key]) for item in tqdm(testset, desc=f"Processing {self.dataset} tasks")])
        return results

    def _format_question_from_keys(self, item: dict[str, Any], keys: list[str]) -> str:
        # When question_key is a list, concatenate values from multiple keys into a single question
        parts = [f"{key}: {item[key]}" for key in keys if key in item]
        return "\n".join(parts)

    def construct_entry(self, result: tuple[dict[str, Any], Any], data: dict[str, Any]) -> dict[str, Any]:
        # Construct result entry
        result_data, log = result
        question_key = self.config.question_key
        answer_key = self.config.answer_key

        # Handle case where question_key is a list
        if isinstance(question_key, list):
            question = self._format_question_from_keys(data, question_key)
        else:
            question = data[question_key]

        groundtruth = data[answer_key]

        entry = {
            "problem": question,
            "groundtruth": groundtruth,
            "response": result_data.get("response"),
            "answer": result_data.get("answer"),
            "log": log,
        }

        # Dynamically import scoring function
        scoring_function = getattr(
            __import__("atom.experiment.utils", fromlist=[self.config.scoring_function]), self.config.scoring_function
        )

        # Pass different parameters based on scoring function
        if self.config.scoring_function == "score_math":
            entry["score"] = scoring_function(entry["answer"], groundtruth, self.dataset)
        else:
            entry["score"] = scoring_function(entry["answer"], groundtruth)
        return entry

    def update_score_log(self, accuracy: float) -> None:
        # Update score log
        log_entry = {
            "start": self.start,
            "end": self.end,
            "token": {"prompt": get_token()[0], "completion": get_token()[1]},
            "call_count": get_call_count(),
            "accuracy": accuracy,
        }

        score_log_file = LOG_DIR.format(dataset=self.dataset, size=self.interval) + "/score.json"
        existing_log = load_json(score_log_file) if os.path.exists(score_log_file) else {}
        count = get_file_count(LOG_DIR, self.interval, self.dataset, exclude_score=True)

        if self.dataset not in existing_log:
            existing_log[self.dataset] = {}
        existing_log[self.dataset][str(count)] = log_entry
        save_json(score_log_file, existing_log)

    def run(self) -> float:
        # Run experiment and return accuracy
        print(f"Running {self.mode} experiment on {self.dataset} dataset from index {self.start} to {self.end}")

        # Load test set
        testset = load_data(self.dataset, "test")[self.start : self.end]
        results = self.gather_results(testset)

        # Build results
        json_obj = [self.construct_entry(result, data) for result, data in zip(results, testset)]
        accuracy = sum(entry["score"] for entry in json_obj) / len(json_obj)

        # Save results
        log_file = get_next_log_file(LOG_DIR, self.interval, self.dataset)
        save_json(log_file, json_obj)

        # Update score log
        self.update_score_log(accuracy)

        # Print result summary
        print(f"Unsolved: {round((1 - accuracy) * len(json_obj))}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Time taken: {duration_formatter(time.time() - self.timestamp)}")

        return accuracy


def optimize_dataset(dataset: str, model: str, start: int = 0, end: int = -1):
    # Optimize dataset questions and save to new file
    print(f"Optimizing {dataset} dataset questions from index {start} to {end}")
    timestamp = time.time()

    # Set model and module
    set_model(model)
    config = DATASET_CONFIGS[dataset]
    set_module(config.module_type)

    # Load test set
    testset = load_data(dataset, "test")[start : None if end == -1 else end]
    question_key = config.question_key
    if isinstance(question_key, list):
        question_key = question_key[0]

    # Create tasks
    def process_item(item):
        try:
            if config.requires_context():
                from atom.experiment.prompter.multihop import contexts

                optimized_question = plugin(item[question_key], contexts(item, dataset))
            else:
                optimized_question = plugin(item[question_key])

            # Create new entry
            new_item = item.copy()
            new_item["original_question"] = item[question_key]
            new_item[question_key] = optimized_question
            return new_item
        except Exception as e:
            print(f"Error processing item: {e}")
            return item  # Return original item on error

    # Process all items sequentially ## FIXME: take out
    optimized_data = [process_item(item) for item in tqdm(testset, desc=f"Optimizing {dataset} questions")]

    # Ensure output directory exists
    os.makedirs(f"experiment/data/{dataset}", exist_ok=True)

    # Save optimized dataset
    output_path = f"experiment/data/{dataset}/contracted.json"
    save_json(output_path, optimized_data)

    elapsed_time = time.time() - timestamp
    print(f"Optimized dataset saved to {output_path}")
    print(f"Time taken: {duration_formatter(elapsed_time)}")

    return optimized_data


def main():
    # Main function
    parser = argparse.ArgumentParser(description="Run experiments on various datasets")
    parser.add_argument(
        "--dataset", type=str, default="math", choices=list(DATASET_CONFIGS.keys()), help="Dataset to run experiment on"
    )
    parser.add_argument("--start", type=int, default=0, help="Start index of the dataset")
    parser.add_argument("--end", type=int, default=10, help="End index of the dataset (-1 for all)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for the experiment")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["atom", "plugin"],
        default="atom",
        help="Mode: atom (standard experiment) or plugin (generate contracted dataset)",
    )

    args = parser.parse_args()

    if args.mode == "plugin":
        # Run plugin mode
        optimize_dataset(dataset=args.dataset, model=args.model, start=args.start, end=args.end)
    elif args.mode == "atom":
        # Run standard experiment
        runner = ExperimentRunner(
            dataset=args.dataset, model=args.model, start=args.start, end=args.end, mode=args.mode
        )
        runner.run()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
