from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Any

from atom.experiment.prompter import math, multichoice, multihop
from atom.experiment.utils import (
    calculate_depth,
    extract_json,
    extract_xml,
    score_math,
    score_mc,
    score_mh,
)
from atom.llm import gen

count = 0
MAX_RETRIES = 5
LABEL_RETRIES = 3
ATOM_DEPTH = 3
score = None

module = None
prompter = None
def set_module(module_name):  # math, multi-choice, multi-hop
    global module, prompter, score
    module = module_name
    if module == "math":
        prompter = math
        score = score_math
    elif module == "multi-choice":
        prompter = multichoice
        score = score_mc
    elif module == "multi-hop":
        prompter = multihop
        score = score_mh

def retry(func_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global MAX_RETRIES
            retries = MAX_RETRIES
            while retries >= 0:
                prompt = getattr(prompter, func_name)(*args, **kwargs)
                
                if module == "multi-hop" and func_name != "contract":
                    response = gen(prompt, response_format="json_object")
                    result = extract_json(response)
                    result["response"] = response
                else:
                    if func_name == "label":
                        response = gen(prompt, response_format="json_object")
                        result = extract_json(response)
                    else:
                        response = gen(prompt, response_format="text")
                        result = extract_xml(response)
                        if isinstance(result, dict):
                            result["response"] = response
                
                if prompter.check(func_name, result):
                    return result
                retries -= 1
            
            global count
            if MAX_RETRIES > 1:
                count += 1
            if count > 300:
                raise Exception("Too many failures")
            return result if isinstance(result, dict) else {}
        return wrapper
    return decorator

def decompose(question: str, **kwargs):
    retries = LABEL_RETRIES
    if module == "multi-hop":
        if "contexts" not in kwargs:
            raise Exception("Multi-hop must have contexts")
        contexts = kwargs["contexts"]
        multistep_result = multistep(question, contexts)
        while retries > 0:
            label_result = label(question, multistep_result)
            try:
                if len(label_result["sub-questions"]) != len(multistep_result["sub-questions"]):
                    retries -= 1
                    continue
                calculate_depth(label_result["sub-questions"])
                break
            except Exception as e:
                print(f"Error in decompose: {e}, retrying...")
                retries -= 1
                continue
        for step, note in zip(multistep_result["sub-questions"], label_result["sub-questions"]):
            step["depend"] = note["depend"]
        return multistep_result
    else:
        multistep_result = multistep(question)
        while retries > 0:
            result = label(question, multistep_result["response"], multistep_result["answer"])
            try:
                calculate_depth(result["sub-questions"])
                result["response"] = multistep_result["response"]
                break
            except Exception as e:
                print(f"Error in decompose: {e}, retrying...")
                retries -= 1
                continue
        return result

def merging(question: str, decompose_result: dict, independent_subqs: list, dependent_subqs: list, **kwargs):
    contract_args = (
        (question, decompose_result, independent_subqs, dependent_subqs, kwargs["contexts"])
        if module == "multi-hop"
        else (question, decompose_result, independent_subqs, dependent_subqs)
    )
    contractd_result = contract(*contract_args)
    
    # Extract thought process and optimized question
    contractd_thought = contractd_result.get("response", "")
    contractd_question = contractd_result.get("question", "")
    
    # Solve the optimized question
    direct_args = (
        (contractd_question, contractd_result.get("context", kwargs.get("contexts")))
        if module == "multi-hop"
        else (contractd_question,)
    )
    contraction_result = direct(*direct_args)
    
    return contractd_thought, contractd_question, contraction_result

def atom(
    question: str,
    contexts: str | None = None,
    direct_result: dict | None = None,
    decompose_result: dict | None = None,
    depth: int | None = None,
    log: dict | None = None,
    atom_method: str = "decompose"
) -> tuple[dict | None, dict]:
    """
    Efficiently processes the query by only running computations
    required for the selected `atom_method`.
    """
    # 1. Initialize logging and check recursion depth
    log = log if log is not None else {}
    index = len(log)
    log[index] = {}

    if depth == 0:
        return None, log

    # direct - Simplest path
    if atom_method == "direct":
        if direct_result is None:
            direct_args = (question, contexts) if module == "multi-hop" else (question,)
            direct_result = direct(*direct_args)
        log[index]["direct"] = direct_result
        return direct_result, log

    # 'decompose' result is a prerequisite for other methods.
    if decompose_result is None:
        decompose_args = {"contexts": contexts} if module == "multi-hop" else {}
        decompose_result = decompose(question, **decompose_args)
    log[index]["decompose"] = decompose_result
    
    # decompose
    if atom_method == "decompose":
        return decompose_result, log

    # if depth is None:  # nothing's done with this
    #     depth = min(ATOM_DEPTH, calculate_depth(decompose_result["sub-questions"]))

    contraction_result = None
    if atom_method in ["contract", "ensemble", "original"]:
        independent_subqs = [sq for sq in decompose_result["sub-questions"] if not sq.get("depend")]
        dependent_subqs = [sq for sq in decompose_result["sub-questions"] if sq.get("depend")]
        
        merging_args = {
            "question": question, "decompose_result": decompose_result,
            "independent_subqs": independent_subqs, "dependent_subqs": dependent_subqs
        }
        if module == "multi-hop":
            merging_args["contexts"] = contexts
        
        contractd_thought, contractd_question, contraction_result = merging(**merging_args)
        
        contraction_result["contraction_thought"] = contractd_thought
        contraction_result["sub-questions"] = [*independent_subqs, {
            "description": contractd_question,
            "response": contraction_result.get("response", ""),
            "answer": contraction_result.get("answer", ""),
            "depend": []
        }]
        log[index]["contract"] = contraction_result

    # contract - depends on decompose
    if atom_method == "contract":
        return contraction_result, log

    # ensemble - depends on all
    if atom_method == "ensemble":
        if direct_result is None:
            direct_args = (question, contexts) if module == "multi-hop" else (question,)
            direct_result = direct(*direct_args)
        log[index]["direct"] = direct_result

        ensemble_args: list[Any] = [
            question,
            [direct_result["response"], decompose_result["response"], contraction_result["response"]]
        ]
        if module == "multi-hop":
            ensemble_args.append(contexts)
        
        ensemble_result = ensemble(*ensemble_args)
        log[index]["ensemble"] = ensemble_result
        return ensemble_result, log

    # original
    if atom_method == "original":
        if direct_result is None:
            direct_args = (question, contexts) if module == "multi-hop" else (question,)
            direct_result = direct(*direct_args)
        log[index]["direct"] = direct_result

        # Get ensemble result
        ensemble_args = [
            question,
            [direct_result["response"], decompose_result["response"], contraction_result["response"]]
        ]
        if module == "multi-hop":
            ensemble_args.append(contexts)
        ensemble_result = ensemble(*ensemble_args)
        log[index]["ensemble"] = ensemble_result
        
        # Score the results against the ensemble answer
        ensemble_answer = ensemble_result.get("answer", "")
        all_results = [direct_result, decompose_result, contraction_result]

        if all(r.get("answer") == ensemble_answer for r in all_results):
            scores = [1.0, 1.0, 1.0]
        else:
            if score is None:
                raise ValueError("Score function is not set. Please call set_module before using scoring.")
            scores = [score(r.get("answer", ""), ensemble_answer) for r in all_results]
        log[index]["scores"] = scores
        
        # Select the best method based on the calculated scores
        methods = {
            2: ("contract", contraction_result),
            0: ("direct", direct_result),
            1: ("decompose", decompose_result),
            -1: ("ensemble", ensemble_result)
        }
        
        max_score_index = scores.index(max(scores))
        method, result = methods.get(max_score_index, methods[-1])
        
        log[index]["method"] = method
        
        # Return appropriate result format
        if index == 0:
            return {
                "method": method,
                "response": result.get("response"),
                "answer": result.get("answer"),
            }, log
        return result, log

    raise ValueError(f"Unknown atom_method: '{atom_method}'")

def plugin(question: str, contexts: str | None = None, sample_num: int = 3):
    # Create tasks for parallel execution
    def process_sample():
        # Get decompose result
        decompose_args = {"contexts": contexts} if module == "multi-hop" else {}
        decompose_result = decompose(question, **decompose_args)
        
        # Separate independent and dependent sub-questions
        independent_subqs = [sub_q for sub_q in decompose_result["sub-questions"] if len(sub_q["depend"]) == 0]
        dependent_subqs = [sub_q for sub_q in decompose_result["sub-questions"] if sub_q not in independent_subqs]
        
        # Get contraction result
        merging_args = {
            "question": question,
            "decompose_result": decompose_result,
            "independent_subqs": independent_subqs,
            "dependent_subqs": dependent_subqs
        }
        if module == "multi-hop":
            merging_args["contexts"] = contexts
            
        contractd_thought, contractd_question, contraction_result = merging(**merging_args)
        
        return {
            "decompose_result": decompose_result,
            "contractd_thought": contractd_thought,
            "contractd_question": contractd_question,
            "contraction_result": contraction_result
        }
    
    # Execute all samples sequentially (no async)
    all_results = [process_sample() for _ in range(sample_num)]
    
    # Get direct result for original question
    direct_args = (question, contexts) if module == "multi-hop" else (question,)
    direct_result = direct(*direct_args)
    
    # Get ensemble result from all contracted results plus direct result
    all_responses = [direct_result["response"]] + [r["contraction_result"]["response"] for r in all_results]
    ensemble_args: list[str | list[str] | None] = [question, all_responses]
    if module == "multi-hop":
        ensemble_args.append(contexts)
    
    ensemble_result = ensemble(*ensemble_args)
    ensemble_answer = ensemble_result.get("answer", "")
    
    # Calculate scores for each contracted result
    scores = []
    token_counts = []
    
    for result in all_results:
        contraction_result = result["contraction_result"]
        # Calculate score compared to ensemble answer
        if score is None:
            raise Exception("Score function is not set. Please call set_module before using scoring.")
        scores.append(score(contraction_result["answer"], ensemble_answer))
        
        # Estimate token count for the response
        token_counts.append(len(contraction_result.get("response", "").split()))
    
    # Find the best result(s) - those with the highest score
    max_score = max(scores)
    best_indices = [i for i, s in enumerate(scores) if s == max_score]
    
    # Among the best results, find the one with the lowest token count
    best_index = min(best_indices, key=lambda i: token_counts[i])
    
    # Return the best result
    best_result = all_results[best_index]
    return best_result["contractd_question"]

@retry("direct")
def direct(question: str | Iterable[str], contexts: str | None =None):
    if isinstance(question, (list | tuple)):
        question = ''.join(map(str, question))
    pass

@retry("multistep")
def multistep(question: str, contexts: str | None =None):
    pass

@retry("label")
def label(question: str, sub_questions: str, answer: str | None =None):
    pass

@retry("contract")
def contract(question: str, sub_result: dict, independent_subqs: list, dependent_subqs: list, contexts: str| None =None):
    pass

@retry("ensemble")
def ensemble(question: str, results: list, contexts: str | None =None):
    pass

@contextmanager
def temporary_retries(value):
    global MAX_RETRIES
    original = MAX_RETRIES
    MAX_RETRIES = value
    try:
        yield
    finally:
        MAX_RETRIES = original
