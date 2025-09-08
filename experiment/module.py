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

def atom_step(
    question: str,
    contexts: str | None = None,
) -> tuple[dict, str]:
    """
    Reference: https://arxiv.org/pdf/2502.12018 (Algorithm 1)
    Performs one iteration of decomposition and contraction as per the AoT algorithm.
    Returns the decomposition graph and the next question state (Q_{i+1}).
    """
    # 1. Decompose the current question (Algorithm 1, l.4)
    decompose_args = {"contexts": contexts} if module == "multi-hop" else {}
    decompose_result = decompose(question, **decompose_args)

    # 2. Separate independent and dependent subquestions (Algorithm 1, l.8-9)
    independent_subqs = [sq for sq in decompose_result["sub-questions"] if not sq.get("depend")]
    dependent_subqs = [sq for sq in decompose_result["sub-questions"] if sq.get("depend")]

    # 3. Contract subquestions into a new independent question (Algorithm 1, l.10)
    contract_args = {
        "question": question, 
        "decompose_result": decompose_result,
        "independent": independent_subqs, 
        "dependent": dependent_subqs
    }
    if module == "multi-hop":
        contract_args["contexts"] = contexts
    
    contracted_result = contract(**contract_args) # contract from prompter.contract function!

    
    # The new question for the next iteration (Q_{i+1})
    next_question = contracted_result.get("question", "")

    return decompose_result, next_question

def run_aot_algorithm(initial_question: str, contexts: str | None = None) -> tuple[dict, dict]:
    """
    Main loop of Algorithm 1, with logging for each step.
    Returns a tuple containing the final answer and a log dictionary.
    """
    i = 0
    max_depth = None
    current_question = initial_question
    log: dict[str, Any] = {"initial_question": initial_question, "steps": list(), "max_depth": None, "final_step": dict()}

    # Main loop from Algorithm 1 (Lines 3-12)
    while max_depth is None or i < max_depth:
        # Decompose and then contract the question to get the next state
        decomposition_graph, next_question = atom_step(current_question, contexts=contexts)

        # Log information for the current iteration
        step_log = {
            "iteration": i,
            "current_question": current_question,
            "decomposition_graph": decomposition_graph,
            "next_question": next_question
        }
        log["steps"].append(step_log)

        # Determine max_depth on the first iteration (Algorithm 1, l. 5-7)
        if max_depth is None:
            max_depth = calculate_depth(decomposition_graph.get("sub-questions", []))
            log["max_depth"] = max_depth
            
        current_question = next_question
        i += 1 # Increment iteration counter (Algorithm 1, l. 11)

    # Solve the final question (Algorithm 1, l. 13)
    solve_args = (current_question, contexts) if module == "multi-hop" else (current_question,)
    final_answer = direct(*solve_args) # Using 'direct' as the solveLLM function

    # Log the final step
    log["final_step"] = {
        "final_question": current_question,
        "final_answer": final_answer
    }

    return final_answer, log  # Return final answer and the log (Algorithm 1, l. 14)


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
def contract(question: str, sub_result: dict, independent: list, dependent: list, contexts: str| None =None):
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
