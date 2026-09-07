import inspect
import json
import pathlib
import re
import sys
import random
import ast
from typing import Callable

import cloudpickle


for entry in (pathlib.Path(__file__).resolve().parent,
               pathlib.Path(__file__).resolve().parent.parent):
    if entry.as_posix() not in sys.path:
        sys.path.insert(0, entry.as_posix())
import difftrust

MANIPULATED_PROMPTS = ("prompt1a", "prompt1c", "prompt1p", "prompt2ac", "prompt2ap", "prompt2cp", "prompt3acp")

# Start of a function definition. Used to locate the target function in a prompt without relying on
# its name: the manipulated prompts rename it to "candidate" in several variants.
DEF_RE = re.compile(r"^[ \t]*def\s+\w+\s*\(", re.M)

class Instance:
    generator_nb_try: int = 100

    def __init__(self, info: dict):
        self.prompt = info["prompt"]
        # The N in "HumanEval/N". Kept separate from self.name, which is the entry point and is not
        # unique across the 164 tasks (six of them occur twice).
        self.task_id = int(info["name"].split("/")[1])
        self.entry_point = info["entry_point"]
        self.canonical_solution = info["solution"]
        self.test = info["test_case"]

        self.code = self.canonical_solution

        self.manipulated_prompts = {
            typ: info[typ] for typ in MANIPULATED_PROMPTS
        }

        # Attributes
        self.name = self.make_name()
        self.description = self.make_description()
        self.ground_truth = self.make_ground_truth()
        self.spec = self.make_spec()
        self.inputs_corpus = self.make_inputs_corpus()
        # None where HumanEvalComm defines no manipulation for this variant. Passing that None on to
        # make_spec would be read as "use the default prompt" and silently produce a copy of the
        # baseline spec, so absent variants are kept explicitly absent instead.
        self.manipulated_specs = {
            typ: (self.make_spec(prompt) if prompt is not None else None)
            for typ, prompt in self.manipulated_prompts.items()
        }
        # A variant that renders identically to the baseline carries no manipulation this pipeline
        # can express: make_spec always uses the ground truth's name and signature, so a
        # manipulation living only in the function name or the parameter names is invisible here.
        # Mark those absent too, rather than leave a duplicate of the baseline labelled as ambiguous.
        baseline = str(self.spec)
        for typ, spec in self.manipulated_specs.items():
            if spec is not None and str(spec) == baseline:
                self.manipulated_specs[typ] = None

    def blind_generator(self):
        inputs = random.choice(self.inputs_corpus)
        return tuple(difftrust.generic.generic_mutator(x, random.choice([1, 10, 30])) for x in inputs)

    def filtered_generator(self):
        for i in range(self.generator_nb_try):
            inputs = random.choice(self.inputs_corpus)
            inputs = tuple(difftrust.generic.generic_mutator(x, random.choice([1, 10, 30])) for x in inputs)
            try:
                self.ground_truth(*inputs)
                return inputs
            except Exception:
                pass

        return random.choice(self.inputs_corpus)

    def make_name(self):
        return self.entry_point

    def make_description(self, prompt: str = None):
        """ Everything after the target function's signature, i.e. its docstring.

        The target is taken to be the *last* function defined in the prompt: HumanEval puts helper
        functions (poly, encode_shift, is_palindrome, ...) before it. Anchoring on `def` rather than
        on `self.name` matters for the manipulated prompts, which rename the function to "candidate"
        in the 1a/1p/2ac/2ap/3acp variants - splitting on the real entry point would then either
        find nothing or latch onto a mention of it inside a doctest.
        """
        prompt = self.prompt if prompt is None else prompt
        if not prompt:
            return ""
        matches = list(DEF_RE.finditer(prompt))
        if not matches:
            return ""
        txt = prompt[matches[-1].start():]
        end_of_signature = txt.find(':\n')
        if end_of_signature == -1:
            return ""
        return txt[end_of_signature + 1:]

    def make_ground_truth(self) -> Callable:
        namespace = {}
        exec(self.code, namespace)
        return namespace[self.name]

    # Do not remove this if you want to use our generated instances
    def print_hi(self):
        print("hi")

    def make_spec(self, prompt: str = None):
        """ Specification of the task, built from `prompt` (the original one by default).

        Pass one of `self.manipulated_prompts` to get the specification of an ambiguous variant.
        """
        name = self.make_name()
        description = self.make_description(prompt)
        signature = str(inspect.signature(self.ground_truth))
        return difftrust.specification.Specification(
            name=name,
            signature=signature,
            description=description
        )
    
    def parse_cases(self):
        """HumanEvalComm test_case is a literal list of {input, output, relation} dicts."""
        return ast.literal_eval(self.test)

    def args_of(self, case, ns):
        inp = case["input"]
        if not isinstance(inp, str):        
            inp = repr(inp)
        return eval(f"({inp},)", dict(ns))  

    def make_inputs_corpus(self):
        ns = {}
        exec(self.code, ns)

        input_corpus = []
        for case in self.parse_cases():
            try:
                input_corpus.append(self.args_of(case, ns))
            except Exception:
                continue
        return input_corpus
    
    def check_validity(self):
        ns = {}
        exec(self.code, ns)

        def as_value(s):
            if not isinstance(s, str):
                return s
            try:
                return eval(s, dict(ns))
            except Exception:
                return s            

        for case in self.parse_cases():
            rel = case["relation"]
            try:
                args = self.args_of(case, ns)
                if rel == "==":
                    got = self.ground_truth(*args)
                    if got != as_value(case["output"]) and str(got) != str(case["output"]).strip():
                        return False
                elif "candidate" in rel:    # e.g. "abs(candidate(1.33) - 0.33) < 1e-6"
                    if not eval(rel, dict(ns, candidate=self.ground_truth)):
                        return False
                else:
                    return False           
            except Exception:
                return False
        return True

 
    def check_speed(self):
        def func(*args):
            try:
                self.ground_truth(*args)
            except Exception as e:
                return e

        return difftrust.checking.check_speed(func, self.blind_generator, 10000, 10.0)

    def check(self):
        return self.check_validity() and self.check_speed()


class Dataset:
    def __init__(self, name: str, task_ids=None):
        self.name = name
        self.instances = []
        # None -> make() builds every task; otherwise only these HumanEval task ids (the N in "HumanEval/N").
        self.task_ids = set(task_ids) if task_ids is not None else None
        self.data_path = pathlib.Path(__file__).parent / ".data"
        if not (self.data_path.exists() and (self.data_path / "HumanEvalComm.json").exists()):
            raise Exception("The folder .data containing HumanEvalComm.json does not exists !")

    def load(self):
        with open(f"{self.data_path.as_posix()}/{self.name}.pkl", "rb") as f:
            self.instances = cloudpickle.load(f)

    def save(self):

        with open(f"{self.data_path.as_posix()}/{self.name}.pkl", "wb") as f:
            cloudpickle.dump(self.instances, f)

    def make(self):

        with open(f"{self.data_path.as_posix()}/HumanEvalComm.json", "r", encoding="utf-8") as f:
            instance_list = json.load(f)
        if self.task_ids is not None:
            instance_list = [
                record for record in instance_list
                if int(record["name"].split("/")[1]) in self.task_ids
            ]
        instance_list = [Instance(instance) for instance in instance_list]

        self.instances = []
        for inst in instance_list:
            accepted = inst.check()
            print(f"{inst.name} : {accepted}")
            if accepted:
                self.instances.append(inst)
                self.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a HumanEvalComm dataset (pickled to .data/<name>.pkl)."
    )
    parser.add_argument("--name", default="dataset-complete", help="Output dataset name (pickle basename).")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated HumanEval task ids to include, e.g. 0,3,7. Omit to use all tasks.",
    )
    args = parser.parse_args()

    task_ids = None
    if args.task_ids:
        task_ids = [int(t) for t in args.task_ids.split(",") if t.strip()]

    dataset = Dataset(args.name, task_ids=task_ids)
    dataset.make()
