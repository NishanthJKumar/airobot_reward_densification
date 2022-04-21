from typing import Dict, Tuple, Callable, List

class BaseClassifiers:
    def __init__(self):
        pass

    def get_typed_predicates(self) -> Dict[str, List[Tuple[Callable, ...]]]:
        raise NotImplementedError("Override me!")

    def get_path_to_domain_and_problem_files(self) -> Tuple[str, str]:
        raise NotImplementedError("Override me!")
