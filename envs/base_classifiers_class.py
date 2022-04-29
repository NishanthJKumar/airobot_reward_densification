from typing import Dict, Tuple, Callable, List

class BaseClassifiers:
    def __init__(self):
        pass

    def get_typed_predicates(self) -> Dict[str, List[Tuple[Callable, ...]]]:
        """Return a dictionary whose keys are '0-arity', '1-arity', and so
        on until the max. arity. The values should be a list of tuples, where
        the first element is the predicate classifier function, and the 
        following elements are the names of the types (as defined in the 
        particular pddl domain file associated with this python file) that 
        the arguments of this predicate function take."""
        raise NotImplementedError("Override me!")

    def get_path_to_domain_and_problem_files(self) -> Tuple[str, str]:
        """Return a tuple of strings containing the paths to the domain
        and problem files."""
        raise NotImplementedError("Override me!")

    def get_success(self, env, state) -> bool:
        """Given the environment and a state, return a boolean indicating
        whether the state is at the goal."""
        raise NotImplementedError("Override me!")