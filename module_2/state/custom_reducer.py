from typing_extensions import TypedDict
from typing import Annotated
from operator import add

def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists

    Args:
        left (list | None): The first list to combine, or None
        right (list | None): The second list to combine, or None

    Returns:
        list: A new list containing all elements from both lists
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class DefaultState(TypedDict):
    foo: Annotated[list[int], add]
    
class CustomreducerState(TypedDict):
    foo: Annotated[list[int], reduce_list]