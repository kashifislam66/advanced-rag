from typing import TypeVar

T = TypeVar("T")


def order_for_context(items_best_first: list[T]) -> list[T]:
    """Place the best items at the start and end, the weakest in the middle."""
    front: list[T] = []
    back: list[T] = []
    for index, item in enumerate(items_best_first):
        if index % 2 == 0:
            front.append(item)
        else:
            back.append(item)
    return front + back[::-1]