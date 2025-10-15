import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    user_input = state['graph_state']

    if random.random() < 0.5:
        return "node_2"

    return "node_3"