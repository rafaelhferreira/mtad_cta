import random
from typing import List, Dict, Tuple

from data_binding.enumerates import Intents
from data_binding.task_result import TaskResult
from dataset_generation import response_building_utils
from dataset_generation.system_responses import SystemResponses


def generate_system_response(current_turns: List[Dict[str, str]], task: TaskResult,
                             current_step: int, current_method: int,
                             intent: str, user_text: str, system_tone: str) -> Tuple[str, str, int, bool]:

    sys_resp = SystemResponses()

    # for each intent create an appropriate system response
    end_conversation = False
    # Fallback intent logic
    if intent == Intents.AMAZONFallbackIntent:
        return random.choice(sys_resp.fallback), "", current_step, end_conversation
    # Stop or Complete Task intent logic
    elif intent in Intents.stop_task_intents():
        end_conversation = True
        negative_response = response_building_utils.get_stop_negative_response(task.get_methods()[current_method].get_steps(), current_step, current_turns[-1], user_text)
        return random.choice(sys_resp.goodbye), negative_response, current_step, end_conversation
    else:
        return None, None, None, None
