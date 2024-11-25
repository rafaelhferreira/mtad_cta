import random
from typing import List, Dict

from data_binding.step import Step


####################################################################################################
# Next Step Utils
####################################################################################################

INITIAL_STEP_ENVELOPES = [
    "Ready to dive in? ",
    "Ready, set, go! ",
    "All set? Here's ",
    "Excited to start? ",
    "Buckle up! ",
    "On your marks, get set, go! ",
]

MIDDLE_STEP_ENVELOPES = [
    "Moving right along, ",
    "Making progress! ",
    "Let's move on to the next step, ",
    "Now for the exciting part, ",
    "We're halfway there! ",
    "Ready for the next step? ",
    "Are you still with me? ",
    "Let's keep the momentum going, ",
    "Great job so far! ",
]

FINAL_STEP_ENVELOPES = [
    "The final touch! ",
    "Almost there! ",
    "And now, for the grand finale, ",
    "The moment you've been waiting for! ",
    "The last step is here! ",
    "You're in the home stretch! ",
    "The finishing flourish! ",
    "Let's bring it all together with ",
    "We've reached the end! ",
]


def add_next_step_envelope(step_text: str, step_number: int, total_steps: int, previous_turns: List[Dict] = None):
    envelope = ""
    if step_number == 0:
        envelope = random.choice(INITIAL_STEP_ENVELOPES)
    elif step_number == total_steps:
        envelope = random.choice(FINAL_STEP_ENVELOPES)
    elif random.random() < 0.25:
        if previous_turns:
            envelope = random.choice([s for s in MIDDLE_STEP_ENVELOPES if s not in " ".join([t['system'] for t in previous_turns])])
        else:
            envelope = random.choice(MIDDLE_STEP_ENVELOPES)
    return envelope + f"Step {step_number+1}: " + step_text


####################################################################################################
# Stop Intent Utils
####################################################################################################

def get_stop_negative_response(steps: List[Step], current_step: int, previous_turn: Dict, user_request: str):
    return previous_turn["system"]
