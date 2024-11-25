from enum import Enum


class SystemResponses:
    fallback = [
        "Sorry, I'm not sure about that.",
    ]
    already_in_first_step = [
        "We are already in the first step.",
    ]
    focus_on_task = [
        "Let's focus on the task for now!",
    ]
    curiosity = [
        "Sure, here's a fun fact: The National cake day is celebrated on November 26th.",
    ]
    step_does_not_exist = [
        "This step does not exist.",
    ]
    middle_of_task = [
        "We are in the middle of task right now.",
    ]
    no_more_details = [
        "This step does not have more details.",
    ]
    end_of_task = [
        "We have reached the end of the task.",
    ]
    does_not_play_music = [
        "This skill does not play music.",
    ]
    question_placeholder = [
        "This is a question.",
    ]
    no_answer_to_question = [
        "Sorry I don't know the answer to that question.",
    ]
    goodbye = [
        "Glad I could help you! See you again soon!",
    ]
    ingredient_replacement_response = [
        "No problem, you can also try {ingredient}.",
    ]


class ResponseType(Enum):

    NEUTRAL = "neutral"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, ResponseType):
            return self.value == other.value
        else:
            return False
