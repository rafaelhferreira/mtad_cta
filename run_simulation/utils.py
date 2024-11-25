import re

from data_binding.enumerates import Intents
from dataset_generation.system_responses import SystemResponses


def take_intent_from_text(user_text: str, intent_prefix: str):
    # [User Intent: Question] What type of egg should I use?
    try:
        # intent_prefix = re.escape(intent_prefix)  # escape special characters in the prefix to avoid problems
        user_text = user_text.replace(intent_prefix, "")
        # match until the first ] and then get the group inside the brackets
        intent = re.search(fr"(.+?)\]", user_text).group(1)
        # remove the intent from the user text
        user_text = re.sub(fr"{re.escape(intent)}\]", "", user_text)  # escape the intent because of special tokens |
        # remove special tokens
        intent = intent.replace("<|", "").replace("|>", "")
        return user_text, intent
    except Exception as e:  # when the intent is not present
        print("Something went wrong when trying to extract the intent from the user text", user_text, intent_prefix)
        return user_text, ""


def take_system_response_from_model_output(user_text: str, system_separator: str, user_final_token: str):
    if user_text:
        if system_separator:  # remove from system separator to the end
            user_text = user_text.split(system_separator)[0]
        if user_final_token:  # remove from user final user token to the end
            user_text = user_text.split(user_final_token)[0]
    return user_text


def clean_special_tokens_plangpt(system_text: str):
    # remove text between < > can have multiple
    system_text = re.sub(r"<\|[^>]*\|>", "", system_text)

    # get the last word
    last_word = system_text.split()[-1]
    # remove the last word if it is a special token (since sometimes special token is not enclosed in < >)
    if last_word.startswith("<|"):
        system_text = " ".join(system_text.split()[:-1])

    return system_text


def is_system_error(system_text: str):
    if re.findall(r"i (don't|do not) know|i('m|am) (sorry|afraid)", system_text, re.IGNORECASE):
        return True
    return False


def is_ending_conversation(intent: str, user_text: str, system_text: str):

    if intent in {Intents.convert_to_pretty_name_model(Intents.AMAZONStopIntent, False),
                  Intents.convert_to_pretty_name_model(Intents.CompleteTaskIntent, False),
                  Intents.convert_to_pretty_name_model(Intents.TerminateCurrentTaskIntent, False),
                  Intents.convert_to_pretty_name_model(Intents.AMAZONCancelIntent, False)}:
        return True

    if system_text in SystemResponses.goodbye:
        return True

    if re.findall(r"(see you again|see you soon|have a wonderful day|see you in the future)", user_text, re.IGNORECASE):
        return True

    if re.findall(r"(^end|^off|goodbye|turn off|stop|stop it|shut up|shut down|be quiet|go to sleep)(conversation|chat|talking|this)?$", user_text, re.IGNORECASE):
        return True

    return False


def get_step_number_from_system_response(system_text: str):
    # in format "Step 1: ..."
    step_number = re.search(r"Step (\d+):", system_text)
    if step_number:
        return int(step_number.group(1))
    return -1
