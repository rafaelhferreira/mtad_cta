from transformers import PreTrainedTokenizer

from dataset_generation.dialog import Dialog


PLANGPT_PROMPT = ("""<|prompter|> You are a taskbot tasked with helping users cook recipes or DIY projects. I will 
give you a recipe and I want you to help me do it step by step. You should always be empathetic, honest, and should 
always help me. If I ask you something that does not relate to the recipe you should politely reject the request and 
try too get me focused on the recipe. I am unsure how to cook something or do something related to the recipe you 
should help me to the best of your ability. Please use a {system_tone} tone of voice. Recipe: {title} Steps: {steps} 
<|endofturn|> <|prompter|> {current_step}. <|endofturn|> <|assistant|> ok! <|endofturn|> {dialog_history} 
<|prompter|> {request} <|endofturn|> <|assistant|> """)

DIALOG_HISTORY_TEMPLATE = ("""<|prompter|> {previous_request} <|endofturn|> <|assistant|> {previous_response} 
<|endofturn|>""")

HASNT_STARTED = "I haven't started cooking yet."
IS_DOING_STEP = "I am currently on Step {step_number}: {step_text}"


def build_raw_samples_plangpt(dialog: Dialog, tokenizer: PreTrainedTokenizer, **kwargs):

    raw_sources = []
    raw_targets = []
    intents = []
    dialog_ids = []
    turn_numbers = []

    prompt_template = PLANGPT_PROMPT
    dialog_history_template = DIALOG_HISTORY_TEMPLATE
    hasn_started = HASNT_STARTED
    is_doing_step = IS_DOING_STEP

    add_intent = kwargs.get("intent_prediction", False)

    target_template = "{system_request} <|endofturn|> {eos_token}"

    context_size = kwargs.get("context_size", 1)

    title = dialog.task.get_title()
    steps = dialog.task.get_methods()[0].get_steps()

    steps_list_text = "\n".join([f"Step {step_number + 1}: {step.get_text()}" for step_number, step in enumerate(steps)])
    # get turns
    turns = dialog.turns

    current_step_number = 0
    for turn_number, turn in enumerate(dialog.turns):

        user_request = turn.user_utterance
        system_request = turn.system_utterance

        dialog_history_text = ""
        if current_step_number == 0 and turn_number == 0:
            current_step_text = hasn_started
        else:
            current_step_text = is_doing_step.format(step_number=current_step_number + 1,
                                                     step_text=steps[current_step_number].get_text())
            for i in range(min([context_size, turn_number])):
                intent_text = ""
                if add_intent:
                    intent_text = f" <intent> {turns[turn_number - (i + 1)].intent} <endofintent> "
                dialog_history_text = dialog_history_template.format(
                    previous_request=turns[turn_number - (i + 1)].user_utterance,
                    previous_response=intent_text + turns[turn_number - (i + 1)].system_utterance
                ) + dialog_history_text

        prompt = prompt_template.format(
            system_tone=dialog.system_tone.replace("_", " "),
            title=title,
            steps=steps_list_text,
            current_step=current_step_text,
            dialog_history=dialog_history_text,
            request=user_request
        ).replace("..", ".").replace("  ", " ")

        prompt = prompt.replace("\n", " ")

        # if turn_dict["intent"] in ["ask_question_ingredients_tools", "ask_question_recipe_steps"]:
        raw_sources.append(prompt.strip())
        target = target_template.format(system_request=system_request, eos_token=tokenizer.eos_token)
        if add_intent:
            target = f"<intent> {turn.intent} <endofintent> {target}"
        raw_targets.append(target.replace("  ", " ").strip())
        intents.append(turn.intent)
        dialog_ids.append(dialog.dialog_id)
        turn_numbers.append(turn_number)

        if turn.current_step < len(steps):
            current_step_number = turn.current_step

    return raw_sources, raw_targets, intents, dialog_ids, turn_numbers
