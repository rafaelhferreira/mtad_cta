
class Intents:
    AddToShoppingListIntent = 'AddToShoppingListIntent'
    AMAZONCancelIntent = 'AMAZON.CancelIntent'
    AMAZONPauseIntent = 'AMAZON.PauseIntent'
    AMAZONPreviousIntent = 'AMAZON.PreviousIntent'
    AMAZONRepeatIntent = 'AMAZON.RepeatIntent'
    AMAZONNoIntent = 'AMAZON.NoIntent'
    AMAZONYesIntent = 'AMAZON.YesIntent'
    GoToStepIntent = 'GoToStepIntent'
    GreetingIntent = 'GreetingIntent'
    IngredientsConfirmationIntent = 'IngredientsConfirmationIntent'
    NoneOfTheseIntent = 'NoneOfTheseIntent'
    PreviousStepIntent = 'PreviousStepIntent'
    ProvideUserNameIntent = 'ProvideUserNameIntent'
    RequestAnonimityIntent = 'RequestAnonimityIntent'
    ResumeTaskIntent = 'ResumeTaskIntent'
    SetTimerIntent = 'SetTimerIntent'
    StartCookingIntent = 'StartCookingIntent'
    StartStepsIntent = 'StartStepsIntent'
    TerminateCurrentTaskIntent = 'TerminateCurrentTaskIntent'
    IdentifyRestrictionsIntent = 'IdentifyRestrictionsIntent'
    IngredientsReplacementIntent = 'IngredientsReplacementIntent'
    QuestionIntent = 'QuestionIntent'
    IdentifyProcessIntent = 'IdentifyProcessIntent'
    MoreDetailIntent = 'MoreDetailIntent'
    AMAZONNextIntent = 'AMAZON.NextIntent'
    NextStepIntent = 'NextStepIntent'
    AMAZONFallbackIntent = 'AMAZON.FallbackIntent'
    AMAZONHelpIntent = 'AMAZON.HelpIntent'
    AMAZONStopIntent = 'AMAZON.StopIntent'
    CompleteTaskIntent = 'CompleteTaskIntent'
    CommonChitChatIntent = 'CommonChitChatIntent'
    AMAZONSelectIntent = 'AMAZON.SelectIntent'
    GetCuriositiesIntent = 'GetCuriositiesIntent'
    LaunchRequestIntent = 'LaunchRequestIntent'
    NumberIntent = 'NumberIntent'
    PlayMusicIntent = 'PlayMusicIntent'
    UserEvent = 'UserEvent'

    pretty_names = {
        "IdentifyProcessIntent": "search",
        "NoneOfTheseIntent": "none of these",
        "AMAZON.CancelIntent": "cancel",
        "AMAZON.YesIntent": "yes",
        "AMAZON.NoIntent": "no",
        "IngredientsConfirmationIntent": "ingredients",
        "StartCookingIntent": "start",
        "StartStepsIntent": "start",
        "AMAZON.NextIntent": "next",
        "NextStepIntent": "next",
        "MoreDetailIntent": "detail",
        "TerminateCurrentTaskIntent": "terminate",
        "AMAZON.HelpIntent": "help",
        "AMAZON.RepeatIntent": "repeat",
        "AMAZON.FallbackIntent": "fallback",
        "AMAZON.PreviousIntent": "previous",  # this is the same as previous
        "PreviousStepIntent": "previous",
        "CompleteTaskIntent": "stop",  # complete is now the same as stop
        "AMAZON.StopIntent": "stop",
        "AMAZON.PauseIntent": "pause",
        "ChitChatIntent": "chit-chat",
        "CommonChitChatIntent": "chit-chat",
        "AMAZON.SelectIntent": "select",
        "GetCuriositiesIntent": "curiosities",
        "LaunchRequestIntent": "launch",
        "NumberIntent": "number",
        "PlayMusicIntent": "music",
        "UserEvent": "user",
        "QuestionIntent": "question",
        "ResumeTaskIntent": "resume",
        "IngredientsReplacementIntent": "replacement",
        "GoToStepIntent": "go to step",
        # this one should not be here but for ease of use it is
        "ARTIFICIAL.DefinitionQuestionIntent": "definition",
        "ARTIFICIAL.Definition": "ARTIFICIAL.DefinitionQuestionIntent",
        "ARTIFICIAL.SensitiveIntent": "sensitive",
        "ARTIFICIAL.Sensitive": "ARTIFICIAL.SensitiveIntent",
    }

    # invert the dictionary and keep both
    inverted_pretty_names = {v: k for k, v in pretty_names.items()}
    inverted_pretty_names.update({k: k for k in pretty_names.keys()})

    @staticmethod
    def is_back_transition_intent(intent: str):
        return intent in {Intents.AMAZONPreviousIntent, Intents.PreviousStepIntent, Intents.AMAZONCancelIntent}

    @staticmethod
    def non_explorative_intents() -> set:
        # basically these are the intents we do not want to increase the probability of when considering exploration
        return {Intents.AMAZONCancelIntent, Intents.TerminateCurrentTaskIntent,
                Intents.AMAZONFallbackIntent, Intents.AMAZONStopIntent,
                Intents.CompleteTaskIntent, Intents.LaunchRequestIntent,
                Intents.PlayMusicIntent, ArtificialIntents.SensitiveIntent, Intents.CommonChitChatIntent,
                Intents.IdentifyProcessIntent,  # here because we are only considering steps for now
                Intents.PreviousStepIntent, Intents.AMAZONPreviousIntent,  # exploration is only moving forward in task
                Intents.ResumeTaskIntent, Intents.AMAZONRepeatIntent,
                }

    @staticmethod
    def non_cooperative_intents() -> set:
        return {Intents.AMAZONFallbackIntent,
                Intents.PlayMusicIntent,
                Intents.IdentifyProcessIntent,  # here because we are only considering steps for now
                Intents.CommonChitChatIntent,
                ArtificialIntents.SensitiveIntent,
                }

    @staticmethod
    def stop_task_intents() -> set:
        # these are the intents that stop the current interaction
        return {Intents.AMAZONCancelIntent, Intents.TerminateCurrentTaskIntent, Intents.AMAZONStopIntent,
                Intents.CompleteTaskIntent}

    @staticmethod
    def impatience_escalation_intents() -> set:
        # these are the intents that escalate when impatience increases
        return {Intents.AMAZONStopIntent, Intents.CompleteTaskIntent, Intents.IdentifyProcessIntent,
                Intents.TerminateCurrentTaskIntent}

    @staticmethod
    def intolerance_escalation_intents() -> set:
        # these are the intents that escalate when intolerance increases
        return {Intents.AMAZONStopIntent, Intents.CompleteTaskIntent, Intents.TerminateCurrentTaskIntent}

    @staticmethod
    def convert_to_pretty_name(intent: str, add_intent_prefix: bool):
        pretty_names = {
            "IdentifyProcessIntent": "Search",
            "NoneOfTheseIntent": "None of These",
            "AMAZON.CancelIntent": "Cancel",
            "AMAZON.YesIntent": "Yes",
            "AMAZON.NoIntent": "No",
            "IngredientsConfirmationIntent": "Ingredients",
            "StartCookingIntent": "Start Cooking",
            "StartStepsIntent": "Start Steps",
            "AMAZON.NextIntent": "Next",
            "NextStepIntent": "Next Step",
            "MoreDetailIntent": "More Detail",
            "TerminateCurrentTaskIntent": "Terminate Task",
            "AMAZON.HelpIntent": "Help",
            "AMAZON.RepeatIntent": "Repeat",
            "AMAZON.FallbackIntent": "Fallback",
            "PreviousStepIntent": "Previous Step",
            "AMAZON.PreviousIntent": "Previous",
            "AMAZON.StopIntent": "Stop",
            "AMAZON.PauseIntent": "Pause",
            "CompleteTaskIntent": "Complete",
            "CommonChitChatIntent": "Chit-Chat",
            "AMAZON.SelectIntent": "Select",
            "GetCuriositiesIntent": "Get Curiosities",
            "LaunchRequestIntent": "Launch Request",
            "NumberIntent": "Number",
            "PlayMusicIntent": "Play Music",
            "UserEvent": "User Event",
            "QuestionIntent": "Question",
            "ResumeTaskIntent": "Resume",
            "IngredientsReplacementIntent": "Ingredient Replacement",
            "GoToStepIntent": "Go To Step",
            # this one should not be here but for ease of use it is
            "ARTIFICIAL.DefinitionQuestionIntent": "Definition Question",
            "ARTIFICIAL.SensitiveIntent": "Sensitive",
        }

        intent = pretty_names.get(intent, intent)
        intent = intent.replace("Intent", "")
        intent = intent.replace("Amazon.", "")

        if add_intent_prefix:
            intent = "Intent: " + intent

        return intent

    @staticmethod
    def convert_to_pretty_name_model(intent: str, add_intent_prefix: bool):
        # does basically the same but tries to avoid having more than one word per intent
        intent = Intents.pretty_names.get(intent, intent)
        intent = intent.replace("Intent", "")
        intent = intent.replace("Amazon.", "")

        if add_intent_prefix:
            intent = "Intent: " + intent

        return intent

    @staticmethod
    def from_pretty_name_to_original(intent: str):
        # this in fact also converts from pretty name to original name
        # but also from original name to the same
        return Intents.inverted_pretty_names.get(intent, None)

    @staticmethod
    def get_intent_description(intent: str):
        intent_descriptions = {
            Intents.AddToShoppingListIntent: 'Add an item to the shopping list.',
            Intents.AMAZONCancelIntent: 'Cancel the current task.',
            Intents.AMAZONPauseIntent: 'Pause the current task.',
            Intents.AMAZONPreviousIntent: 'Go back to the previous step.',
            Intents.AMAZONRepeatIntent: 'Ask to repeat the last system utterance.',
            Intents.AMAZONNoIntent: 'Negative response.',
            Intents.AMAZONYesIntent: 'Affirmative response.',
            Intents.GoToStepIntent: 'Navigate to a specific step in the task.',
            Intents.GreetingIntent: 'User greets the system.',
            Intents.IngredientsConfirmationIntent: 'Ask for the ingredients.',
            Intents.NoneOfTheseIntent: 'Indicate none of the provided options are suitable.',
            Intents.PreviousStepIntent: 'Go back to the previous step.',
            Intents.ProvideUserNameIntent: 'Provide name.',
            Intents.RequestAnonimityIntent: 'Request anonymity or privacy.',
            Intents.ResumeTaskIntent: 'Resume the current task.',
            Intents.SetTimerIntent: 'Set a timer for a specific duration.',
            Intents.StartCookingIntent: 'Start the task.',
            Intents.StartStepsIntent: 'Start the task.',
            Intents.TerminateCurrentTaskIntent: 'Terminate the current task.',
            Intents.IdentifyRestrictionsIntent: 'Mention dietary or other restrictions.',
            Intents.IngredientsReplacementIntent: 'Ask for a replacement for an ingredient or tool in the task.',
            Intents.QuestionIntent: 'Ask a question or seek extra information. '
                                    'It is not an ingredient or tool replacement.',
            Intents.IdentifyProcessIntent: 'Search for a different task.',
            Intents.MoreDetailIntent: 'Request more detailed information.',
            Intents.AMAZONNextIntent: 'Proceed to the next step of the task.',
            Intents.NextStepIntent: 'Proceed to the next step of the task.',
            Intents.AMAZONFallbackIntent: 'Mentions something random or unsupported.',
            Intents.AMAZONHelpIntent: 'Requests help or assistance.',
            Intents.AMAZONStopIntent: 'Stops the current interaction.',
            Intents.CompleteTaskIntent: 'Completes a task.',
            Intents.CommonChitChatIntent: 'Engage in common chit-chat or casual conversation.',
            Intents.AMAZONSelectIntent: 'Select or choose an option from a list.',
            Intents.GetCuriositiesIntent: 'Ask for a fun fact.',
            Intents.LaunchRequestIntent: 'Start the interaction.',
            Intents.NumberIntent: 'Mention a number.',
            Intents.PlayMusicIntent: 'Ask to play music.',
            Intents.UserEvent: 'Clicks on the screen.',
            ArtificialIntents.SensitiveIntent: 'Provides a dangerous or sensitive request.',
            ArtificialIntents.DefinitionQuestionIntent: 'Ask for a definition or explanation of a concept.'
        }

        return intent_descriptions.get(intent, None)


class ArtificialIntents:
    # intents created specifically for the dataset generation
    # they are not part of the actual skill
    DefinitionQuestionIntent = 'ARTIFICIAL.DefinitionQuestionIntent'
    SensitiveIntent = 'ARTIFICIAL.SensitiveIntent'


class DataSource:
    WIKIHOW = 'wikihow'
    RECIPE = 'recipe'
    MULTISOURCE = 'multi_source'

    datasource_to_int = {
        RECIPE: 0,
        WIKIHOW: 1,
        None: 2,
        MULTISOURCE: 3
    }

    int_to_datasource = {
        0: RECIPE,
        1: WIKIHOW,
        2: None,
        3: MULTISOURCE
    }


class Speaker:
    user = "user"
    system = "system"
