import uuid
from typing import Dict, List, Union, Tuple

from data_binding.enumerates import Speaker, Intents
from data_binding.task_result import TaskResult
from dataset_generation.create_system_response import generate_system_response
from dataset_generation.system_responses import SystemResponses
from dataset_generation.utils import random_choices_from_dict, lowercase_and_remove_punctuation_user_utterance


class DialogTurn:

    def __init__(self, intent: str, user_utterance: str, system_utterance: str, current_step: int,
                 negative_response: str = "", forced_system_error: bool = False):
        self.intent = intent
        self.user_utterance = user_utterance
        self.system_utterance = system_utterance
        self.current_step = current_step
        self.negative_response = negative_response
        self.forced_system_error = forced_system_error
        self.generated_text = None  # only used for generating text in eval

    def dialog_turn_dict(self) -> Dict[str, str]:
        dialog_dict = {
            "intent": self.intent,
            "current_step": self.current_step,
            Speaker.user: self.user_utterance,
            Speaker.system: self.system_utterance,
            "negative_response": self.negative_response,
            "forced_system_error": self.forced_system_error
        }
        if self.generated_text:
            dialog_dict["generated_text"] = self.generated_text
        return dialog_dict

    def __repr__(self):
        return str(self.dialog_turn_dict())

    def apply_forced_system_error(self) -> bool:
        # changes current system utterance to negative response to simulate a system error
        # (only useful when training user simulator)
        if self.negative_response and self.system_utterance != self.negative_response:
            self.system_utterance = self.negative_response
            self.forced_system_error = True
            return True
        return False


class Dialog:

    def __init__(self, task: TaskResult, dialog_id: str = None, system_tone: str = None,
                 user_profile=None, lower_case_and_remove_punctuation_user: bool = False,
                 simulator_model_path: str = None):

        from user_simulator.traits_and_profiles.user_profile import UserProfile

        self.task = task
        self.user_profile = user_profile  # type: Union[None, UserProfile]
        if not dialog_id:
            self.dialog_id = str(uuid.uuid4())
        else:
            self.dialog_id = dialog_id
        self.turns: List[DialogTurn] = []
        self.current_step = 0
        self.current_method = 0
        self.system_tone = system_tone
        self.lower_case_and_remove_punctuation_user = lower_case_and_remove_punctuation_user
        self.simulator_model_path = simulator_model_path

    def add_turn(self, intent: Union[None, str], user_utterance: Union[None, str], system_utterance: str,
                 current_step: int, negative_response: str = "",
                 forced_system_error: bool = False, lowercase_and_remove_punctuation_user: bool = False):

        if lowercase_and_remove_punctuation_user:
            user_utterance = lowercase_and_remove_punctuation_user_utterance(user_utterance)

        turn = DialogTurn(intent=intent, user_utterance=user_utterance, system_utterance=system_utterance,
                          current_step=current_step, negative_response=negative_response,
                          forced_system_error=forced_system_error)
        self.turns.append(turn)

    def create_and_add_turn_from_prob_dict(self, probs_dict: Dict[str, float],
                                           collected_utterances: Dict[str, Dict[str, int]],
                                           use_weight_for_utterance: bool,
                                           ) -> Tuple[str, str, str, bool]:

        utterance = None
        system_response = None
        intent = None
        end_conversation = False

        negative_response = ""
        while not utterance or not system_response:
            intent = random_choices_from_dict(probs_dict, use_weights=True)[0]  # type: str

            # if intent is to stop but we have not reached the minimum number of turns, then we skip and try again
            if intent in Intents.stop_task_intents() and (self.user_profile and self.number_turns() < self.user_profile.min_number_turns):
                continue

            # special case to avoid repeating previous step in first step
            elif intent in [Intents.PreviousStepIntent, Intents.AMAZONPreviousIntent] and self.turns and \
                    self.turns[-1].system_utterance in SystemResponses.already_in_first_step:
                pass  # while will take care of getting another intent
            else:
                if intent == Intents.AMAZONYesIntent:
                    intent = Intents.NextStepIntent

                utterance = self.__get_user_utterance__(
                    intent=intent, collected_utterances=collected_utterances,
                    use_weight_for_utterance=use_weight_for_utterance
                )

                # generate the system response
                system_response, negative_response, current_step, end_conversation = generate_system_response(
                    current_turns=self.turns_dict(),
                    task=self.task,
                    current_step=self.current_step,
                    current_method=self.current_method,
                    intent=intent,
                    user_text=utterance,
                    system_tone=self.system_tone
                )
                self.current_step = current_step if current_step is not None else self.current_step

                if system_response == negative_response:
                    negative_response = ""

                if "stop" in utterance.lower():
                    end_conversation = True

        self.add_turn(intent=intent, user_utterance=utterance, system_utterance=system_response,
                      current_step=self.current_step, negative_response=negative_response,
                      lowercase_and_remove_punctuation_user=self.lower_case_and_remove_punctuation_user)
        return intent, utterance, system_response, end_conversation

    def dialog_dict(self):
        return {
            "dialog_id": self.dialog_id,
            "task": self.task.get_result_dict(),
            "dialog": self.turns_dict(),
            "system_tone": self.system_tone,
            "user_profile": self.user_profile.to_dict() if self.user_profile else None
        }

    def turns_dict(self):
        return [turn.dialog_turn_dict() for turn in self.turns]

    def number_turns(self) -> int:
        return len(self.turns)

    def __repr__(self):
        return str(self.dialog_dict())

    def __get_user_utterance__(self, intent: str, collected_utterances: Dict[str, Dict[str, int]],
                               use_weight_for_utterance: bool) -> str:

        if self.user_profile:
            updated_distribution = self.user_profile.apply_utterance_traits(
                current_dialog=self, current_intent=intent,
                probs_dict=collected_utterances[intent]
            )
            utterance = random_choices_from_dict(updated_distribution, use_weights=use_weight_for_utterance)[0]
        else:
            utterance = random_choices_from_dict(collected_utterances[intent], use_weights=use_weight_for_utterance)[0]

        return utterance
