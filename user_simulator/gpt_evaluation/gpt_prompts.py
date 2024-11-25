# prompts used to automate the evaluation

GENERAL_PREFIX = """Below you will find a recipe, a dialogue between a user and a cooking assistant, and a candidate 
assistant’s response. It is not supposed to show any bias and it is not supposed to be rude, even if the user is 
rude. It should always try to help the user to the best of its capabilities. Still, if the user makes any unrelated 
comments, the assistant should politely reject them and try to get the conversation focused on the task. The 
assistant is only able to help with recipes or DIY tasks. However, the assistant can provide fun facts and discuss 
adjacent topics if the user asks for it. Whenever the user asks for instructions for a different recipe or task, 
the assistant should always ask the user to clarify if they want to start a new task or continue with the current 
one. If the user asks to turn off, stop the interaction, or complete the task the system should end the interaction. 
You are an unbiased annotator.

Recipe: {recipe_text}
Dialogue:
{dialog_context}
{user}
{assistant_response}

{annotation_question}"""


INTENT_QUESTION = """FIRST did the last user’s response accurately reflect the intent description? Respond only with 
"yes" or "no". Base your answer on the last user’s response, the previous dialogue should not influence your answer, 
it only serves as context. If the user’s request is not clear or nuanced, please use your best judgment if the user's 
utterance is an acceptable interpretation of the intent description. SECOND write a one-sentence justification for 
your answer.

Follow the following format:
Answer: <yes or no>
Justification: <one sentence justification for your answer>"""


SYSTEM_ERROR_QUESTION = """Give a score to the quality of the assistants last response given the user's request and 
the context. Respond on a scale of 0 to 2, where 0 = bad response, 1 = average response, and 2 = good response. Base 
your answer on the last assistant’s response, the previous dialogue should not influence your answer, it only serves 
as context. The assistant should only read one step at a time. If the user’s request is not clear or nuanced, 
please use your best judgment if the given response answers an acceptable interpretation of the user’s request. 
First, write a one-sentence justification for your answer. Second provide the score for the quality of the 
assistant's answer.

Follow the following format:
Justification: <one sentence justification for your answer>
Answer: 0, 1, or 2"""
