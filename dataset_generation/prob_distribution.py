# Manual probabilities (they do not need to sum to 1 because random.choices will normalize them as weights)
MANUAL_PROBS = {
    "NextStepIntent": {
        "NextStepIntent": 0.4,
        "AMAZON.YesIntent": 0.1,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
        "IdentifyProcessIntent": 0.05,
        "AMAZON.RepeatIntent": 0.1,
        "CommonChitChatIntent": 0.05,
        "QuestionIntent": 0.2,
        "GetCuriositiesIntent": 0.1,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.1,
        "IngredientsReplacementIntent": 0.12,
        "MoreDetailIntent": 0.05,
        "ResumeTaskIntent": 0.05,
        "GoToStepIntent": 0.1,
        "AMAZON.PreviousIntent": 0.05,
        "CompleteTaskIntent": 0.1,
        "PreviousStepIntent": 0.05
    },
    "AMAZON.YesIntent": {
        "NextStepIntent": 0.4,
        "AMAZON.YesIntent": 0.1,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
        "IdentifyProcessIntent": 0.05,
        "AMAZON.RepeatIntent": 0.1,
        "QuestionIntent": 0.2,
        "GetCuriositiesIntent": 0.1,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.1,
        "IngredientsReplacementIntent": 0.12,
        "CommonChitChatIntent": 0.05
    },
    "AMAZON.FallbackIntent": {
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "NextStepIntent": 0.3,
        "AMAZON.StopIntent": 0.05,
        "AMAZON.YesIntent": 0.1,
        "IdentifyProcessIntent": 0.05,
        "CommonChitChatIntent": 0.05,
        "QuestionIntent": 0.2,
        "GetCuriositiesIntent": 0.1,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.1,
        "IngredientsReplacementIntent": 0.12,
        "PreviousStepIntent": 0.05,
        "AMAZON.RepeatIntent": 0.1,
        "ResumeTaskIntent": 0.05,
        "AMAZON.PreviousIntent": 0.05,
        "GoToStepIntent": 0.1
    },
    "ARTIFICIAL.SensitiveIntent": {
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "NextStepIntent": 0.3,
        "AMAZON.StopIntent": 0.05,
        "AMAZON.YesIntent": 0.1,
        "IdentifyProcessIntent": 0.05,
        "CommonChitChatIntent": 0.05,
        "QuestionIntent": 0.2,
        "GetCuriositiesIntent": 0.1,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.1,
        "IngredientsReplacementIntent": 0.12,
        "PreviousStepIntent": 0.05,
        "AMAZON.RepeatIntent": 0.1,
        "ResumeTaskIntent": 0.05,
        "AMAZON.PreviousIntent": 0.05,
        "GoToStepIntent": 0.1
    },
    "IdentifyProcessIntent": {
        "AMAZON.YesIntent": 0.2,
        "IdentifyProcessIntent": 0.05,
        "NextStepIntent": 0.2,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
        "ResumeTaskIntent": 0.05,
        "QuestionIntent": 0.2,
        "GetCuriositiesIntent": 0.1,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.1,
        "IngredientsReplacementIntent": 0.12,
        "CommonChitChatIntent": 0.05
    },
    "ResumeTaskIntent": {
        "NextStepIntent": 0.4,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "ResumeTaskIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
        "AMAZON.YesIntent": 0.01
    },
    "AMAZON.RepeatIntent": {
        "NextStepIntent": 0.4,
        "AMAZON.RepeatIntent": 0.05,
        "AMAZON.FallbackIntent": 0.1,
        "ARTIFICIAL.SensitiveIntent": 0.1,
        "AMAZON.YesIntent": 0.2
    },
    "QuestionIntent": {
        "NextStepIntent": 0.3,
        "QuestionIntent": 0.1,
        "GetCuriositiesIntent": 0.05,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.05,
        "IngredientsReplacementIntent": 0.05,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "ResumeTaskIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
        "IdentifyProcessIntent": 0.05
    },
    "GetCuriositiesIntent": {
        "NextStepIntent": 0.3,
        "QuestionIntent": 0.1,
        "GetCuriositiesIntent": 0.05,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.05,
        "IngredientsReplacementIntent": 0.05,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "ResumeTaskIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
        "IdentifyProcessIntent": 0.05
    },
    "ARTIFICIAL.DefinitionQuestionIntent": {
        "NextStepIntent": 0.3,
        "QuestionIntent": 0.1,
        "GetCuriositiesIntent": 0.05,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.05,
        "IngredientsReplacementIntent": 0.05,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "ResumeTaskIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
        "IdentifyProcessIntent": 0.05
    },
    "CommonChitChatIntent": {
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "NextStepIntent": 0.2,
        "AMAZON.StopIntent": 0.05,
        "CommonChitChatIntent": 0.05,
        "ResumeTaskIntent": 0.1
    },
    "IngredientsReplacementIntent": {
        "NextStepIntent": 0.3,
        "QuestionIntent": 0.1,
        "GetCuriositiesIntent": 0.05,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.05,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "ResumeTaskIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
        "IdentifyProcessIntent": 0.05,
        "AMAZON.RepeatIntent": 0.05
    },
    "GoToStepIntent": {
        "GoToStepIntent": 0.2,
        "NextStepIntent": 0.4,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "QuestionIntent": 0.1,
        "GetCuriositiesIntent": 0.05,
        "ARTIFICIAL.DefinitionQuestionIntent": 0.05,
        "IngredientsReplacementIntent": 0.05,
    },
    "MoreDetailIntent": {
        "IdentifyProcessIntent": 0.05,
        "NextStepIntent": 0.3,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "MoreDetailIntent": 0.05
    },
    "AMAZON.PreviousIntent": {
        "AMAZON.PreviousIntent": 0.3,
        "NextStepIntent": 0.7
    },
    "PreviousStepIntent": {
        "PreviousStepIntent": 0.3,
        "NextStepIntent": 0.7
    }
}


MANUAL_PROBS_REVISED = MANUAL_PROBS.copy()
MANUAL_PROBS_REVISED["IdentifyProcessIntent"] = {
        "AMAZON.YesIntent": 0.4,
        "AMAZON.NoIntent": 0.4,
        "NextStepIntent": 0.15,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "AMAZON.StopIntent": 0.1,
        "ResumeTaskIntent": 0.2,
        "CommonChitChatIntent": 0.1
    }
MANUAL_PROBS_REVISED["AMAZON.NoIntent"] = {
        "ResumeTaskIntent": 0.4,
        "NextStepIntent": 0.2,
        "CommonChitChatIntent": 0.1,
        "AMAZON.FallbackIntent": 0.05,
        "ARTIFICIAL.SensitiveIntent": 0.05,
        "AMAZON.StopIntent": 0.05,
    }
