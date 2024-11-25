class ModelMergingTypes:

    # decoding time
    mtad = "mtad"  # multi trait adaptive decoding
    sampling = "sampling"  # sampling based on weights of loras
    mtad_level_aware = "mtad_level_aware"  # mtad-la that generates intent first with dialogue level profiles and remaining with utterance level profiles (allows to have 100% explorative and 100% concise at the same time)

    # weights merging
    # https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraModel.add_weighted_adapter
    linear = "linear"
    ties = "ties"
    ties_svd = "ties_svd"
    dare_ties = "dare_ties"
    dare_linear = "dare_linear"
    dare_ties_svd = "dare_ties_svd"
    dare_linear_svd = "dare_linear_svd"

    @staticmethod
    def is_decoding_time(method: str):
        return method in {ModelMergingTypes.mtad, ModelMergingTypes.sampling,
                          ModelMergingTypes.mtad_level_aware}

    @staticmethod
    def is_weights_merging(method: str):
        return method in {ModelMergingTypes.linear, ModelMergingTypes.ties, ModelMergingTypes.ties_svd,
                          ModelMergingTypes.dare_ties, ModelMergingTypes.dare_linear, ModelMergingTypes.dare_ties_svd,
                          ModelMergingTypes.dare_linear_svd}

    @staticmethod
    def merging_method_to_pretty_name(method: str):
        merging_method_to_pretty_name = {
            # decoding time
            ModelMergingTypes.mtad: "mTAD",
            ModelMergingTypes.sampling: "Sampling",
            ModelMergingTypes.mtad_level_aware: "mTAD Level Aware",

            # weights
            ModelMergingTypes.linear: "Linear",
            ModelMergingTypes.ties: "Ties",
            ModelMergingTypes.ties_svd: "Ties SVD",
            ModelMergingTypes.dare_ties: "DARE Ties",
            ModelMergingTypes.dare_linear: "DARE Linear",
            ModelMergingTypes.dare_ties_svd: "DARE Ties SVD",
            ModelMergingTypes.dare_linear_svd: "DARE Linear SVD",
        }
        return merging_method_to_pretty_name.get(method, method)
