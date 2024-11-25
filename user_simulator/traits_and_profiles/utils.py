from typing import Dict, Union, Set, Iterable


def get_by_top_k(top_k: int, counts_dict: Dict[str, Union[int, float]]) -> Dict[str, int]:
    sorted_utterances = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
    return {utterance[0]: utterance[1] for utterance in sorted_utterances[:top_k]}


def get_by_top_p(top_p: float, counts_dict: Dict[str, Union[int, float]]) -> Dict[str, int]:
    sorted_utterances = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
    total_counts = sum([utterance[1] for utterance in sorted_utterances])
    current_sum = 0
    for i, utterance in enumerate(sorted_utterances):
        current_sum += utterance[1] / total_counts
        if current_sum >= top_p:
            return {utterance[0]: utterance[1] for utterance in sorted_utterances[:i + 1]}
    return {utterance[0]: utterance[1] for utterance in sorted_utterances}


def get_values_in_percentage_range(data: Dict[str, Union[int, float]], bottom_percentage: float, top_percentage: float):
    # order the dict by value
    ordered_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse=False)}

    # get the bottom and top values
    bottom_index = int(len(ordered_data) * bottom_percentage)
    top_index = int(len(ordered_data) * top_percentage)

    # get the values between the bottom and top
    result = {}
    for i, (key, value) in enumerate(ordered_data.items()):
        if i >= bottom_index and (i < top_index or top_percentage == 1.0):
            result[key] = value

    if len(result) == 0:
        if bottom_percentage == 0.0:
            # get the fist one
            result = {list(ordered_data.keys())[0]: list(ordered_data.values())[0]}
        elif top_percentage == 1.0:
            # get the last one
            result = {list(ordered_data.keys())[-1]: list(ordered_data.values())[-1]}

    if len(result) == 0:
        raise Exception("Could not find any value in the percentage range")

    return result


def get_list_in_percentage_range(data: Iterable[Union[int, float]],
                                 bottom_percentage: float, top_percentage: float) -> Set[Union[int, float]]:
    # order data
    ordered_data = sorted(data, reverse=False)

    # get the bottom and top values
    bottom_index = int(len(ordered_data) * bottom_percentage)
    top_index = int(len(ordered_data) * top_percentage)

    # get the values between the bottom and top
    result = set()
    for i, value in enumerate(ordered_data):
        if i >= bottom_index and (i < top_index or top_percentage == 1.0):
            result.add(value)

    if len(result) == 0:
        if bottom_percentage == 0.0:
            # get the fist one
            result = {ordered_data[0]}

        elif top_percentage == 1.0:
            # get the last one
            result = {ordered_data[-1]}

    if len(result) == 0:
        raise Exception("Could not find any value in the percentage range")

    return result


# because of compatibility issue we set the stopwords here:
nltk_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                  'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
                  'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                  'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                  'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                  'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
                  'for',
                  'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'to',
                  'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                  'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                  'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                  's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
                  're', 've', 'y',
                  'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
                  'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                  "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                  'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
