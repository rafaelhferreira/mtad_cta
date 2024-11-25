from __future__ import annotations
from typing import Union, List
from enum import Enum
from .method import Method
from .step import Step


class MethodType(Enum):
    RECIPE_COMPONENT = 'Component'
    WIKIHOW_PART = 'Part'
    WIKIHOW_METHOD = 'Method'


class TaskResult:
    _result: dict = None

    _title: str = None
    _rating: float = None  # 0 - 5
    _methods: List[Method] = None
    _thumbnail_img_url: str = None
    _data_source: str = None
    _unique_id: str = None
    _description: str = None

    # Wikihow specific (will be empty for recipes)
    _has_parts: bool = None
    _has_methods: bool = None
    _title_description: str = None
    _first_line_title_description: str = None
    _views: int = None
    _categories: List[str] = None
    _QAs: List[str] = None

    # Recipe specific (will be empty for wikihow)
    _rating_count: int = None
    _difficulty: str = None
    _n_servings: int = None
    _total_time_: str = None
    _total_time_int_: int = None
    _tools: List[str] = None
    _keywords: List[str] = None
    _diets: List[str] = None

    def __init__(self, result: dict) -> None:
        self._result = result

    def get_result_dict(self) -> dict:
        return self._result

    def get_total_time_int(self) -> int:
        return self._total_time_int_

    def _set_total_time_int(self, total_time_int):
        self._total_time_int_ = total_time_int

    def get_data_source(self) -> str:
        return self._data_source

    def _set_data_source(self, data_source: str) -> None:
        self._data_source = data_source

    def get_title(self) -> str:
        return self._title

    def _set_title(self, title: str) -> None:
        self._title = title

    def get_unique_id(self) -> str:
        return self._unique_id

    def _set_unique_id(self, unique_id: str):
        self._unique_id = unique_id

    def get_rating(self) -> float:
        return self._rating

    def _set_rating(self, rating: float) -> None:
        self._rating = rating

    def get_rating_count(self) -> int:
        return self._rating_count

    def _set_rating_count(self, rating_count: int) -> None:
        self._rating_count = rating_count

    def get_methods(self) -> List[Method]:
        return self._methods

    def _set_methods(self, methods: List[Method]) -> None:
        self._methods = methods

    def get_thumbnail_img_url(self) -> str:
        return self._thumbnail_img_url

    def _set_thumbnail_img_url(self, thumbnail_img_url: str) -> None:
        self._thumbnail_img_url = thumbnail_img_url

    def get_has_parts(self) -> bool:
        return self._has_parts

    def _set_has_parts(self, has_parts: bool) -> None:
        self._has_parts = has_parts

    def get_has_methods(self) -> bool:
        return self._has_methods

    def _set_has_methods(self, has_methods: bool) -> None:
        self._has_methods = has_methods

    def get_difficulty(self) -> str:
        return self._difficulty

    def _set_difficulty(self, difficulty: str) -> None:
        self._difficulty = difficulty

    def get_n_servings(self) -> int:
        return self._n_servings

    def _set_n_servings(self, n_servings: int) -> None:
        self._n_servings = n_servings

    def get_total_time(self) -> str:
        return self._total_time_min

    def _set_total_time(self, total_time: str) -> None:
        self._total_time_min = total_time

    def get_tools(self) -> List[str]:
        return self._tools

    def _set_tools(self, tools: List[str]) -> None:
        self._tools = tools

    def get_keywords(self) -> List[str]:
        return self._keywords

    def _set_keywords(self, keywords: List[str]) -> None:
        self._keywords = keywords

    def get_diets(self) -> List[str]:
        return self._diets

    def _set_diets(self, diets: List[str]) -> None:
        self._diets = diets

    def get_description(self) -> str:
        return self._description

    def _set_description(self, description: str) -> None:
        self._description = description

    def get_title_description(self) -> str:
        return self._title_description

    def _set_title_description(self, title_description: str) -> None:
        self._title_description = title_description

    def get_first_line_title_description(self) -> str:
        return self._first_line_title_description

    def _set_first_line_title_description(self, first_line_title_description: str) -> None:
        self._first_line_title_description = first_line_title_description

    def get_number_parts(self) -> Union[int, None]:
        if self.get_has_parts():
            return len(self.get_methods())
        else:
            return None  # if it does not have parts we return None

    def get_total_number_steps(self, current_method: int) -> int:
        if self.get_has_parts():  # if it has parts we count all steps
            return sum(len(method.get_steps()) for method in self.get_methods())
        else:  # we count only the current_method
            return len(self.get_methods()[current_method].get_steps())

    def does_step_exist(self, current_method: int, current_step: int) -> bool:
        try:
            method_steps = self.get_methods()[current_method].get_steps()
            if len(method_steps) <= current_step:
                return False
            return True
        except Exception as es:  # if it gives an exception it is probably because the step does not exist
            print(es)
            return False

    def _set_views(self, views: int) -> None:
        self._views = views

    def get_views(self) -> int:
        return self._views

    def _set_categories_(self, categories: List[str]) -> None:
        self._categories = categories

    def get_categories(self) -> List[str]:
        return self._categories

    def get_qas(self) -> List[str]:
        return self._QAs

    def _set_qas(self, qas: List[str]) -> None:
        self._QAs = qas


class DummyTaskResult(TaskResult):

    def __init__(self, task_dict: dict) -> None:

        super().__init__(task_dict)

        self._set_title(task_dict.get('title', ''))
        self._set_rating(task_dict.get('rating', 0))
        self.__set_methods__(task_dict)
        self._set_thumbnail_img_url(task_dict.get('thumbnail_img_url', ''))
        self._set_data_source(task_dict.get('data_source', ''))
        self._set_unique_id(task_dict.get('id', ''))
        self._set_description(task_dict.get('description', ''))

    def __set_methods__(self, result: dict) -> None:

        steps = []
        # simplest form of steps
        for step_text in result.get("methods"):
            step_obj = Step(text=step_text, detail_text=[], img_url=None, step_duration_seconds=None,
                            step_ingredients=None, step_video_url=None, tips_list=None)
            steps.append(step_obj)

        methods = [Method(name="Method", steps=steps, method_img_url=None, ingredients=None)]

        super()._set_methods(methods)


def get_time_hour_min(total_time_min):
    time_min = total_time_min % 60
    time_hour = int((total_time_min - time_min) / 60) % 24
    time_days = int(int(int((total_time_min - time_min) / 60) - time_hour) / 24)
    plural_hours = "s" if time_hour > 1 else ""
    plural_minutes = "s" if time_min > 1 else ""
    plural_days = "s" if time_days > 1 else ""
    if time_days > 0:
        if time_hour > 0 and time_min > 0:
            time = f"{time_days}day{plural_days} {time_hour} hour{plural_hours} and {time_min} minute{plural_minutes} "
        elif time_hour == 0 and time_min > 0:
            time = f"{time_days}day{plural_days} {time_min} minute{plural_minutes} "
        else:
            time = f"{time_days}day{plural_days} {time_hour} hour{plural_hours} "
    else:
        if time_hour > 0 and time_min > 0:
            time = f"{time_hour} hour{plural_hours} and {time_min} minute{plural_minutes} "
        elif time_hour == 0 and time_min > 0:
            time = f"{time_min} minute{plural_minutes} "
        else:
            time = f"{time_hour} hour{plural_hours} "
    return time
