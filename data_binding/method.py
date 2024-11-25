from typing import Optional, List
from data_binding.ingredient import Ingredient
from data_binding.step import Step


class Method:
    _name: str = None
    _method_img_url: str = None
    _ingredients: List[Ingredient] = None
    _steps: List[Step] = None

    def __init__(self, name: str, method_img_url: Optional[str], ingredients: Optional[List[Ingredient]],
                 steps: List[Step]):
        self._name = name
        self._method_img_url = method_img_url
        self._ingredients = ingredients
        self._steps = steps

    def get_name(self) -> str:
        return self._name

    def set_name(self, name: str) -> None:
        self._name = name

    def get_method_img_url(self) -> str:
        return self._method_img_url

    def set_method_img_url(self, method_img_url: str) -> None:
        self._method_img_url = method_img_url

    def get_ingredients(self) -> List[Ingredient]:
        return self._ingredients

    def set_ingredients(self, ingredients: List[Ingredient]) -> None:
        self._ingredients = ingredients

    def get_steps(self) -> List[Step]:
        return self._steps

    def set_steps(self, steps: List[Step]) -> None:
        self._steps = steps
