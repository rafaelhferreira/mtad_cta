from typing import Optional


class Ingredient:
    _text: str = None  # Usually combines the ingredient, the preparation and the quantity
    _img_url: str = None

    _ingredient_name: str = None
    _quantity: float = None
    _unit: str = None
    _preparation: str = None

    def __init__(self, text: str, img_url: Optional[str], ingredient_name: Optional[str], quantity: Optional[float],
                 unit: Optional[str], preparation: Optional[str]) -> None:
        self._text = text.replace('_', ' ') if text else ""
        self._img_url = img_url
        self._ingredient_name = ingredient_name
        self._quantity = quantity
        self._unit = unit
        self._preparation = preparation

    def get_text(self) -> str:
        return self._text

    def get_img_url(self) -> str:
        return self._img_url

    def get_ingredient_name(self) -> str:
        return self._ingredient_name

    def get_quantity(self) -> float:
        return self._quantity

    def get_unit(self) -> str:
        return self._unit

    def get_preparation(self) -> str:
        return self._preparation
