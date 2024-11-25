from typing import Optional, List


class Step:

    _display_text: str = None
    _detail_text: List[str] = []
    _img_url: str = None

    # Only applicable for recipes
    _step_duration_seconds: int = None
    _step_ingredients: List[str] = None
    _step_video_url: Optional[str] = None

    # only for wikihow
    _tips_list: List[str] = None

    def __init__(self, text: str, detail_text: List[str], img_url: Optional[str],
                 step_duration_seconds: Optional[int], step_ingredients: Optional[List[str]], step_video_url: None,
                 tips_list: List[str] = None):
        self._text = self.clean_step_text(text)
        self._detail_text = [self.clean_step_text(t) for t in detail_text] if detail_text else detail_text
        self._img_url = img_url
        self._step_video_url = step_video_url
        self._step_duration_seconds = step_duration_seconds
        self._step_ingredients = step_ingredients
        self._tips_list = [self.clean_step_text(t) for t in tips_list] if tips_list else tips_list

    @staticmethod
    def clean_step_text(text: str):
        # cleans some stuff from the text for alexa to read more human like
        if text:
            return text.replace('approx.', 'approximately').replace(r'\xa0', r' ').replace("˚", "°")
        else:
            return text

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text

    def get_detail_text(self) -> List[str]:
        return self._detail_text

    def set_detail_text(self, detail_text: List[str]) -> None:
        self._detail_text = detail_text

    def get_img_url(self) -> str:
        return self._img_url

    def set_img_url(self, img_url: str) -> None:
        self._img_url = img_url

    def get_step_duration_seconds(self) -> int:
        return self._step_duration_seconds

    def set_step_duration_seconds(self, step_duration_seconds: int) -> None:
        self._step_duration_seconds = step_duration_seconds

    def get_step_ingredients(self) -> List[str]:
        return self._step_ingredients

    def set_step_ingredients(self, step_ingredients: List[str]) -> None:
        self._step_ingredients = step_ingredients

    def get_step_video_url(self) -> str:
        return self._step_video_url

    def get_tips_text(self):
        return self._tips_list

    def set_tips_text(self, tips_list: List[str]) -> None:
        self._tips_list = tips_list
