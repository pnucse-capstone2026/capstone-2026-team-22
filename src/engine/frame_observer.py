"""RL이 소유한 프레임마다 GlobalMap 관측을 끼워 넣는 PyBoy 래퍼."""

from __future__ import annotations

from typing import Any


class ObservingPyBoy:
    """``tick`` 뒤에 ``gmap.update_from_pyboy`` 를 호출하는 투명 프록시."""

    __slots__ = ("_pyboy", "_gmap")

    def __init__(self, pyboy: Any, gmap: Any) -> None:
        object.__setattr__(self, "_pyboy", pyboy)
        object.__setattr__(self, "_gmap", gmap)

    @property
    def raw(self) -> Any:
        """감싸기 전의 PyBoy 인스턴스."""
        return object.__getattribute__(self, "_pyboy")

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_pyboy"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_pyboy"), name, value)

    def tick(self, *args: Any, **kwargs: Any) -> Any:
        pyboy = object.__getattribute__(self, "_pyboy")
        result = pyboy.tick(*args, **kwargs)
        object.__getattribute__(self, "_gmap").update_from_pyboy(pyboy)
        return result
