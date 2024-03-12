import inspect
from typing import Annotated, Any, ClassVar, get_args, get_origin, get_type_hints, Mapping, TypeVar

import forge
from typing_extensions import Self

from feign import copy_function, feign


_T = TypeVar('_T')


class UtilityBase:
    _no_value = object()
    _private = object()
    _include = object()

    @classmethod
    def private(cls, hint: _T) -> _T:
        return Annotated[hint, cls._private]

    @classmethod
    def is_private(cls, hint) -> bool:
        return get_origin(hint) is Annotated and get_args(hint)[1] is cls._private

    @classmethod
    def include(cls, hint: _T) -> _T:
        return Annotated[hint, cls._include]

    @classmethod
    def is_include(cls, hint) -> bool:
        return get_origin(hint) is Annotated and get_args(hint)[1] is cls._include

    _params: ClassVar[Mapping[str, Any]]

    def set_params(self, **kwargs) -> Self:
        for key, val in kwargs.items():
            if key in self._params and val is not self._no_value:
                setattr(self, key, val)
        return self

    @classmethod
    def fix_call_signature(cls):
        # return cls.__call__
        return feign(copy_function(cls.__call__), inspect.signature(forge.compose(*(
            forge.delete(name)
            for name, param in inspect.signature(cls.__call__).parameters.items()
            if param.kind is param.KEYWORD_ONLY
        ), forge.insert((
            forge.kwarg(name, default=cls._no_value, type=hint)
            for name, hint in cls._params.items()
        ), index=-1))(cls.__call__)))

    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._params = dict((
            item for item in get_type_hints(cls, include_extras=True).items()
            for name, hint in [item]
            if cls.is_include(hint) or not (
                name.startswith('_')
                or cls.is_private(hint)
                or get_origin(hint) is ClassVar
                or isinstance(getattr(cls, item[0], None), property)
            )))
        cls.__call__ = cls.fix_call_signature()
