import cloudpickle
from collections.abc import Mapping


class EqualHandler:
    """Base class for all repr handlers."""

    def can_handle(self, obj):
        """Determine if this handler can process the object."""
        raise NotImplementedError

    def handle(self, obj1, obj2, builder, visited):
        """Return the string representation of the object."""
        raise NotImplementedError


class BasicTypeHandler(EqualHandler):
    """Handler for basic built-in types."""

    def can_handle(self, obj):
        return isinstance(obj, (bool, type(None), str))

    def handle(self, obj1, obj2, builder, visited):
        """Return the string representation of the object."""
        return obj1 == obj2


class NumericTypeHandler(EqualHandler):
    """Handler for basic built-in types."""
    maximum_proportional_difference = 0.001

    def can_handle(self, obj):
        return isinstance(obj, (int, float))

    def handle(self, obj1, obj2, builder, visited):
        """Return the string representation of the object."""
        if isinstance(obj1, int) and isinstance(obj2, int):
            return obj1 == obj2
        obj1 = float(obj1)
        obj2 = float(obj2)
        diff = abs(obj1 - obj2)
        if diff == 0.0:
            return True
        div = max(abs(obj1), abs(obj2))
        if div == 0.0:
            return True
        diff_prop = diff / div
        return diff_prop <= self.maximum_proportional_difference


class SequenceHandler(EqualHandler):
    """Handler for sequence types like list and tuple."""
    brackets = {"list": '[]', 'tuple': '()'}

    def can_handle(self, obj):
        return isinstance(obj, (list, tuple))

    def handle(self, obj1, obj2, builder, visited):
        try:
            return visited[(id(obj1), id(obj2))]
        except KeyError:
            # For now, we assume the equality to be True
            visited[(id(obj1), id(obj2))] = True

        if len(obj1) != len(obj2):
            result = False
        else:
            result = True
            for i in range(len(obj1)):
                if not builder.equal(obj1[i], obj2[i], visited):
                    result = False
                    break
        visited[(id(obj1), id(obj2))] = result
        return result


class SetHandler(EqualHandler):
    """Handler for set and frozenset types."""

    def can_handle(self, obj):
        return isinstance(obj, (set, frozenset))

    def handle(self, obj1, obj2, builder, visited):
        try:
            return visited[(id(obj1), id(obj2))]
        except KeyError:
            # For now, we assume the equality to be True
            visited[(id(obj1), id(obj2))] = True

        if len(obj1) != len(obj2):
            result = False
        else:
            obj1_ = list(obj1)
            obj2_ = list(obj2)
            result = True
            for i in range(len(obj1_)):
                if not builder.equal(obj1_[i], obj2_[i], visited):
                    result = False
                    break

        visited[(id(obj1), id(obj2))] = result
        return result


class MappingHandler(EqualHandler):
    """Handler for mapping types like dict."""

    def can_handle(self, obj):
        return isinstance(obj, Mapping)

    def handle(self, obj1, obj2, builder, visited):
        try:
            return visited[(id(obj1), id(obj2))]
        except KeyError:
            # For now, we assume the equality to be True
            visited[(id(obj1), id(obj2))] = True

        if len(obj1) != len(obj2):
            result = False
        else:
            obj1_ = list(obj1.items())
            obj2_ = list(obj2.items())
            result = True
            for i in range(len(obj1_)):
                if not builder.equal(obj1_[i], obj2_[i], visited):
                    result = False
                    break

        visited[(id(obj1), id(obj2))] = result
        return result


class ObjectHandler(EqualHandler):
    """Handler for user-defined objects."""

    def can_handle(self, obj):
        return True  # Fallback handler

    def handle(self, obj1, obj2, builder, visited):
        try:
            return visited[(id(obj1), id(obj2))]
        except KeyError:
            # For now, we assume the equality to be True
            visited[(id(obj1), id(obj2))] = True

        if type(obj1) is not type(obj2):
            result = False
        else:
            try:
                attrs1 = vars(obj1)
                attrs2 = vars(obj2)
                result = builder.equal(attrs1, attrs2, visited)
            except TypeError:
                try:
                    result = cloudpickle.dumps(obj1) == cloudpickle.dumps(obj2)
                except Exception:
                    result = id(obj1) == id(obj2)

        visited[(id(obj1), id(obj2))] = result
        return result


class EqualBuilder:
    """Main class to build string representations of objects."""

    def __init__(self):
        self.handlers = [
            NumericTypeHandler(),
            BasicTypeHandler(),
            SequenceHandler(),
            SetHandler(),
            MappingHandler(),
            ObjectHandler()
        ]

    def equal(self, obj1, obj2, visited=None):
        if visited is None:
            visited = {}
        if id(obj1) == id(obj2):
            return True
        for handler in self.handlers:
            if handler.can_handle(obj1) and handler.can_handle(obj2):
                return handler.handle(obj1, obj2, self, visited)
        return False  # Fallback


_builder = EqualBuilder()
generic_equal = _builder.equal
