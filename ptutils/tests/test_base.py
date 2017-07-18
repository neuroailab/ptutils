from __future__ import print_function

import os
import unittest
from pprint import pprint as print
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises

from ptutils.base import Module, Configuration


class TestModule(unittest.TestCase):

    test_types = {
        # Numeric types
        'test_int': int(),
        'test_long': long(),
        'test_float': float(),
        'test_complex': complex(),

        # Sequence types,
        'test_str': str(),
        'test_set': set(),
        'test_list': list(),
        'test_tuple': tuple(),
        'test_range': range(1),
        'test_xrange': xrange(1),
        'test_unicode': unicode(),
        'test_bytearray': bytearray(),
        'test_frozenset': frozenset(),
        'test_buffer': buffer('test_buffer'),

        # Mapping types,
        'test_dict': dict(),

        # Other built-in types,
        'test_module': os,
        'test_none': None,
        'test_bool': True,
        'test_type': type,
        'test_function': lambda x: x,
        'test_class': type(str(), tuple(), dict()),
        'test_obj': type(str(), tuple(), dict())(),
        'test_obj': type('TestClass', (object,), {'method': lambda self: self})(),
        'test_method': type(str(), tuple(), {'method': lambda self: self}).method,
    }

    @classmethod
    def setup_class(cls):
        """Setup_class is called once for each class before any tests are run."""
        pass

    @classmethod
    def teardown_class(cls):
        """Teardown_class is called once for each class before any tests are run."""
        pass

    def test_init(self):
        base = Module()
        assert
        for key, value in self.test_types.items():
            mod = Module(value)


class TestConfiguration(TestModule):

    def setup(self):
        """Setup is called before _each_ test method is executed."""
        pass

    def teardown(self):
        """Teardown is called after _each_ test method is executed."""
        pass

    def test_init(self):
        base_c = Configuration()
        int_c = Configuration(self.test_types['test_int'])
        assert(isinstance(base_c, Module))
        assert(isinstance(int_c, Module))

    # def test_return_true(self):
    #     a = A()
    #     assert_equal(a.return_true(), True)
    #     assert_not_equal(a.return_true(), False)

    # def test_raise_exc(self):
    #     a = A()
    #     assert_raises(KeyError, a.raise_exc, "A value")

    # @raises(KeyError)
    # def test_raise_exc_with_decorator(self):
    #     a = A()
    #     a.raise_exc("A message")


