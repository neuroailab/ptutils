"""ptutils Runner tests.

These tests show basic procedures for training, validating, and extracting features from
models.

Note about MongoDB:
The tests require a MongoDB instance to be available on the port defined by "testport" in
the code below.   This db can either be local to where you run these tests (and therefore
on 'localhost' by default) or it can be running somewhere else and then by ssh-tunneled on
the relevant port to the host where you run these tests.  [That is, before testing, you'd run
         ssh -f -N -L  [testport]:localhost:[testport] [username]@mongohost.xx.xx
on the machine where you're running these tests.   [mongohost] is the where the mongodb
instance is running.
"""
from __future__ import division, print_function, absolute_import

import os
import re
import sys
import time
import errno
import shutil
import logging
import pymongo
import unittest
import numpy as np
from bson.objectid import ObjectId

import torch
import torch.nn as nn
import torch.optim as optim

from ptutils import base, data, error, model, runner, database

LOG_LEVEL = 'WARNING'
MONOGO_PORT = 27017


def setUpModule():
    """Set up module once, before any TestCases are run."""
    logging.basicConfig()


def tearDownModule():
    """Tear down module after all TestCases are run."""
    pass


class TestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class once before any test methods are run."""
        cls.setup_log()
        cls.test_class = base.Base

    def setUp(self):
        """Set up class before each test method is executed."""
        pass

    def tearDown(self):
        """Tear Down is called after each test method is executed."""
        pass

    def test_init(self):
        """Test various combiniations of possible inits."""
        # Test empty base class
        base = self.test_class()
        self.assertEqual(base.name, 'base')

    # Test to_params ----------------------------------------------------------

    def test_to_params(self):
        """Illustrate the behavior of the `Base.to_state` method.

        The `to_state` method is an enhanced version of pytorch's native
        `torch.nn.module.state_dict` that seeks to establish a namespace
        with respect to that base object. One can therefore obtain a pseudo-
        global namespace of a ptutils experiment via the `to_state` method
        of the root Base class, which will be the `Runner` class in almost
        every time.

        Similar to pytorch, `to_state` returns a 'flat' (non-nested) ordered
        dict that maps names to parameters. However, a key difference is that
        a Base object needn't be an instance of a torch.nn.Module or even have
        nn.Modules as immediate children for it to be able to produce a
        `state_dict`-like dictionary.

        Caveat:
        """
        # Base with torch.nn.Module child.
        base = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.linear = linear
        self.assertEqual(base.to_state().keys(),
                         ['linear.weight', 'linear.bias'])

    def test_to_state_with_base_child_with_module_child(self):
        """Base with Base child with torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.child.linear = linear
        self.assertEqual(base.to_state().keys(),
                         ['child.linear.weight', 'child.linear.bias'])

    def test_to_state_with_base_and_module_child(self):
        """Base with Base child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.linear = linear
        self.assertItemsEqual(
            base.to_state().keys(),
            ['linear.weight', 'linear.bias'])

    def test_to_state_with_module_child_and_base_child_with_module_child(self):
        """Base child with torch.nn.Module child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        child_linear = torch.nn.Linear(4, 4)
        base.child = child
        base.linear = linear
        base.child.child_linear = child_linear
        self.assertItemsEqual(
            base.to_state().keys(),
            ['linear.weight', 'linear.bias',
             'child.child_linear.weight', 'child.child_linear.bias'])
    # Test from_params --------------------------------------------------------

    def test_from_params(self):
        params = {'test_param_name': 'test_param_value',
                  'test_base_name': self.test_class()}
        base = self.test_class.from_params(**params)

    # def test_from_params(self):
        # params = {'invalid_param_key': 'invalid_param_value'}
        # with self.assertRaises(error.ParamError):
            # base = self.test_class.from_params(**params)

    # Test to_state -----------------------------------------------------------

    def test_to_state(self):
        """Illustrate the behavior of the `Base.to_state` method.

        The `to_state` method is an enhanced version of pytorch's native
        `torch.nn.module.state_dict` that seeks to establish a namespace
        with respect to that base object. One can therefore obtain a pseudo-
        global namespace of a ptutils experiment via the `to_state` method
        of the root Base class, which will be the `Runner` class in almost
        every time.

        Similar to pytorch, `to_state` returns a 'flat' (non-nested) ordered
        dict that maps names to parameters. However, a key difference is that
        a Base object needn't be an instance of a torch.nn.Module or even have
        nn.Modules as immediate children for it to be able to produce a
        `state_dict`-like dictionary.

        Caveat:
        """
        # Base with torch.nn.Module child.
        base = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.linear = linear
        self.assertEqual(base.to_state().keys(),
                         ['linear.weight', 'linear.bias'])

    def test_to_state_with_base_child_with_module_child(self):
        """Base with Base child with torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.child.linear = linear
        self.assertEqual(base.to_state().keys(),
                         ['child.linear.weight', 'child.linear.bias'])

    def test_to_state_with_base_and_module_child(self):
        """Base with Base child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.linear = linear
        self.assertItemsEqual(
            base.to_state().keys(),
            ['linear.weight', 'linear.bias'])

    def test_to_state_with_module_child_and_base_child_with_module_child(self):
        """Base child with torch.nn.Module child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        child_linear = torch.nn.Linear(4, 4)
        base.child = child
        base.linear = linear
        base.child.child_linear = child_linear
        self.assertItemsEqual(
            base.to_state().keys(),
            ['linear.weight', 'linear.bias',
             'child.child_linear.weight', 'child.child_linear.bias'])

    # Test from_state ---------------------------------------------------------

    def test_from_state(self):
        """Illustrate the basic behaviour of `from_state`.

        The `from_state` method is `to_state`'s 'inverse' that restores the
        params of a Base object.

        """
        # Generate the state of a test base class.
        base = self.setup_base()
        state = base.to_state()

        # Alter the state of the base class.
        base.linear = torch.nn.Linear(1, 1)  # Reinitialize linear layer.
        new_state = base.to_state()

        # The altered state should _not_ be the same as the original.
        for old, new in zip(state.values(), new_state.values()):
            self.assertFalse(torch.equal(old, new))

        # Now, restore the base to its original state.
        base.from_state(state)
        restored = base.to_state()

        # The restored state should now be the same.
        for s, r in zip(state.values(), restored.values()):
            self.assertTrue(torch.equal(s, r))

    def test_from_state_restore_params_None(self):
        """Demonstate how to selectively restore parameters."""
        state = self.setup_base().to_state()

        # Test None, which should restore all params.
        restore_params = None
        s = self.setup_base().from_state(state, restore_params).to_state()
        for old, new in zip(state.values(), s.values()):
            self.assertTrue(torch.equal(old, new))

    def test_from_state_restore_params_list_of_strings(self):
        """Demonstate how to selectively restore parameters."""
        state = self.setup_base().to_state()

        # Test list of strings.
        restore_params = ['linear.weight']
        s = self.setup_base().from_state(state, restore_params).to_state()
        self.assertTrue(torch.equal(
            state['linear.weight'], s['linear.weight']))
        self.assertFalse(torch.equal(state['linear.bias'], s['linear.bias']))

    def test_from_state_restore_params_regex(self):
        """Demonstate how to selectively restore parameters."""
        state = self.setup_base().to_state()

        # Test regex.
        restore_params = re.compile(r'linear.bias')
        s = self.setup_base().from_state(state, restore_params).to_state()
        self.assertTrue(torch.equal(state['linear.bias'], s['linear.bias']))
        self.assertFalse(torch.equal(
            state['linear.weight'], s['linear.weight']))

    def test_from_state_restore_params_invalid(self):
        """Demonstate how to selectively restore parameters."""
        state = self.setup_base().to_state()

        # Test invalid type (should raise TypeError).
        restore_params = {'invalid_key': 'invalid_value'}
        with self.assertRaises(TypeError):
            self.setup_base().from_state(state, restore_params).to_state()

    def test_from_state_restore_mapping(self):
        """Demonstrate restore parameter mapping."""
        old_state = self.setup_base().to_state()

        # A new base to receive old state.
        new_base = self.test_class()
        new_linear = torch.nn.Linear(1, 1)
        new_base.new_linear = new_linear

        # Define a param mapping from old params to new params.
        restore_mapping = {'linear.bias': 'new_linear.bias',
                           'linear.weight': 'new_linear.weight'}

        # Map old params to new params
        new_base.from_state(old_state, restore_mapping=restore_mapping)
        new_state = new_base.to_state()

        for old, new in zip(old_state.values(), new_state.values()):
            self.assertTrue(torch.equal(old, new))

    def test_from_state_restore_params_and_restore_mapping(self):
        """Demonstrate simultaneous param restoring and mapping."""
        base = self.test_class()
        base.layer1 = torch.nn.Linear(2, 4)
        base.layer2 = torch.nn.Linear(4, 8)
        s = base.to_state()

        # New base with different structure/names.
        new_base = self.test_class()
        new_base.new_layer1 = torch.nn.Linear(2, 4)  # Change name of layer1.
        new_base.layer2 = torch.nn.Linear(4, 8)      # layer2 name remains.

        # Restore only layer1 params and map to new names.
        restore_params = re.compile(r'layer1')
        restore_mapping = {'layer1.bias': 'new_layer1.bias',
                           'layer1.weight': 'new_layer1.weight'}
        new_base.from_state(s, restore_params, restore_mapping)
        ns = new_base.to_state()

        # layer1 has been restored under the new name `new_layer1`.
        self.assertTrue(torch.equal(s['layer1.bias'], ns['new_layer1.bias']))
        self.assertTrue(torch.equal(
            s['layer1.weight'], ns['new_layer1.weight']))

        # layer2 has been reinitialized.
        self.assertFalse(torch.equal(s['layer2.bias'], ns['layer2.bias']))
        self.assertFalse(torch.equal(s['layer2.weight'], ns['layer2.weight']))

    def test_from_state_invalid_structure(self):

        # Incompatiible names
        base = self.test_class()
        invalid_state = self.setup_base().to_state()
        with self.assertRaises(KeyError):
            base.from_state(invalid_state)

        # Incompatiible shapes
        base = self.setup_base()
        invalid_base = self.test_class()
        invalid_base.linear = torch.nn.Linear(2, 2)
        invalid_state = invalid_base.to_state()
        with self.assertRaises(RuntimeError):
            base.from_state(invalid_state)

    # Test apply ---------------------------------------------------------------

    @unittest.skip('Incomplete')
    def test_apply(self):
        """Test apply."""
        base = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.linear = linear
        self.assertEqual(base.to_state().keys(),
                         ['linear.weight', 'linear.bias'])

    @unittest.skip('Incomplete')
    def test_apply_with_base_child_with_module_child(self):
        """Base with Base child with torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.child.linear = linear
        self.assertEqual(base.to_state().keys(),
                         ['child.linear.weight', 'child.linear.bias'])

    @unittest.skip('Incomplete')
    def test_apply_with_base_and_module_child(self):
        """Base with Base child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.linear = linear
        self.assertItemsEqual(
            base.to_state().keys(),
            ['linear.weight', 'linear.bias'])

    @unittest.skip('Incomplete')
    def test_apply_with_module_child_and_base_child_with_module_child(self):
        """Base child with torch.nn.Module child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        child_linear = torch.nn.Linear(4, 4)
        base.child = child
        base.linear = linear
        base.child.child_linear = child_linear
        self.assertItemsEqual(
            base.to_state().keys(),
            ['linear.weight', 'linear.bias',
             'child.child_linear.weight', 'child.child_linear.bias'])

    # Test cuda ----------------------------------------------------------------

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_cuda(self):
        base = self.setup_base()
        base.cuda()

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_cuda_with_base_child_with_module_child(self):
        """Base with Base child with torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        base.child = child
        base.devices = [0, 1]
        base.child.linear = torch.nn.Linear(2, 2)
        base.cuda()

        self.assertTrue(base.use_cuda)
        self.assertTrue(base.child.use_cuda)
        self.assertEqual(base.devices, [0, 1])
        self.assertEqual(base.child.devices, [0, 1])


        base = self.test_class()
        child = self.test_class()
        base.child = child
        base.devices = [0, 1]
        base.child.devices = [2, 3]
        base.child.linear = torch.nn.Linear(2, 2)
        base.cuda()

        self.assertEqual(base.devices, [0, 1])
        self.assertEqual(base.child.devices, [2, 3])

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_cuda_with_base_and_module_child(self):
        """Base with Base child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.linear = linear
        base.cuda()

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda is not available')
    def test_cuda_with_module_child_and_base_child_with_module_child(self):
        """Base child with torch.nn.Module child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        child_linear = torch.nn.Linear(4, 4)
        base.child = child
        base.linear = linear
        base.child.child_linear = child_linear
        base.cuda()

    # Test cpu -----------------------------------------------------------------

    def test_cpu(self):
        base = self.setup_base()
        base.cpu()

    def test_cpu_with_base_child_with_module_child(self):
        """Base with Base child with torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.child.linear = linear
        base.cpu()

    def test_cpu_with_base_and_module_child(self):
        """Base with Base child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.linear = linear
        base.cpu()

    def test_cpu_with_module_child_and_base_child_with_module_child(self):
        """Base child with torch.nn.Module child and torch.nn.Module child."""
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        child_linear = torch.nn.Linear(4, 4)
        base.child = child
        base.linear = linear
        base.child.child_linear = child_linear
        base.cpu()

    @classmethod
    def setup_base(cls, value=None):
        # Generate test base with 1x1 Linear module.
        base = cls.test_class()
        linear = torch.nn.Linear(1, 1)
        base.linear = linear
        return base

    @classmethod
    def setup_log(cls):
        cls.log = logging.getLogger(':'.join([__name__, cls.__name__]))
        cls.log.setLevel(LOG_LEVEL)


class Test(unittest.TestCase):
    """Test class with convenient database access."""

    # Port on which the MongoDB instance to be used by tests needs to be running.
    port = MONOGO_PORT
    # Host on which the MongoDB instance to be used by tests needs to be running.
    host = 'localhost'
    # Name of the mongodb database where results will be stored by tests.
    database_name = 'ptutils-test'
    # Name of the mongodb collection where results will be stored by tests.
    collection_name = 'testcol'
    # Name of local directory to cache checkpoints.
    cache_dir = 'ptutils_test_cache_dir'

    @classmethod
    def setUpClass(cls):
        """Set up class once before any test methods are run."""
        cls.setup_log()
        cls.setup_conn()

    @classmethod
    def tearDownClass(cls):
        """Tear down class after all test methods have run."""
        # cls.remove_directory(cls.cache_dir)
        cls.remove_database(cls.database_name)

        # Close primary MongoDB connection.
        cls.conn.close()

    @classmethod
    def setup_log(cls):
        cls.log = logging.getLogger(':'.join([__name__, cls.__name__]))
        cls.log.setLevel(LOG_LEVEL)

    @classmethod
    def setup_conn(cls):
        cls.conn = pymongo.MongoClient(host=cls.host, port=cls.port)

    @classmethod
    def remove_checkpoint(cls, checkpoint):
        """Remove a tf.train.Saver checkpoint."""
        cls.log.info('Removing checkpoint: {}'.format(checkpoint))
        # TODO: remove ckpt
        cls.log.info('Checkpoint successfully removed.')
        raise NotImplementedError

    @classmethod
    def remove_directory(cls, directory):
        """Remove a directory."""
        try:
            cls.log.info('Removing directory: {}'.format(directory))
            shutil.rmtree(directory)
        except OSError:
            cls.log.info('Directory does not exist.')
        else:
            cls.log.info('Directory successfully removed.')

    @classmethod
    def remove_database(cls, database_name):
        """Remove a MonogoDB database."""
        cls.log.info('Removing database: {}'.format(database_name))
        cls.conn.drop_database(database_name)
        cls.log.info('Database successfully removed.')

    @classmethod
    def remove_collection(cls, collection_name):
        """Remove a MonogoDB collection."""
        cls.log.info('Removing collection: {}'.format(collection_name))
        cls.conn[cls.database_name][collection_name].drop()
        cls.log.info('Collection successfully removed.')

    @classmethod
    def remove_document(cls, document):
        raise NotImplementedError

    @staticmethod
    def makedirs(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class TestMongoInterface(Test):

    def setUp(self):
        self.dbinterface = database.MongoInterface(self.database_name,
                                                   self.collection_name)

    def tearDown(self):
        del self.dbinterface

    def test_to_params(self):
        self.assertDictContainsSubset(
            {'host': self.host,
             'port': self.port,
             'database_name': self.database_name,
             'collection_name': self.collection_name},
            self.dbinterface.to_params())

    def test_from_params(self):
        params = {'host': self.host,
                  'port': self.port,
                  'database_name': self.database_name,
                  'collection_name': self.collection_name}
        dbinterface = database.MongoInterface.from_params(**params)
        params = self.dbinterface.to_params()
        dbinterface = database.MongoInterface.from_params(**params)

    def test_save(self):
        doc = {'exp_id': 'test_save', 'step': 0}
        self.dbinterface.save(doc)
        r = self.conn[self.database_name][self.collection_name].find(
            {'exp_id': 'test_save'})
        self.assertDictContainsSubset({'exp_id': 'test_save', 'step': 0}, r[0])

    def test_save_numpy_array(self):
        array = np.array([1, 2, 3])
        doc = {'exp_id': 'test_save_numpy_array', 'array': array}
        self.dbinterface.save(doc)
        r = self.conn[self.database_name][self.collection_name].find(
            {'exp_id': 'test_save_numpy_array'})
        self.assertIsInstance(r[0]['array'], ObjectId)

    def test_save_torch_tensor(self):
        tensor = torch.Tensor([1, 2, 3])
        doc = {'exp_id': 'test_save_torch_tensor', 'tensor': tensor}
        self.dbinterface.save(doc)
        r = self.conn[self.database_name][self.collection_name].find(
            {'exp_id': 'test_save_torch_tensor'})
        self.assertIsInstance(r[0]['tensor'], ObjectId)

    def test_save_state(self):
        b = base.Base()
        linear = torch.nn.Linear(2, 2)
        b.linear = linear
        state = b.to_state()
        doc = {'exp_id': 'test_save_state', 'state': state}
        self.dbinterface.save(doc)
        r = self.conn[self.database_name][self.collection_name].find(
            {'exp_id': 'test_save_state'})
        for param in r[0]['state'].values():
            self.assertIsInstance(param, ObjectId)

    def test_load(self):
        doc = {'exp_id': 'test_load', 'step': 0}
        self.dbinterface.save(doc)
        r = self.dbinterface.load({'exp_id': 'test_load'})
        self.assertDictContainsSubset({'exp_id': 'test_load', 'step': 0}, r)

    def test_load_numpy_array(self):
        array = np.array([1, 2, 3])
        doc = {'exp_id': 'test_load_numpy_array', 'array': array}
        self.dbinterface.save(doc)
        r = self.dbinterface.load({'exp_id': 'test_load_numpy_array'})
        self.assertTrue(np.array_equal(doc['array'], r['array']))

    def test_load_torch_tensor(self):
        tensor = torch.Tensor([1, 2, 3])
        doc = {'exp_id': 'test_load_torch_tensor', 'tensor': tensor}
        self.dbinterface.save(doc)
        r = self.dbinterface.load({'exp_id': 'test_load_torch_tensor'})
        self.assertTrue(torch.equal(doc['tensor'], r['tensor']))

    def test_load_state(self):
        b = base.Base()
        linear = torch.nn.Linear(2, 2)
        b.linear = linear
        state = b.to_state()
        doc = {'exp_id': 'test_load_state', 'state': state}
        self.dbinterface.save(doc)
        r = self.dbinterface.load({'exp_id': 'test_load_state'})
        restored_state = r['state']
        self.assertItemsEqual(state.keys(), restored_state.keys())
        for name in state:
            self.assertTrue(torch.equal(state[name], restored_state[name]))

    def test_delete(self):
        doc = {'exp_id': 'test_delete', 'step': 0}
        object_id = self.dbinterface.save(doc)
        self.dbinterface.delete(object_id[0])
        r = self.conn[self.database_name][self.collection_name].find(
            {'exp_id': 'test_delete'})
        self.assertEqual(r.count(), 0)


class TestModel(Test):

    @classmethod
    def setUpClass(cls):
        """Set up class once before any test methods are run."""
        cls.setup_log()
        cls.setup_conn()

        # Test primary Model class.
        cls.test_class = model.Model

    def test_from_params_empty(self):
        pass

    def test_from_params(self):
        pass


# class TestMNISTModel(Test):

#     def test_from_params(self):
#         model_params = {
#             'name': 'MNIST',
#             'devices': [0, 1],
#             'net': model.ConvMNIST,
#             'fc': 'fc',
#             'criterion': {
#                 {nn.CrossEntropyLoss: {}}},
#             'optimizer': '',
#             }

#         mnist = model.Model.from_params(model_params)
#         self.log.info(mnist)
#         self.log.info(mnist._params)
#         self.log.info(mnist.to_params())


class TestRunner(Test):

    @classmethod
    def setUpClass(cls):
        """Set up class once before any test methods are run."""
        cls.setup_log()
        cls.setup_conn()

        # Test primary Runner class.
        cls.test_class = runner.Runner

    def setUp(self):
        """Set up class before _each_ test method is executed."""
        # self.setup_cache()
        self.setup_params()

    def tearDown(self):
        """Tear Down is called after _each_ test method is executed."""
        pass

    @unittest.skip('skipping...')
    def test_training_from_objects(self):
        runner = self.test_class(exp_id='test_exp_id')

        runner.num_steps = 500
        runner.model = model.MNISTModel()
        runner.dataprovider = data.MNISTProvider()
        runner.dbinterface = database.MongoInterface(self.database_name,
                                                     self.collection_name)

        runner.save_params = {'metric_freq': 10}
        runner.load_params = {'restore': False}

        self.log.info(runner)
        runner.train_from_params()

    def test_init(self):
        """Test various combiniations of possible inits."""
        # Test empty base class
        runner = self.test_class(exp_id='test')
        self.assertEqual(runner.name, 'runner')

    @unittest.skip('Skip')
    def test_training(self):
        """Illustrate training.

        This test illustrates how basic training is performed using the
        tfutils.base.train_from_params function.  This is the first in a sequence of
        interconnected tests. It creates a pretrained model that is used by
        the next few tests (test_validation and test_feature_extraction).

        As can be seen by looking at how the test checks for correctness, after the
        training is run, results of training, including (intermittently) the full
        variables needed to re-initialize the tensorflow model, are stored in a
        MongoDB.

        Also see docstring of the for more detailed information about usage.

        """
        exp_id = 'training0'
        params = self.setup_params(exp_id=exp_id)

        runner = self.test_class.from_params(**params)
        runner.train_from_params()

        # test if results are as expected
        assert runner.dbinterface.collection.find(
            {'exp_id': exp_id}).count() == 26
        assert runner.dbinterface.collection.find(
            {'exp_id': 'training0', 'saved_filters': True}).distinct('step') == [0, 200, 400]

        r = runner.dbinterface.collection.find(
            {'exp_id': exp_id, 'step': 0})[0]
        self.asserts_for_record(r, params, train=True)

        r = runner.dbinterface.collection.find(
            {'exp_id': exp_id, 'step': 20})[0]
        self.asserts_for_record(r, params, train=True)

        # run another 500 steps of training on the same experiment id.

        runner.train_params['num_steps'] = 1000
        runner.train_from_params()

        # test if results are as expected

        # run 500 more steps but save to a new experiment id.
        runner.train_params['num_steps'] = 1500
        runner.load_params = {'exp_id': 'training0'}
        runner.save_params['exp_id'] = 'training1'

        runner.train_from_params()

    def setup_params(self, exp_id=None):
        """Create params that can be used for training."""
        model_params = {
            'func': model.MNIST,
            'devices': [0, 1],
            'net': '',
            'criterion': '',
            'optimizer': ''}

        save_params = {
            'save_metric_freq': 100,  # Enforce!
            'intermediate': '',
            'save_valid_freq': 20,
            'save_filters_freq': 200,
            'cache_filters_freq': 100}

        load_params = {
            'restore': False,
            'restore_params': None,
            'restore_mapping': None}

        dbinterface_params = {
            'func': database.MongoInterface,
            'host': self.host,
            'port': self.port,
            'database_name': self.database_name,
            'collection_name': self.collection_name}

        dataprovider_params = {'func': data.MNISTProvider,
                               'batch_size': 100,
                               'n_threads': 4},

        train_params = {'num_steps': 500}
        validation_params = {}

        params = {
            'exp_id': 'exp',
            'use_cuda': True,
            'devices': [0, 1, 2, 3],
            'num_steps': 500,
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'train_params': train_params,
            'validation_params': validation_params,
            'dbinterface_params': dbinterface_params,
            'dataprovider_params': dataprovider_params}

        return params

    def test_enforce_exp_id(self):
        runner = self.test_class(exp_id=None)
        with self.assertRaises(error.ExpIdError):
            runner.train_from_params()

    @staticmethod
    def asserts_for_record(r, params, train=False):
        if r.get('saved_filters'):
            assert r['_saver_write_version'] == 2
            assert r['_saver_num_data_files'] == 1
        assert type(r['duration']) == float

        should_contain = ['save_params', 'load_params',
                          'model_params', 'validation_params']
        assert set(should_contain).difference(r['params'].keys()) == set()

        vk = r['params']['validation_params'].keys()
        vk1 = r['validation_results'].keys()
        assert set(vk) == set(vk1)

        assert r['params']['model_params']['seed'] == 0
        assert r['params']['model_params']['func']['modname'] == 'tfutils.model'
        assert r['params']['model_params']['func']['objname'] == 'mnist_tfutils'
        assert set(['hidden1', 'hidden2', u'softmax_linear']).difference(
            r['params']['model_params']['cfg_final'].keys()) == set()

        _k = vk[0]
        should_contain = ['agg_func', 'data_params', 'num_steps',
                          'online_agg_func', 'queue_params', 'targets']
        assert set(should_contain).difference(
            r['params']['validation_params'][_k].keys()) == set()

        if train:
            assert r['params']['model_params']['train'] is True
            for k in ['num_steps', 'queue_params']:
                assert r['params']['train_params'][k] == params['train_params'][k]

            should_contain = ['loss_params', 'optimizer_params',
                              'train_params', 'learning_rate_params']
            assert set(should_contain).difference(r['params'].keys()) == set()
            assert r['params']['train_params']['thres_loss'] == 100
            assert r['params']['train_params']['data_params']['func']['modname'] == 'tfutils.data'
            assert r['params']['train_params']['data_params']['func']['objname'] == 'MNIST'

            assert r['params']['loss_params']['agg_func']['modname'] == 'tensorflow.python.ops.math_ops'
            assert r['params']['loss_params']['agg_func']['objname'] == 'reduce_mean'
            assert r['params']['loss_params']['loss_per_case_func']['modname'] == 'tensorflow.python.ops.nn_ops'
            assert r['params']['loss_params']['loss_per_case_func']['objname'] == 'sparse_softmax_cross_entropy_with_logits'
            assert r['params']['loss_params']['targets'] == ['labels']
        else:
            assert not r['params']['model_params']['train']
            assert 'train_params' not in r['params']


if __name__ == '__main__':
    unittest.main()
