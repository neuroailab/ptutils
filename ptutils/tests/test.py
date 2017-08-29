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

import torch
import torch.nn as nn
import torch.optim as optim

from ptutils import base, data, model, database


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
        pass

    def test_init(self):
        """Test various combiniations of possible inits."""
        # Test empty base class
        base = self.test_class()
        self.assertEqual(base.name, 'base')

    def test_from_params(self):
        params = {'test_param_name': 'test_param_value',
                  'test_base_name': self.test_class()}
        base = self.test_class.from_params(params)
        # self.log.info(base)
        # self.log.info(base._params)
        # self.log.info(base._bases)

    def test_to_params(self):
        params = {'test_params': 'test'}
        base = self.test_class.from_params(params)
        # self.log.info(base.to_params())

    def test_to_state(self):

        # Base with torch.nn.Module child.
        base = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.linear = linear
        self.assertEqual(base.to_state().keys(),
                         ['linear.weight', 'linear.bias'])

        # Base with Base child with torch.nn.Module child.
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.child.linear = linear
        self.assertEqual(base.to_state().keys(),
                         ['child.linear.weight', 'child.linear.bias'])

        # Base with Base child and torch.nn.Module child.
        base = self.test_class()
        child = self.test_class()
        linear = torch.nn.Linear(2, 2)
        base.child = child
        base.linear = linear
        self.assertItemsEqual(
            base.to_state().keys(),
            ['linear.weight', 'linear.bias'])

        # Base child with torch.nn.Module child and torch.nn.Module child.
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

    def test_from_state(self):
        # Generate the state of a test base class.
        base = self.setup_base()
        state = base.to_state()

        # Alter the state of the base class.
        base.linear = torch.nn.Linear(1, 1)  # Reinitialize linear layer.
        new_state = base.to_state()

        # The altered state should not be the same as the original.
        for old, new in zip(state.values(), new_state.values()):
            self.assertFalse(torch.equal(old, new))

        # Now, restore the base to its original state.
        base.from_state(state)
        restored = base.to_state()

        # The restored state should be the same.
        for s, r in zip(state.values(), restored.values()):
            self.assertTrue(torch.equal(s, r))

    def test_from_state_restore_params(self):
        state = self.setup_base().to_state()

        # Test None, which should restore all params.
        restore_params = None
        s = self.setup_base().from_state(state, restore_params).to_state()
        for old, new in zip(state.values(), s.values()):
            self.assertTrue(torch.equal(old, new))

        # Test list of strings.
        restore_params = ['linear.weight']
        s = self.setup_base().from_state(state, restore_params).to_state()
        self.assertTrue(torch.equal(state['linear.weight'], s['linear.weight']))
        self.assertFalse(torch.equal(state['linear.bias'], s['linear.bias']))

        # Test regex.
        restore_params = re.compile(r'linear.bias')
        s = self.setup_base().from_state(state, restore_params).to_state()
        self.assertTrue(torch.equal(state['linear.bias'], s['linear.bias']))
        self.assertFalse(torch.equal(state['linear.weight'], s['linear.weight']))

        # Test invalid type (should raise TypeError).
        restore_params = {'invalid_key': 'invalid_value'}
        with self.assertRaises(TypeError):
            self.setup_base().from_state(state, restore_params).to_state()

    def test_from_state_param_mapping(self):
        old_state = self.setup_base().to_state()

        # A new base to receive old state.
        new_base = self.test_class()
        new_linear = torch.nn.Linear(1, 1)
        new_base.new_linear = new_linear

        # Define a param mapping from old params to new params.
        param_mapping = {'linear.bias': 'new_linear.bias',
                         'linear.weight': 'new_linear.weight'}

        # Map old params to new params
        new_base.from_state(old_state, param_mapping=param_mapping)
        new_state = new_base.to_state()

        for old, new in zip(old_state.values(), new_state.values()):
            self.assertTrue(torch.equal(old, new))

    def test_from_state_restore_params_and_param_mapping(self):
        base = self.test_class()
        base.layer1 = torch.nn.Linear(2, 4)
        base.layer2 = torch.nn.Linear(4, 8)
        s = base.to_state()

        new_base = self.test_class()
        new_base.new_layer1 = torch.nn.Linear(2, 4)  # Change name of layer1.
        new_base.layer2 = torch.nn.Linear(4, 8)      # layer2 name remains.

        # Restore only layer1 params and map to new names.
        restore_params = re.compile(r'layer1')
        param_mapping = {'layer1.bias': 'new_layer1.bias',
                         'layer1.weight': 'new_layer1.weight'}
        new_base.from_state(s, restore_params, param_mapping)
        ns = new_base.to_state()

        # layer1 has been restored under the new name `new_layer1`.
        self.assertTrue(torch.equal(s['layer1.bias'], ns['new_layer1.bias']))
        self.assertTrue(torch.equal(s['layer1.weight'], ns['new_layer1.weight']))

        # layer2 has been reinitialized.
        self.assertFalse(torch.equal(s['layer2.bias'], ns['layer2.bias']))
        self.assertFalse(torch.equal(s['layer2.weight'], ns['layer2.weight']))

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
        cls.log.setLevel('DEBUG')


@unittest.skip('Skipping TestRunner')
class TestRunner(unittest.TestCase):

    # Port on which the MongoDB instance to be used by tests needs to be running.
    port = 27017
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

        # Test primary Runner class.
        cls.test_class = base.Runner

    @classmethod
    def tearDownClass(cls):
        """Tear down class after all test methods have run."""
        # cls.remove_directory(cls.cache_dir)
        cls.remove_database(cls.database_name)

        # Close primary MongoDB connection.
        cls.conn.close()

    def setUp(self):
        """Set up class before _each_ test method is executed."""
        # self.setup_cache()
        self.setup_params()

    def tearDown(self):
        """Tear Down is called after _each_ test method is executed."""
        pass

    @unittest.skip('skipping')
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
        self.log.info(params)

        runner = self.test_class.from_params(**params)
        # self.log.info(runner)
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

    @classmethod
    def setup_log(cls):
        cls.log = logging.getLogger(':'.join([__name__, cls.__name__]))
        cls.log.setLevel('DEBUG')

    @classmethod
    def setup_conn(cls):
        cls.conn = pymongo.MongoClient(host=cls.host, port=cls.port)

    @classmethod
    def setup_cache(cls):
        cls.cache_dir = os.path.join(cls.cache_dir,
                                     '%s:%d' % (cls.host, cls.port),
                                     cls.database_name,
                                     cls.collection_name,
                                     cls.EXP_ID)

        cls.makedirs(cls.cache_dir)
        cls.save_path = os.path.join(cls.cache_dir, 'checkpoint')

    def setup_params(self, exp_id=None):
        """Create params that can be used for training."""
        model_params = {'func': model.MNIST,
                        'devices': [0, 1]}
        save_params = {

            'save_valid_freq': 20,
            'save_filters_freq': 200,
            'cache_filters_freq': 100}

        dbinterface_params = {
            'func': database.MongoInterface,
            'host': self.host,
            'port': self.port,
            'dbname': self.database_name,
            'collname': self.collection_name}

        dataprovider_params = {'func': data.MNISTProvider,
                               'batch_size': 100,
                               'n_threads': 4},

        train_params = {'num_steps': 500}

        load_params = {'do_restore': True}

        loss_params = {'func': nn.CrossEntropyLoss}

        optimizer_params = {'func': optim.SGD,
                            'momentum': 0.9,
                            'lr': 0.05}

        params = {
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'train_params': train_params,
            'loss_params': loss_params,
            'optimizer_params': optimizer_params,
            'dbinterface_params': dbinterface_params,
            'dataprovider_params': dataprovider_params}

        return params

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
