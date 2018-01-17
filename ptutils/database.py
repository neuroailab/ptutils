import bson
import copy
import time
import gridfs
import pickle
import hashlib
import datetime
import threading
import numpy as np
import pymongo as pm
from bson.binary import Binary
from bson.objectid import ObjectId

import torch
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

from .base import Base

jsonpickle_numpy.register_handlers()


class DBInterface(Base):
    """Interface for all DBInterface subclasses.

    Your database class should subclass this interface by maintaining the
    regular attribute
        `db_name`
    and implementing the following methods:

    `save(obj)`
        Save the python object `obj` to the database `db` and return an identifier
        `object_id`.

    `load(obj)`

    `delete(obj)`
        Remove `obj` from the database `self.db_name`.
    """

    def __init__(self, *args, **kwargs):
        super(DBInterface, self).__init__(*args, **kwargs)

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def delete(self):
        raise NotImplementedError()


class MongoInterface(DBInterface):
    """Simple and lightweight mongodb interface for saving experimental data files."""

    def __init__(self,
                 database_name,
                 collection_name,
                 host='localhost',
                 port=27017,
                 **kwargs):
        super(MongoInterface, self).__init__(**kwargs)

        self.host = host
        self.port = port
        self.database_name = database_name
        self.collection_name = collection_name

        self.checkpoint_thread = None
        self.client = pm.MongoClient(self.host, self.port)
        self.database = self.client[self.database_name]
        self.collection = self.database[self.collection_name]
        self.filesystem = gridfs.GridFS(self.database)
        self._exclude_from_params = ['client', 'database', 'collection',
                                     'filesystem', 'checkpoint_thread']

    @classmethod
    def from_params(cls, database_name, collection_name, **params):
        return cls(database_name, collection_name, **params)

    def _close(self):
        self.client.close()

    # def __del__(self):
        # self._close()

    # def __repr__(self):
    #     """Return module string representation."""
    #     repstr = '{} ({}): (\n'.format(type(self).__name__, self.name)
    #     for name, param in self._params.items():
    #         if name in ['host', 'port', 'database_name', 'collection_name']:
    #             repstr += '  ({}): {} \n'.format(name, param)
    #     repstr = repstr + ')'
    #     return repstr

    # Public methods: ---------------------------------------------------------

    def save(self, document, multithreaded=True):
        """Store a dictionary or list of dictionaries as as a document in collection.

        The collection is specified in the initialization of the object.

        Note that if the dictionary has an '_id' field, and a document in the
        collection as the same '_id' key-value pair, that object will be
        overwritten.  Any tensors will be stored in the gridFS,
        replaced with ObjectId pointers, and a list of their ObjectIds will be
        also be stored in the 'tensor_id' key-value pair.  If re-saving an
        object- the method will check for old gridfs objects and delete them.

        If multithreaded is true, calls private _save method to spawn new thread.

        Args:
            document: dictionary of arbitrary size and structure,
            can contain tensors. Can also be a list of such objects.
            multithreaded (boolean, Optional)

        Returns:
            id_values: list of ObjectIds of the inserted object(s).

        """
        if multithreaded:
            thread = threading.Thread(target=self._save, args=(document,))
            thread.daemon = True
            thread.start()
            self.checkpoint_thread = thread
        else:
            return self._save(document)

    def load_from_ids(self, ids):
        """Conveience function to load from a list of ObjectIds or from their
         string representations.  Takes a singleton or a list of either type.

        Args:
            ids: can be an ObjectId, string representation of an ObjectId,
            or a list containing items of either type.

        Returns:
            out: list of documents from the DB.  If a document w/the object
                did not exist, a None object is returned instead.

        """
        self.sync_with_host()
        if type(ids) is not list:
            ids = [ids]

        out = []

        for id in ids:
            if type(id) is ObjectId:
                obj_id = id
            elif type(id) is str or type(id) is unicode:
                try:
                    obj_id = ObjectId(id)
                except TypeError:
                    obj_id = id
            out.append(self.load({'_id': obj_id}))

        return out

    def load(self, query, get_tensors=True):
        """Perform a search using the presented query.

        Args:
            query: dictionary of key-value pairs to use for querying the mongodb

        Returns:
            all_results: list of full documents from the collection

        """
        self.sync_with_host()
        query = self._mongoify(query)
        results = self.collection.find(query, sort=[('insertion_date', -1)]).limit(1)
        # results = self.collection.find(query, sort=[('insertion_date', -1)])

        if get_tensors:
            all_results = [self._de_mongoify(
                self._load_tensor(doc)) for doc in results]
        else:
            all_results = [self._de_mongoify(doc) for doc in results]

        return all_results

    def delete(self, object_id):
        """Delete a specific document from the collection based on the objectId.

        Note that it first deletes all the gridFS files pointed to by ObjectIds
        within the document.

        Use with caution, clearly.

        Args:
            object_id: an id of an object in the database.
        """
        document_to_delete = self.collection.find_one({"_id": object_id})
        tensors_to_delete = document_to_delete['_tensor_ids']
        for tensor_id in tensors_to_delete:
            self.filesystem.delete(tensor_id)
        self.collection.remove(object_id)

    def sync_with_host(self):
        if self.checkpoint_thread is not None:
            self.checkpoint_thread.join()
            self.checkpoint_thread = None

    # Private methods ---------------------------------------------------------
    def _save(self, document):
        """Helper method that saves document in database.

        The collection is specified in the initialization of the object.

        Note that if the dictionary has an '_id' field, and a document in the
        collection as the same '_id' key-value pair, that object will be
        overwritten.  Any tensors will be stored in the gridFS,
        replaced with ObjectId pointers, and a list of their ObjectIds will be
        also be stored in the 'tensor_id' key-value pair.  If re-saving an
        object- the method will check for old gridfs objects and delete them.

        Args:
            document: dictionary of arbitrary size and structure,
            can contain tensors. Can also be a list of such objects.

        Returns:
            id_values: list of ObjectIds of the inserted object(s).

        """
        # Simplfy things below by making even a single document a list.

        if not isinstance(document, list):
            document = [document]

        object_ids = []
        for doc in document:
            doc = self._extract_data_from_variables(doc)
            if 'state' in doc.keys():
                state_on_cpu = self._move_to_cpu(doc['state'])
                doc['state'] = state_on_cpu

            doc_copy = copy.deepcopy(doc)

            # Make a list of any existing referenced gridfs files.
            try:
                self._old_tensor_ids = doc_copy['_tensor_ids']
            except KeyError:
                self._old_tensor_ids = []

            self._new_tensor_ids = []

            # Replace tensors with either a new gridfs file or a reference to
            # the old gridfs file.
            doc_copy = self._save_tensors(doc_copy)

            doc['_tensor_ids'] = self._new_tensor_ids
            doc_copy['_tensor_ids'] = self._new_tensor_ids

            # Cleanup any remaining gridfs files (these used to be pointed to by document, but no
            # longer match any tensor that was in the db.
            for id in self._old_tensor_ids:
                self.filesystem.delete(id)
            self._old_tensor_ids = []

            # Add insertion date field to every document.
            doc['insertion_date'] = datetime.datetime.now()
            doc_copy['insertion_date'] = datetime.datetime.now()

            # Insert into the collection and restore full data into original
            # document object
            doc_copy = self._mongoify(doc_copy)

            new_id = self.collection.save(doc_copy)
            doc['_id'] = new_id
            object_ids.append(new_id)

        return object_ids

    def _move_to_cpu(self, state):
        """Move state to CPU.

        Args:
            state (dict): A PyTorch-like state_dict
        """
        return {n: t.cpu() for n, t in state.items()}

    def _tensor_to_binary(self, tensor):
        """Utility method to turn an tensor/array into a BSON Binary string.

        Called by save_tensors.

        Args:
            tensor: tensor of arbitrary dimension.

        Returns:
            BSON Binary object a pickled tensor.
        """
        try:
            return Binary(pickle.dumps(tensor.cpu(), protocol=2), subtype=128)
        except AttributeError:
            return Binary(pickle.dumps(tensor, protocol=2), subtype=128)
        # return Binary(jsonpickle.encode(tensor))
        # return jsonpickle.encode(tensor)

    def _binary_to_tensor(self, binary):
        """Convert a pickled tensor string back into a tensor.

        Called by load_tensors.

        Args:
            binary: BSON Binary object a pickled tensor.

        Returns:
            Tensor of arbitrary dimension.

        """
        return pickle.loads(binary)
        # return jsonpickle.decode(binary)

    def _replace(self, document, replace='.', replacement='__'):
        """Replace `replace` in dictionary keys with `replacement`."""
        for (key, value) in document.items():
            new_key = key.replace(replace, replacement)
            if isinstance(value, dict):
                document[new_key] = self._replace(document.pop(key),
                                                  replace=replace,
                                                  replacement=replacement)
            else:
                document[new_key] = document.pop(key)
        return document

    def __extract_data_from_variables(self, document):
        for (key, value) in document.items():
            if isinstance(value, dict):
                document[key] = self._extract_data_from_variables(value)
            elif isinstance(value, list):
                document[key] = self._extract_data_from_variables(value)
            elif isinstance(value, torch.autograd.Variable):
                # get data from variable
                try:
                    # if variable is on GPU
                    document[key] = value.data.cpu()
                except Exception:
                    document[key] = value.data
        return document

    @staticmethod
    def _extract_data_from_variables(value):
        if isinstance(value, torch.autograd.Variable):
            # get data from variable
            try:
                # if variable is on GPU
                return value.data.cpu()
            except Exception as e:
                return value.data
        if isinstance(value, dict):
            return {k: MongoInterface._extract_data_from_variables(v)
                    for k, v in value.items()}
        elif isinstance(value, list):
            return [MongoInterface._extract_data_from_variables(v) for v in value]
        return value

    def __mongoify(self, document):
        """Modify the document so that it can be stored in MongoDB.

        Called before saving to the database. Replaces '.' (which are rejected
        by mongo) in keys with '__'  and serializes objects that are unserializable.

        Args:
            document: dict to be saved in mongo

        """
        # for (key, value) in document.items():
        for key in document:
            print(key)
        for key in document:
            new_key = key.replace('.', '__')  # mongo cannot use '.' in doc key
            document[new_key] = document.pop(key)
            print('key: {}'.format(key))
            if isinstance(popped_value, dict):
                print('dict: {}'.format(key))
                document[new_key] = self._mongoify(popped_value)
            # elif isinstance(popped_value, list):
                # document[new_key] = [self._mongoify(_) if isinstance(_, (dict, type)) else _ for _ in popped_value]
            else:
                if isinstance(popped_value, type):  # mongo cannot natively serialize these; use jsonpickle
                    print('type: {}'.format(key))
                    document[new_key] = jsonpickle.encode(popped_value)
                else:
                    document[new_key] = popped_value
        return document

    def _mongoify(self, value):
        """Modify the document so that it can be stored in MongoDB.

        Called before saving to the database. Replaces '.' (which are rejected
        by mongo) in keys with '__'  and serializes objects that are unserializable.

        Args:
            document: dict to be saved in mongo

        """
        # for (key, value) in document.items():
        if isinstance(value, type):
            return jsonpickle.encode(value)
        elif isinstance(value, dict):
            return {k.replace('.', '__'): self._mongoify(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._mongoify(v) for v in value]
        else:
            return value

    def _de_mongoify(self, value):
        if isinstance(value, dict):
            return {k.replace('__', '.'): self._de_mongoify(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._de_mongoify(v) for v in value]
        else:
            try:
                return jsonpickle.decode(value)
            except Exception:
                return value

    def __de_mongoify(self, document):
        # untested
        for (key, value) in document.items():
            new_key = key.replace('__', '.')  # mongo cannot use '.' in doc key
            popped_value = document.pop(key)
            if isinstance(value, dict):
                document[new_key] = self._de_mongoify(popped_value)
            else:
                try:
                    # these should be classes that were serialized with jsonpickle
                    # before being stored in the database
                    document[new_key] = jsonpickle.decode(popped_value)
                except Exception:
                    document[new_key] = popped_value
        return document

    def __load_tensor(self, document):
        """Replace ObjectIds with their corresponding gridFS data.

        Utility method to recurse through a document and gather all ObjectIds and
        replace them one by one with their corresponding data from the gridFS collection.

        Skips any entries with a key of '_id'.

        Note that it modifies the document in place.

        Args:
            document: dictionary-like document, storable in mongodb.

        Returns:
            document: dictionary-like document, storable in mongodb.

        """
        for (key, value) in document.items():
            if isinstance(value, ObjectId) and key != '_id':
                # if key == '_Variable_data':
                    # document = torch.autograd.Variable(
                        # self._binary_to_tensor(self.filesystem.get(value).read()))
                # else:
                document[key] = self._binary_to_tensor(self.filesystem.get(value).read())

            elif isinstance(value, dict):
                document[key] = self._load_tensor(value)
        return document

    def _save_tensors(self, value):
        """Replace tensors with a reference to their location in gridFS.

        Utility method to recurse through a document and replace all tensors
        and store them in the gridfs, replacing the actual tensors with references to the
        gridfs path.

        Called by save()

        Note that it modifies the document in place, although we return it, too

        Args:
            document: dictionary like-document, storable in mongodb.

        Returns:
            document: dictionary like-document, storable in mongodb.

        """
        if isinstance(value, np.ndarray) or torch.is_tensor(value):
            data_BSON = self._tensor_to_binary(value)
            data_MD5 = hashlib.md5(data_BSON).hexdigest()

            # Does this tensor match the hash of anything in the object
            # already?
            match = False
            for tensor_id in self._old_tensor_ids:
                print('Checking if {} is already in the db... '.format(tensor_id))
                if data_MD5 == self.filesystem.get(tensor_id).md5:
                    match = True
                    # print('Tensor is already in the db. Replacing tensor with old OjbectId: {}'.format(tensor_id))
                    self._old_tensor_ids.remove(tensor_id)
                    self._new_tensor_ids.append(tensor_id)
                    return tensor_id
            if not match:
                # print('Tensor is not in the db. Inserting new gridfs file...')
                tensor_id = self.filesystem.put(self._tensor_to_binary(value))
                self._new_tensor_ids.append(tensor_id)
                return tensor_id
        elif isinstance(value, dict):
            return {k: self._save_tensors(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._save_tensors(v) for v in value]

        elif isinstance(value, np.number):
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.inexact):
                return float(value)

        return value

    def _load_tensor(self, value):
        """Replace ObjectIds with their corresponding gridFS data.

        Utility method to recurse through a document and gather all ObjectIds and
        replace them one by one with their corresponding data from the gridFS collection.

        Skips any entries with a key of '_id'.

        Note that it modifies the document in place.

        Args:
            document: dictionary-like document, storable in mongodb.

        Returns:
            document: dictionary-like document, storable in mongodb.

        """
        if isinstance(value, ObjectId):
            try:
                return self._binary_to_tensor(self.filesystem.get(value).read())
            except Exception:
                pass
        if isinstance(value, dict):
            return {k: self._load_tensor(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._load_tensor(v) for v in value]

        return value

    # def __save_tensors(self, document):
    #     """Replace tensors with a reference to their location in gridFS.

    #     Utility method to recurse through a document and replace all tensors
    #     and store them in the gridfs, replacing the actual tensors with references to the
    #     gridfs path.

    #     Called by save()

    #     Note that it modifies the document in place, although we return it, too

    #     Args:
    #         document: dictionary like-document, storable in mongodb.

    #     Returns:
    #         document: dictionary like-document, storable in mongodb.

    #     """
    #     for (key, value) in document.items():

    #         if isinstance(value, torch.autograd.Variable):
    #             value = {'_Variable_data': value.data}

    #         if isinstance(value, np.ndarray) or torch.is_tensor(value):
    #             data_BSON = self._tensor_to_binary(value)
    #             data_MD5 = hashlib.md5(data_BSON).hexdigest()

    #             # Does this tensor match the hash of anything in the object
    #             # already?
    #             match = False
    #             for tensor_id in self._old_tensor_ids:
    #                 print('Checking if {} is already in the db... '.format(tensor_id))
    #                 if data_MD5 == self.filesystem.get(tensor_id).md5:
    #                     match = True
    #                     # print('Tensor is already in the db. Replacing tensor with old OjbectId: {}'.format(tensor_id))
    #                     document[key] = tensor_id
    #                     self._old_tensor_ids.remove(tensor_id)
    #                     self._new_tensor_ids.append(tensor_id)
    #             if not match:
    #                 # print('Tensor is not in the db. Inserting new gridfs file...')
    #                 tensor_id = self.filesystem.put(self._tensor_to_binary(value))
    #                 document[key] = tensor_id
    #                 self._new_tensor_ids.append(tensor_id)

    #         elif isinstance(value, dict):
    #             document[key] = self._save_tensors(value)
    #         elif isinstance(value, list):
    #             document[key] = self._save_tensors(value)

    #         elif isinstance(value, np.number):
    #             if isinstance(value, np.integer):
    #                 document[key] = int(value)
    #             elif isinstance(value, np.inexact):
    #                 document[key] = float(value)

    #     return document
