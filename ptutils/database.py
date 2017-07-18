import copy
import gridfs
import hashlib
import datetime
import numpy as np
import pymongo as pm
import cPickle as pickle
from bson.binary import Binary
from bson.objectid import ObjectId

import torch
import base


class DBInterface(base.DBInterface):
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

    __name__ = 'dbinterface'

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

    __name__ = 'mongointerface'

    _DEFAULTS = {
        'port': 27017,
        'hostname': 'localhost',
        'db_name': 'DEFAULT_DATABASE',
        'collection_name': 'DEFAULT_COLLECTION',
    }

    # NOTE call via MongoInterface(**config)
    def __init__(self, db_name, collection_name, hostname='localhost', port=27017):
        super(MongoInterface, self).__init__(db_name=db_name,
                                             collection_name=collection_name,
                                             hostname=hostname,
                                             port=port)
        self.db_name = db_name
        self.collection_name = collection_name
        self.hostname = hostname
        self.port = port

    # def __init__(self, *args, **kwargs):
    #     super(MongoInterface, self).__init__(*args, **kwargs)

        # for key, value in MongoInterface._DEFAULTS.items():
        #     if not hasattr(self, key):
        #         self[key] = value

        self.client = pm.MongoClient(self.hostname, self.port)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.fs = gridfs.GridFS(self.db)

    def _close(self):
        self.client.close()

    def __del__(self):
        self._close()

    # def state_dict(self):
        # pass

    # Public methods: ---------------------------------------------------------

    def save(self, document):
        """Stores a dictionary or list of dictionaries as as a document in collection.
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

            # TODO: Only Variables created explicitly by the user (graph leaves)
            # support the deepcopy protocal at the moment... Thus, a RuntimeError
            # is raised when Variables not created by the users are saved.
            docCopy = copy.deepcopy(doc)

            # Make a list of any existing referenced gridfs files.
            try:
                self._old_tensor_ids = docCopy['_tensor_ids']
            except KeyError:
                self._old_tensor_ids = []

            self._new_tensor_ids = []

            # Replace tensors with either a new gridfs file or a reference to
            # the old gridfs file.
            docCopy = self._save_tensors(docCopy)
            docCopy['_tensor_ids'] = self._new_tensor_ids
            doc['_tensor_ids'] = self._new_tensor_ids

            # Cleanup any remaining gridfs files (these used to be pointed to by document, but no
            # longer match any tensor that was in the db.
            for id in self._old_tensor_ids:
                self.fs.delete(id)
            self._old_tensor_ids = []

            # Add insertion date field to every document.
            docCopy['insertion_date'] = datetime.datetime.now()
            doc['insertion_date'] = datetime.datetime.now()

            # Insert into the collection and restore full data into original
            # document object
            docCopy = self._dot_to_vbar(docCopy)
            new_id = self.collection.save(docCopy)
            doc['_id'] = new_id
            object_ids.append(new_id)

        return object_ids

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
        """Performs a search using the presented query.

        Args:
            query: dictionary of key-value pairs to use for querying the mongodb

        Returns:
            all_results: list of full documents from the collection
        """
        query = self._dot_to_vbar(query)
        results = self.collection.find(query)

        if get_tensors:
            all_results = [self._vbar_to_dot(
                self._load_tensor(doc)) for doc in results]
        else:
            all_results = [self._vbar_to_dot(doc) for doc in results]

        if all_results:
            if len(all_results) > 1:
                return all_results
            elif len(all_results) == 1:
                return all_results[0]
            else:
                return None
        else:
            return None

    def delete(self, object_id):
        """Deletes a specific document from the collection based on the objectId.
        Note that it first deletes all the gridFS files pointed to by ObjectIds
        within the document.

        Use with caution, clearly.

        Args:
            object_id: an id of an object in the database.
        """
        document_to_delete = self.collection.find_one({"_id": object_id})
        tensors_to_delete = document_to_delete['_tensor_ids']
        for tensor_id in tensors_to_delete:
            self.fs.delete(tensor_id)
        self.collection.remove(object_id)

    # Private methods ---------------------------------------------------------

    def _tensor_to_binary(self, tensor):
        """Utility method to turn an tensor/array into a BSON Binary string.

        Called by save_tensors.

        Args:
            tensor: tensor of arbitrary dimension.

        Returns:
            BSON Binary object a pickled tensor.
        """
        return Binary(pickle.dumps(tensor, protocol=2), subtype=128)

    def _binary_to_tensor(self, binary):
        """Utility method to turn a a pickled tensor string back into
        a tensor.

        Called by load_tensors.

        Args:
            binary: BSON Binary object a pickled tensor.

        Returns:
            Tensor of arbitrary dimension.
        """
        return pickle.loads(binary)

    def _dot_to_vbar(self, document):
        """Convert periods in dictionary keys to vertical bars (|)."""
        for (key, value) in document.items():
            if isinstance(value, dict):
                self._dot_to_vbar(value)
            else:
                new_key = key.replace('.', '|')
                document[new_key] = document.pop(key)
        return document

    def _vbar_to_dot(self, document):
        """Convert vertical bars (|) in dictionary keys to to periods."""
        for (key, value) in document.items():
            if isinstance(value, dict):
                self._dot_to_vbar(value)
            else:
                new_key = key.replace('|', '.')
                document[new_key] = document.pop(key)
        return document

    def _load_tensor(self, document):
        """Utility method to recurse through a document and gather all ObjectIds and
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
                if key == '_Variable_data':
                    document = torch.autograd.Variable(
                        self._binary_to_tensor(self.fs.get(value).read()))
                else:
                    document[key] = self._binary_to_tensor(
                        self.fs.get(value).read())
            elif isinstance(value, dict):
                document[key] = self._load_tensor(value)
        return document

    def _save_tensors(self, document):
        """Utility method to recurse through a document and replace all tensors
        and store them in the gridfs, replacing the actual tensors with references to the
        gridfs path.

        Called by save()

        Note that it modifies the document in place, although we return it, too

        Args:
            document: dictionary like-document, storable in mongodb.

        Returns:
            document: dictionary like-document, storable in mongodb.
        """
        for (key, value) in document.items():

            if isinstance(value, torch.autograd.Variable):
                value = {'_Variable_data': value.data}

            if isinstance(value, np.ndarray) or torch.is_tensor(value):
                data_BSON = self._tensor_to_binary(value)
                data_MD5 = hashlib.md5(data_BSON).hexdigest()

                # Does this tensor match the hash of anything in the object
                # already?
                match = False
                for tensor_id in self._old_tensor_ids:
                    print('Checking if {} is already in the db... '.format(tensor_id))
                    if data_MD5 == self.fs.get(tensor_id).md5:
                        match = True
                        # print('Tensor is already in the db. Replacing tensor with old OjbectId: {}'.format(tensor_id))
                        document[key] = tensor_id
                        self._old_tensor_ids.remove(tensor_id)
                        self._new_tensor_ids.append(tensor_id)
                if not match:
                    # print('Tensor is not in the db. Inserting new gridfs file...')
                    tensor_id = self.fs.put(self._tensor_to_binary(value))
                    document[key] = tensor_id
                    self._new_tensor_ids.append(tensor_id)

            elif isinstance(value, dict):
                document[key] = self._save_tensors(value)

            elif isinstance(value, np.number):
                if isinstance(value, np.integer):
                    document[key] = int(value)
                elif isinstance(value, np.inexact):
                    document[key] = float(value)

        return document
