import inspect
from abc import ABCMeta, abstractmethod
from urllib.request import urlopen


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass

class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf


class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def get(self, filepath):
        value_buf = urlopen(filepath).read()
        return value_buf

    def get_text(self, filepath):
        value_buf = urlopen(filepath).read()
        return value_buf.decode('utf-8')


class FileClient:
    """A general file client to access files in different backend.

    The client loads a file or text in a specified backend from its path
    and return it as a binary file. it can also register other backend
    accessor with a given name and backend class.

    Attributes:
        backend (str): The storage backend type. Options are "disk", "ceph",
            "memcached", "lmdb" and "http".
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'disk': HardDiskBackend,
        'http': HTTPBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    @classmethod
    def _register_backend(cls, name, backend, force=False):
        if not isinstance(name, str):
            raise TypeError('the backend name should be a string, '
                            f'but got {type(name)}')
        if not inspect.isclass(backend):
            raise TypeError(
                f'backend should be a class but got {type(backend)}')
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f'backend {backend} is not a subclass of BaseStorageBackend')
        if not force and name in cls._backends:
            raise KeyError(
                f'{name} is already registered as a storage backend, '
                'add "force=True" if you want to override it')

        cls._backends[name] = backend

    @classmethod
    def register_backend(cls, name, backend=None, force=False):
        """Register a backend to FileClient.

        This method can be used as a normal class method or a decorator.

        .. code-block:: python

            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

            FileClient.register_backend('new', NewBackend)

        or

        .. code-block:: python

            @FileClient.register_backend('new')
            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

        Args:
            name (str): The name of the registered backend.
            backend (class, optional): The backend class to be registered,
                which must be a subclass of :class:`BaseStorageBackend`.
                When this method is used as a decorator, backend is None.
                Defaults to None.
            force (bool, optional): Whether to override the backend if the name
                has already been registered. Defaults to False.
        """
        if backend is not None:
            cls._register_backend(name, backend, force=force)
            return

        def _register(backend_cls):
            cls._register_backend(name, backend_cls, force=force)
            return backend_cls

        return _register

    def get(self, filepath):
        return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)
