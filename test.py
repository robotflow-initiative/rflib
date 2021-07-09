from rflib.utils import Registry

if __name__ == '__main__':
    TEST = Registry('test')
    @TEST.register_module()
    class TestNet:
        pass