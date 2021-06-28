# Copyright (c) Open-MMLab. All rights reserved.
import pytest

import rflib


def test_iter_cast():
    assert rflib.list_cast([1, 2, 3], int) == [1, 2, 3]
    assert rflib.list_cast(['1.1', 2, '3'], float) == [1.1, 2.0, 3.0]
    assert rflib.list_cast([1, 2, 3], str) == ['1', '2', '3']
    assert rflib.tuple_cast((1, 2, 3), str) == ('1', '2', '3')
    assert next(rflib.iter_cast([1, 2, 3], str)) == '1'
    with pytest.raises(TypeError):
        rflib.iter_cast([1, 2, 3], '')
    with pytest.raises(TypeError):
        rflib.iter_cast(1, str)


def test_is_seq_of():
    assert rflib.is_seq_of([1.0, 2.0, 3.0], float)
    assert rflib.is_seq_of([(1, ), (2, ), (3, )], tuple)
    assert rflib.is_seq_of((1.0, 2.0, 3.0), float)
    assert rflib.is_list_of([1.0, 2.0, 3.0], float)
    assert not rflib.is_seq_of((1.0, 2.0, 3.0), float, seq_type=list)
    assert not rflib.is_tuple_of([1.0, 2.0, 3.0], float)
    assert not rflib.is_seq_of([1.0, 2, 3], int)
    assert not rflib.is_seq_of((1.0, 2, 3), int)


def test_slice_list():
    in_list = [1, 2, 3, 4, 5, 6]
    assert rflib.slice_list(in_list, [1, 2, 3]) == [[1], [2, 3], [4, 5, 6]]
    assert rflib.slice_list(in_list, [len(in_list)]) == [in_list]
    with pytest.raises(TypeError):
        rflib.slice_list(in_list, 2.0)
    with pytest.raises(ValueError):
        rflib.slice_list(in_list, [1, 2])


def test_concat_list():
    assert rflib.concat_list([[1, 2]]) == [1, 2]
    assert rflib.concat_list([[1, 2], [3, 4, 5], [6]]) == [1, 2, 3, 4, 5, 6]


def test_requires_package(capsys):

    @rflib.requires_package('nnn')
    def func_a():
        pass

    @rflib.requires_package(['numpy', 'n1', 'n2'])
    def func_b():
        pass

    @rflib.requires_package('numpy')
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "nnn" are required in method "func_a" but '
                   'not found, please install them first.\n')

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == (
        'Prerequisites "n1, n2" are required in method "func_b" but not found,'
        ' please install them first.\n')

    assert func_c() == 1


def test_requires_executable(capsys):

    @rflib.requires_executable('nnn')
    def func_a():
        pass

    @rflib.requires_executable(['ls', 'n1', 'n2'])
    def func_b():
        pass

    @rflib.requires_executable('mv')
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "nnn" are required in method "func_a" but '
                   'not found, please install them first.\n')

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == (
        'Prerequisites "n1, n2" are required in method "func_b" but not found,'
        ' please install them first.\n')

    assert func_c() == 1


def test_import_modules_from_strings():
    # multiple imports
    import os.path as osp_
    import sys as sys_
    osp, sys = rflib.import_modules_from_strings(['os.path', 'sys'])
    assert osp == osp_
    assert sys == sys_

    # single imports
    osp = rflib.import_modules_from_strings('os.path')
    assert osp == osp_
    # No imports
    assert rflib.import_modules_from_strings(None) is None
    assert rflib.import_modules_from_strings([]) is None
    assert rflib.import_modules_from_strings('') is None
    # Unsupported types
    with pytest.raises(TypeError):
        rflib.import_modules_from_strings(1)
    with pytest.raises(TypeError):
        rflib.import_modules_from_strings([1])
    # Failed imports
    with pytest.raises(ImportError):
        rflib.import_modules_from_strings('_not_implemented_module')
    with pytest.warns(UserWarning):
        imported = rflib.import_modules_from_strings(
            '_not_implemented_module', allow_failed_imports=True)
        assert imported is None
    with pytest.warns(UserWarning):
        imported = rflib.import_modules_from_strings(
            ['os.path', '_not_implemented'], allow_failed_imports=True)
        assert imported[0] == osp
        assert imported[1] is None
