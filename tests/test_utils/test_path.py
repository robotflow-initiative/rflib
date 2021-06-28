# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import pytest

import rflib


def test_is_filepath():
    assert rflib.is_filepath(__file__)
    assert rflib.is_filepath('abc')
    assert rflib.is_filepath(Path('/etc'))
    assert not rflib.is_filepath(0)


def test_fopen():
    assert hasattr(rflib.fopen(__file__), 'read')
    assert hasattr(rflib.fopen(Path(__file__)), 'read')


def test_check_file_exist():
    rflib.check_file_exist(__file__)
    with pytest.raises(FileNotFoundError):
        rflib.check_file_exist('no_such_file.txt')


def test_scandir():
    folder = osp.join(osp.dirname(osp.dirname(__file__)), 'data/for_scan')
    filenames = ['a.bin', '1.txt', '2.txt', '1.json', '2.json']
    assert set(rflib.scandir(folder)) == set(filenames)
    assert set(rflib.scandir(Path(folder))) == set(filenames)
    assert set(rflib.scandir(folder, '.txt')) == set(
        [filename for filename in filenames if filename.endswith('.txt')])
    assert set(rflib.scandir(folder, ('.json', '.txt'))) == set([
        filename for filename in filenames
        if filename.endswith(('.txt', '.json'))
    ])
    assert set(rflib.scandir(folder, '.png')) == set()

    filenames_recursive = [
        'a.bin', '1.txt', '2.txt', '1.json', '2.json', 'sub/1.json',
        'sub/1.txt'
    ]
    assert set(rflib.scandir(folder,
                            recursive=True)) == set(filenames_recursive)
    assert set(rflib.scandir(Path(folder),
                            recursive=True)) == set(filenames_recursive)
    assert set(rflib.scandir(folder, '.txt', recursive=True)) == set([
        filename for filename in filenames_recursive
        if filename.endswith('.txt')
    ])
    with pytest.raises(TypeError):
        list(rflib.scandir(123))
    with pytest.raises(TypeError):
        list(rflib.scandir(folder, 111))
