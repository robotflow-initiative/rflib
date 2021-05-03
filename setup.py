import glob
import os
import re
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup
import torch

try:
    from torch.utils.cpp_extension import BuildExtension
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')


def get_version():
    version_file = 'rflib/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


install_requires = parse_requirements()


def get_extensions():
    extensions = []

    if os.getenv('RFLIB_WITH_TRT', '0') != '0':
        ext_name = 'rflib._ext_trt'
        from torch.utils.cpp_extension import include_paths, library_paths
        library_dirs = []
        libraries = []
        include_dirs = []
        tensorrt_path = os.getenv('TENSORRT_DIR', '0')
        tensorrt_lib_path = glob.glob(
            os.path.join(tensorrt_path, 'targets', '*', 'lib'))[0]
        library_dirs += [tensorrt_lib_path]
        libraries += ['nvinfer', 'nvparsers', 'nvinfer_plugin']
        libraries += ['cudart']
        kwargs = {}
        define_macros = []
        extra_compile_args = {'cxx': []}

        include_path = os.path.abspath('./rflib/ops/csrc')
        include_trt_path = os.path.abspath('./rflib/ops/csrc/tensorrt')
        include_dirs.append(include_path)
        include_dirs.append(include_trt_path)
        include_dirs.append(os.path.join(tensorrt_path, 'include'))
        include_dirs += include_paths(cuda=True)

        op_files = glob.glob('./rflib/ops/csrc/tensorrt/plugins/*')
        define_macros += [('RFLIB_WITH_CUDA', None)]
        define_macros += [('RFLIB_WITH_TRT', None)]
        cuda_args = os.getenv('RFLIB_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        library_dirs += library_paths(cuda=True)

        kwargs['library_dirs'] = library_dirs
        kwargs['libraries'] = libraries

        from setuptools import Extension
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            language='c++',
            library_dirs=library_dirs,
            libraries=libraries)
        extensions.append(ext_ops)

    if os.getenv('RFLIB_WITH_OPS', '0') == '0':
        return extensions

    ext_name = 'rflib._ext'
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('RFLIB_WITH_CUDA', None)]
        cuda_args = os.getenv('RFLIB_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        op_files = glob.glob('./rflib/ops/csrc/pytorch/*')
        extension = CUDAExtension
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob('./rflib/ops/csrc/pytorch/*.cpp')
        extension = CppExtension

    include_path = os.path.abspath('./rflib/ops/csrc')
    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)

    if os.getenv('RFLIB_WITH_ORT', '0') != '0':
        ext_name = 'rflib._ext_ort'
        from torch.utils.cpp_extension import library_paths, include_paths
        import onnxruntime
        library_dirs = []
        libraries = []
        include_dirs = []
        ort_path = os.getenv('ONNXRUNTIME_DIR', '0')
        library_dirs += [os.path.join(ort_path, 'lib')]
        libraries.append('onnxruntime')
        kwargs = {}
        define_macros = []
        extra_compile_args = {'cxx': []}

        include_path = os.path.abspath('./rflib/ops/csrc/onnxruntime')
        include_dirs.append(include_path)
        include_dirs.append(os.path.join(ort_path, 'include'))

        op_files = glob.glob('./rflib/ops/csrc/onnxruntime/cpu/*')
        if onnxruntime.get_device() == 'GPU' or os.getenv('FORCE_CUDA',
                                                          '0') == '1':
            define_macros += [('RFLIB_WITH_CUDA', None)]
            cuda_args = os.getenv('RFLIB_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files += glob.glob('./rflib/ops/csrc/onnxruntime/gpu/*')
            include_dirs += include_paths(cuda=True)
            library_dirs += library_paths(cuda=True)
        else:
            include_dirs += include_paths(cuda=False)
            library_dirs += library_paths(cuda=False)

        kwargs['library_dirs'] = library_dirs
        kwargs['libraries'] = libraries

        from setuptools import Extension
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            language='c++',
            library_dirs=library_dirs,
            libraries=libraries)
        extensions.append(ext_ops)

    return extensions


setup(
    name='rflib',
    version=get_version(),
    description='RFVision Foundation',
    keywords='robot vision',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
    ],
    url='https://github.com/WenqiangX/rflib',
    author='RFLib Authors',
    author_email='vinjohn@sjtu.edu.cn',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    zip_safe=False)
