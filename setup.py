from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamferdist',
    ext_modules=[
        CUDAExtension('chamferdistcuda', [
            'chamferdist/chamfer_cuda.cpp',
            'chamferdist/chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
