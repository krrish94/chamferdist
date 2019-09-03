from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

long_description = 'A pytorch module to compute Chamfer distance \
                    between two point sets (pointclouds).'

requirements = [
    'Cython',
    'torch>1.1.0',
]

setup(
    name='chamferdist',
    version='0.2.0',
    description='Pytorch Chamfer distance', 
    long_description=long_description,
    ext_modules=[
        CUDAExtension('chamferdistcuda', [
            'chamferdist/chamfer_cuda.cpp',
            'chamferdist/chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
