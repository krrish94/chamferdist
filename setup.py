from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

package_name = 'chamferdist'
version = '0.3.0'
requirements = [
    'Cython',
    'torch>1.1.0',
]
long_description = 'A pytorch module to compute Chamfer distance \
                    between two point sets (pointclouds).'

setup(
    name='chamferdist',
    version=version,
    description='Pytorch Chamfer distance', 
    long_description=long_description,
    requirements=requirements,
    ext_modules=[
        CUDAExtension('chamferdistcuda', [
            'chamferdist/chamfer_cuda.cpp',
            'chamferdist/chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
