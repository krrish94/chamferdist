# chamferdist: PyTorch Chamfer distance

> **NOTE**: This is a borrowed implementation from the elegant [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet/tree/master/extension) GitHub repo, and all I did was to simply package it.

A simple example Pytorch module to compute Chamfer distance between two pointclouds. Basically a wrapper around the elegant implementation from [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet/tree/master/extension).

### Installation

You can install the package using `pip`.

```
pip install chamferdist
```

### Building from source

In your favourite python/conda virtual environment, execute the following commands. 

> **NOTE**: This assumes you have PyTorch installed already (preferably, > 1.1.0; untested for earlier releases).

```python
python setup.py install
```

### Running (example)

That's it! You're now ready to go. Here's a quick guide to using the package. Fire up a terminal. Import the package.

```python
>>> import torch
>>> from chamferdist import ChamferDist
```

Create two random pointclouds. Each pointcloud is a 3D tensor with dimensions `batchsize` x `number of points` x `number of dimensions`.

```python
>>> pc1 = torch.randn(1, 100, 3).cuda().contiguous()
>>> pc2 = torch.randn(1, 50, 3).cuda().contiguous()
```

Initialize a `ChamferDist` object.
```python
>>> chamferDist = ChamferDistance()
```

Now, compute Chamfer distance.
```python
>>> dist1, dist2, idx1, idx2 = chamferDist(pc1, pc2)
>>> print(dist1.shape, dist2.shape, idx1.shape, idx2.shape)
```

Here, `dist1` is the Chamfer distance between `pc1` and `pc2`. Note that Chamfer distance is not bidirectional (and, in stricter parlance, it is not a _distance metric_). The Chamfer distance in the other direction, i.e., `pc2` to `pc1` is stored in the variable `dist2`.

For each point in `pc1`, `idx1` stores the index of the closest point in `pc2`. For each point in `pc2`, `idx2` stores the index of the closest point in `pc1`.


### Citing (the original implementation, AtlasNet)

If you find this work useful, you might want to cite the *original* implementation from which this codebase was borrowed (stolen!) - AtlasNet.

```
@inproceedings{groueix2018,
    title={{AtlasNet: A Papier-M\^ach\'e Approach to Learning 3D Surface Generation}},
    author={Groueix, Thibault and Fisher, Matthew and Kim, Vladimir G. and Russell, Bryan and Aubry, Mathieu},
    booktitle={Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```
