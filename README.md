# NIROM

TODO: Add description of contents

## Installation: 
The easiest way of running this software is by using docker.
Based on the development image of the FEniCS project, we can start a docker container with all dependencies with the following commands:

### Initialize docker container
```
docker run -ti -v $PWD:/home/shared -w /home/shared --rm quay.io/fenicsproject/dev
```
### Install additional dependencies
For mesh generationt to work, we rely on gmsh, meshio and pygmsh
```
sudo apt-get update -qq && \
sudo apt-get install -y -qq libglu1 libxrender1 libxft2 libxinerama1 libxcursor1 && \
pip3 -q install --upgrade sympy --user && \
pip3 install gmsh --user && \
export HDF5_MPI="ON" && \
export CC=mpicc && \
export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/mpich/ && \
pip3 install --no-binary=h5py h5py meshio pygmsh --user && \
echo "alias gmsh=/usr/local/lib/python3.6/site-packages/gmsh-4.8.0-Linux64-sdk/bin/gmsh" >> ~/.bashrc && \
exec bash && \
pip3 install tqdm --user
```

#### GMSH alias
Sometimes gmsh is installed in `/usr/local/lib/python3.6/site-packages/gmsh-4.8.0-Linux64-sdk/bin/gmsh`
instead of `/usr/local/lib/python3.6/site-packages/gmsh-4.8.0-Linux64-sdk/bin/gmsh` and the alias should be adjusted accordingly.

#### Ubuntu 20.04
If you base your docker installation on the docker image `ubuntu:20.04` the following instructions should work:
```
apt-get update -qq && \
apt-get install -y -qq software-properties-common python3-pip libglu1 libxrender1 libxft2 libxinerama1 && \
add-apt-repository -y ppa:fenics-packages/fenics && \
apt install -y --no-install-recommends fenics && \
pip3 -q install --upgrade sympy && \
pip3 install gmsh && \
export HDF5_MPI="ON" && \ 
export CC=mpicc && \ 
export HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/openmpi/" && \
pip3 install --no-binary=h5py h5py meshio pygmsh && \
echo "alias gmsh=/usr/local/lib/python3.8/site-packages/gmsh-4.8.0-Linux64-sdk/bin/gmsh" >> ~/.bashrc && \
exec bash  && \
pip3 install tqdm
```

