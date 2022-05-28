PipEnv Installation
https://docs.python-guide.org/dev/virtualenvs/

Then you can run this script using pipenv run:

$ pipenv run python main.py


Install virtualenv via pip:

$ pip install virtualenv
Test your installation:

$ virtualenv --version


Create a virtual environment for a project:
$ cd project_folder
$ virtualenv venv


You can also use the Python interpreter of your choice (like python2.7).

$ virtualenv -p /usr/bin/python2.7 venv
or change the interpreter globally with an env variable in ~/.bashrc:

$ export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7
To begin using the virtual environment, it needs to be activated:
$ source venv/bin/activate

You can also use the Python interpreter of your choice (like python2.7).

$ virtualenv -p /usr/bin/python2.7 venv
or change the interpreter globally with an env variable in ~/.bashrc:

$ export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7
To begin using the virtual environment, it needs to be activated:
$ source venv/bin/activate

Tensorflow installation
https://www.youtube.com/watch?v=GNRg2P8Vqqs


https://github.com/PINTO0309/Tensorflow-bin/#usage:
Make your project directory:
cd Desktop
mkdir tf_pi
cd tf_pi

Make a virtual environment (I'm assuming you have Python 3):
python3 -m pip install virtualenv
virtualenv env
source env/bin/activate

tensorflow-2.5.0-cp37-none-linux_armv7l.whl

Run the commands based on https://github.com/PINTO0309/Tensorfl...
https://github.com/PINTO0309/Tensorflow-bin/#usage:
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
python3 -m pip install keras_applications==1.0.8 --no-deps
python3 -m pip install keras_preprocessing==1.1.0 --no-deps
#python3 -m pip install h5py==2.9.0
python3 -m pip install h5py==2.9.0
sudo apt-get install -y openmpi-bin libopenmpi-dev
sudo apt-get install -y libatlas-base-dev
python3 -m pip install -U six wheel mock

Pick a tensorflow release from https://github.com/lhelontra/tensorfl... (I picked 2.0.0): 
https://github.com/lhelontra/tensorflow-on-arm/releases
wget https://github.com/lhelontra/tensorflow-2.0.0-cp37-none-linux_armv7l.whl
python3 -m pip uninstall tensorflow
#python3 -m pip install tensorflow-2.0.0-cp37-none-linux_armv7l.whl

wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl

#python3 -m pip install tensorflow-2.5.0-cp37-none-linux_armv7l.whl
python3 -m pip install tensorflow-2.4.0-cp37-none-linux_armv7l.whl


RESTART YOUR TERMINAL

Reactivate your virtual environment:
cd Desktop
cd tf_pi
source env/bin/activate

Test:
Open a python interpreter by executing: python3 
import tensorflow
tensorflow.__version__

This should have no errors and output: 2.0.0


Example of Python 3.x + Tensorflow v2 series


$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython3 libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev
$ sudo pip3 install pip --upgrade
$ sudo pip3 install keras_applications==1.0.8 --no-deps
$ sudo pip3 install keras_preprocessing==1.1.0 --no-deps
$ sudo pip3 install numpy==1.22.1
$ sudo pip3 install h5py==3.1.0
$ sudo pip3 install pybind11
$ pip3 install -U --user six wheel mock
$ wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/main/tensorflow-2.8.0-cp39-none-linux_aarch64_numpy1221_download.sh"
$ sudo chmod +x tensorflow-2.8.0-cp39-none-linux_aarch64_numpy1221_download.sh
$ ./tensorflow-2.8.0-cp39-none-linux_aarch64_numpy1221_download.sh
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-2.8.0-cp39-none-linux_aarch64.whl

【Required】 Restart the terminal.




Final steps of Installation - https://www.youtube.com/watch?v=QLZWQlg-Pk0

INSTRUCTIONS AND TIMESTAMPS:
0:00 Intro
0:09   1. Find your .whl file
Check architecture: uname -m
Check OS: cat /etc/os-release
    sudo apt update
    sudo apt full-upgrade
Check python version: python -V
Check available .whl's here https://github.com/PINTO0309/Tensorfl...
https://github.com/PINTO0309/Tensorflow-bin/tree/main/previous_versions


Be sure to match your python version and architecture

**Use pyenv if you need a different python version - LINK TO PYENV TUTORIAL: https://youtu.be/QdlopCUuXxw

2:45   2. Make your project directory:
cd Desktop
mkdir project
cd project

3:00   3. Make a virtual environment:
python3 -m pip install virtualenv
python3 -m virtualenv env
source env/bin/activate

3:34   4. Run the commands from https://github.com/PINTO0309/Tensorfl...
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython3 libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev
pip install -U wheel mock six

3:58   5. Select the .whl from https://github.com/PINTO0309/Tensorfl...
Select "view raw" then copy the URL
https://github.com/PINTO0309/Tensorflow-bin
Run:
wget [Raw file URL]
sudo chmod +x [Raw file URL]
./[Tensorflow file]

wget 'https://github.com/PINTO0309/Tensorflow-bin/releases/download/v2.8.0/tensorflow-2.8.0-cp39-none-linux_aarch64.whl'
sudo pip uninstall tensorflow
pip uninstall tensorflow
pip install  tensorflow-[Your version here].whl

6:00   6. Restart the shell
exec $SHELL

6:11   7. Reactivate your virtual environment:
cd Desktop
cd project
source env/bin/activate

6:24   8. Test:
python
import tensorflow
tensorflow.__version__
quit()

6:55   9. (optional) If there's an hdf5 warning run this command:
This is from: https://docs.h5py.org/en/stable/build...
pip uninstall h5py
HDF5_VERSION=[Desired version] pip install --no-binary=h5py h5py==3.1.0


