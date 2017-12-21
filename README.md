# face-recognition-service
Face recognition online service, allow user training it.

## Installation 

```
brew install cmake
```
```
brew install boost-python --with-python3 --without-python
```
```
git clone https://github.com/davisking/dlib.git
```
```
cd dlib
mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1; cmake --build .
```

## dlib C++ library 
Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems.
https://github.com/davisking/dlib
To compile Boost.Python yourself download boost from boost.org and then go into the boost root folder

```
./bootstrap.sh --with-libraries=python
./b2
sudo ./b2 install
```

## Flask 

Use flask as Python framework build api service. Api http://flask.pocoo.org/docs/0.12/api

## Database
we can use any database to store users info. in this project i just use sqlite3 support by Python default.
## Face recognition Python Library


