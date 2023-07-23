#!/bin/bash

# check os
echo "check os"
is_Windows=false
uNames=`uname -s`
osName=${uNames: 0: 4}
if [ "$osName" == "Darw" ] # Darwin
then
	echo "Mac OS X"
elif [ "$osName" == "Linu" ] # Linux
then
	echo "GNU/Linux"
elif [ "$osName" == "MING" ] # MINGW, windows, git-bash
then 
    is_Windows=true
	echo "Windows, git-bash" 
else
	echo "unknown os"
fi

# check if twine is installed
echo "check if twine is installed"
twine --version
if [ $? -ne 0 ]
then
    echo "twine is not installed, will install it first! "
    # install twine
    pip install twine
fi

# delete old dist
if $is_Windows
then
    echo "delete old dist on Windows, git-bash"
    rmdir dist build zhei.egg-info /s /q
else
    echo "delete old dist on Mac OS X or GNU/Linux"
    rm -rf dist build *.egg-info
fi


# pip install twine
python setup.py build sdist
twine upload dist/*


# delete dist
if $is_Windows
then
    echo "delete dist on Windows, git-bash"
    rmdir dist build zhei.egg-info /s /q
else
    echo "delete dist on Mac OS X or GNU/Linux"
    rm -rf dist build *.egg-info
fi