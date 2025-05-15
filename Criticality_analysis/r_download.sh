#!/bin/bash
#

# make d2_code folder
mkdir d2_code

# change to d2_code folder
cd d2_code

# download d2 code
wget https://figshare.com/ndownloader/articles/25927081/versions/1
# latest version
# wget https://figshare.com/ndownloader/articles/25927081/versions/2

# mv name from 1 to a zip file and extract code
mv 1 25927081.zip
unzip 25927081.zip
