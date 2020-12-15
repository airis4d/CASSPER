#!/bin/bash
sudo apt-get install python3-pip
sudo pip3 install virtualenv
python3 -m venv cassper
source cassper/bin/activate
pip install -r requirements.txt

