#!/bin/bash
source ./cassper/bin/activate
ls "$1"/labels/ > "$1"/Trfiles.txt
./Train8.sh "$1"/ Fresh new InceptionV4 FRRN-B
