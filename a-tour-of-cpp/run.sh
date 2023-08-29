#!/bin/sh

FILE=$1
OUT=$(uuid).out

#g++ -std=c++20 -fmodules-ts -xc++-system-header std
g++ -std=c++20 -fmodules-ts $FILE -o $OUT
OUTPUT=$(./$OUT)

rm $OUT

echo "${OUTPUT}"