#!/bin/bash

# all arguments except first, to compile multiple files
FILES=${@: 1}
OUT=$(uuid).out

g++ -std=c++20 -fmodules-ts $FILES -o $OUT
OUTPUT=$(./$OUT)

rm $OUT

echo "${OUTPUT}"