#!/usr/bin/env bash


conda create -n st_test1 python==3.7
conda activate st_test1

# Test 1: OK
pip install dist/SummerTime-0.1-py3-none-any.whl
pip uninstall SummerTime

# Test 2: OK
pip3 install dist/SummerTime-0.1-py3-none-any.whl


conda create -n st_test2 python==3.9
conda activate st_test2

# Test 3: OK
pip install dist/SummerTime-0.1-py3-none-any.whl
pip uninstall SummerTime

# Test 4: OK
pip3 install dist/SummerTime-0.1-py3-none-any.whl

# Test 5: Wrong Answer. What is the entry for our package?
#>>> import SummerTime
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#ModuleNotFoundError: No module named 'SummerTime'
