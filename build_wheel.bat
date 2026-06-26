@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
where cl.exe
python setup.py bdist_wheel --dist-dir dist/
