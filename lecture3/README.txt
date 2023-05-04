To run this simulation you need access to a python interpreter.
A way to get started quickly is to install Anaconda
https://docs.anaconda.com/anaconda/
Anconda is good but will install a lot of things you don't need.


Alternatively you can set things up yourself...
You'll want to use the command prompt. Type 'dir <enter>' to list the contents of the present working directory.
Use the 'cd' command to change the working directory (folder).

To install Python on the computer download and install Python 3 (not via command prompt)
https://www.python.org

Set up a new virtual environment (in the command prompt):
>>Python -m venv <name of the new virtual environ>

activate the new environment
>>call <name of the new virtual environ>\scripts\activate.bat

install all the required packages
>> pip install <package name>

for the simualtion you will need the following three packages:
-numpy
-scipy
-matplotlib

to run the script from the command run the following from the containing folder
>> python 1Dsimulation.py
alterantively if you don't want to python to close after you've run the script run
>> python -i 1Dsimulation.py