## Setting up the project on Windows
```winget install Git.Git
winget install -e --id Microsoft.VisualStudioCode
winget install -e --id Python.Python.3.12

python3 -m venv ./venv
mkdir git
cd git

git clone https://github.com/shayakbanerjee99/blackscholes-with-markovian-switching.git
cd .\blackscholes-with-markovian-switching\
```

#### Switch to command prompt
```
venv\Scripts\activate.bat
pip install pandas
pip install matplotlib
pip install mplfinance
pip install numpy
pip install scipy



```
