# Experiment 1: Setting up the Spyder IDE and Running a Python Program

> **File:** `exp1.py`  |  **Tools:** Python 3, Spyder IDE, pip / venv

## 🎯 Aim

To set up the Spyder IDE environment and to execute a Python program in it.

## 📘 Theory

Spyder (Scientific Python Development Environment) is an open-source IDE written in Python for data science work. Its window is divided into an editor, an IPython console in which the code runs, a Variable Explorer that shows every variable and its value, a Plots pane and a Help pane. A file is run with the Run button or F5, a selected part with F9, and the debugger can be used to step through the code line by line.

Spyder should be installed inside the environment that holds the libraries of the project. First a virtual environment is created with python -m venv, then pip install spyder (or conda install spyder in Anaconda) is run, and the IDE is started with the command spyder. In Settings the interpreter is pointed to the environment, so that the console imports the same TensorFlow, PyTorch and Gymnasium that the experiments use.

The program of this experiment checks that the setup works. It prints the Python version, the operating system, the number of CPU cores and whether a virtual environment is active, compares plain Python with NumPy, analyses a string, computes Fibonacci numbers and reads and writes a file. Its output below was captured from the terminal.

- **IDE**: Program that combines an editor, a console and a debugger.
- **Virtual environment**: Isolated Python installation with its own packages.
- **IPython console**: Interactive console of Spyder in which the code is executed.
- **Variable Explorer**: Spyder pane that lists all variables with type, size and value.

## 🪜 Algorithm / Steps

1. Create a virtual environment and activate it.
2. Install Spyder with pip install spyder and start it with the command spyder.
3. Select the interpreter of the environment in the Spyder preferences.
4. Write exp1.py: print the system details, time a sum of squares in Python and in NumPy, analyse a string, compute Fibonacci numbers and write and read a file.
5. Run the file with F5 (here: python3 exp1.py) and read the console.

## 💻 Code

```python
import os
import platform
import sys
import timeit
from collections import Counter

import numpy as np

print("Python version  :", platform.python_version(), f"({platform.python_implementation()})")
print("Operating system:", platform.platform(terse=True))
print("CPU cores       :", os.cpu_count())
print("virtual env     :", "yes" if sys.prefix != sys.base_prefix else "no", "-", os.path.basename(sys.prefix))
print("modules loaded  :", len(sys.modules))

loop = timeit.timeit("sum(i * i for i in range(10000))", number=200)
vec = timeit.timeit("np.sum(np.arange(10000) ** 2)", globals={"np": np}, number=200)
print(f"\n200 runs of a sum of squares: plain Python {loop * 1000:.1f} ms, NumPy {vec * 1000:.1f} ms")
print(f"NumPy is about {loop / vec:.0f} times faster for this task")

course = "reinforcement learning and deep learning lab"
letters = Counter(c for c in course if c.isalpha())
print("\nmost common letters in the course title:", letters.most_common(4))
print("title case      :", course.title())
print("words by length :", sorted(course.split(), key=len, reverse=True))


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


print("\nfibonacci numbers 10 to 15 :", [fib(n) for n in range(10, 16)])
print("fibonacci(90)              :", fib(90))

with open("/tmp/rldl_demo.txt", "w") as fh:
    fh.write("line one\nline two\nline three\n")
with open("/tmp/rldl_demo.txt") as fh:
    rows = fh.read().splitlines()
print("\nfile written and read back:", rows, "->", len(rows), "lines")
```

## 🖥️ Output

```text
Python version  : 3.13.5 (CPython)
Operating system: macOS-26.3.1
CPU cores       : 10
virtual env     : yes - venv
modules loaded  : 448

200 runs of a sum of squares: plain Python 51.9 ms, NumPy 1.8 ms
NumPy is about 29 times faster for this task

most common letters in the course title: [('e', 7), ('n', 7), ('r', 4), ('a', 4)]
title case      : Reinforcement Learning And Deep Learning Lab
words by length : ['reinforcement', 'learning', 'learning', 'deep', 'and', 'lab']

fibonacci numbers 10 to 15 : [55, 89, 144, 233, 377, 610]
fibonacci(90)              : 2880067194370816120

file written and read back: ['line one', 'line two', 'line three'] -> 3 lines
```

## ✅ Result

The program reported Python 3.13 on macOS, with a virtual environment active. A sum of squares repeated 200 times took about 52 ms in plain Python but about 1.8 ms in NumPy, around 29 times faster. The letters 'e' and 'n' are the commonest in the course title (7 times each), fibonacci(90) = 2880067194370816120 was computed exactly, and the file with three lines was written and read back correctly.

## 🎤 Viva Questions

**Q. What is Spyder?**  
A. A free Python IDE for scientific work with an editor, a console and a variable explorer.

**Q. How is Spyder installed?**  
A. With pip install spyder or conda install spyder inside the Python environment, and started with the command spyder.

**Q. Which key runs a file in Spyder?**  
A. F5.

**Q. What is the Variable Explorer?**  
A. A pane that lists all variables with their type, size and value.

**Q. What is a virtual environment?**  
A. An isolated Python installation with its own packages.

---
[⬅ Back to index](README.md)
