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
