import numpy as np

y = np.zeros(0)
y2 = np.append(y, [None])
print(y2)


while y2 is None:
    print("yes")