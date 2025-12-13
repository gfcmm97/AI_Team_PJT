import numpy as np
from threshold_gate import is_reliable

s = np.random.rand(500)
ok, diag = is_reliable(s)

print("OK:", ok)
print("Diagnostics:")
for k, v in diag.items():
    print(k, v)
