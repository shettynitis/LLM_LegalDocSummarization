import time, numpy as np
MAX_MEAN_MS = 40

def test_latency(predict):
    t0 = time.time()
    out = predict("One‑sentence summary of clause 7.2:")
    dt  = (time.time() - t0)*1000
    mean = dt/len(out.split())
    assert mean < MAX_MEAN_MS, f"Mean {mean:.1f} ms > {MAX_MEAN_MS}"