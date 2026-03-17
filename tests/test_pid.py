# tests/test_pid.py
import time
import math
from metrics.pid import PIDController
from metrics.collector import LatencyWindow, AnomalyDetector

passed = failed = 0

def run(name, fn):
    global passed, failed
    t0 = time.perf_counter()
    try:
        fn()
        print(f"  PASS  {name:<55} {(time.perf_counter()-t0)*1000:.2f}ms")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name:<55} --> {e}")
        failed += 1

# PID tests 

def test_pid_positive_error_gives_positive_output():
    """Queue above target  output should be positive (scale up)."""
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, target=0.0)
    out = pid.update(measured=5.0)   # queue depth 5, target 0
    assert out > 0, f"expected positive output, got {out}"
    print(f"        measured=5 target=0  output={out:.3f} (scale up signal)")

def test_pid_zero_error_gives_zero_output():
    """Queue at target  output should be ~0."""
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, target=3.0)
    out = pid.update(measured=3.0)
    assert abs(out) < 0.01, f"expected ~0, got {out}"
    print(f"        measured=target=3  output={out:.3f}")

def test_pid_integral_accumulates():
    """Sustained error should cause integral to grow over time."""
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, target=0.0,
                        output_max=100.0, integral_max=100.0)
    outputs = []
    for _ in range(5):
        time.sleep(0.01)
        outputs.append(pid.update(measured=1.0))
    assert outputs[-1] > outputs[0], \
        f"integral should grow: {outputs}"
    print(f"        integral outputs over 5 ticks: "
          f"{[round(o,3) for o in outputs]}")

def test_pid_derivative_dampens():
    """Error improving fast  derivative term should reduce output."""
    pid = PIDController(kp=0.0, ki=0.0, kd=1.0, target=0.0,
                        output_min=-100.0, output_max=100.0)
    time.sleep(0.01)
    out1 = pid.update(measured=10.0)   # large error
    time.sleep(0.01)
    out2 = pid.update(measured=2.0)    # error improving fast  negative derivative
    print(f"        tick1 (error=10): {out1:.3f}  tick2 (error=2): {out2:.3f}")
    assert out2 < out1, "derivative should dampen when error is reducing"

def test_pid_output_clamped():
    """Output must never exceed [output_min, output_max]."""
    pid = PIDController(kp=10.0, ki=0.0, kd=0.0, target=0.0,
                        output_min=-2.0, output_max=2.0)
    out = pid.update(measured=1000.0)
    assert out == 2.0, f"expected clamped to 2.0, got {out}"
    out2 = pid.update(measured=-1000.0)
    assert out2 == -2.0, f"expected clamped to -2.0, got {out2}"
    print(f"        large error clamped correctly to +/-2.0")

def test_pid_anti_windup():
    """Integral must not grow unbounded when output is saturated."""
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, target=0.0,
                        output_max=2.0, integral_max=5.0)
    for _ in range(100):
        time.sleep(0.001)
        pid.update(measured=100.0)
    assert abs(pid._integral) <= 5.0, \
        f"integral exceeded anti-windup limit: {pid._integral}"
    print(f"        integral clamped at {pid._integral:.3f} "
          f"(limit=5.0) after 100 ticks")

def test_pid_convergence():
    """
    Simulate a simple first-order system and verify the PID drives
    queue depth toward target.
    System model: queue_depth(t+1) = queue_depth(t) - u(t) * 0.5
    """
    pid = PIDController(kp=0.8, ki=0.1, kd=0.05, target=0.0,
                        output_min=-5.0, output_max=5.0)
    queue = 10.0   # start with deep queue
    history = [queue]

    for _ in range(20):
        time.sleep(0.01)
        u = pid.update(queue)
        queue = max(0.0, queue - u * 0.5)   # simple system model
        history.append(round(queue, 2))

    print(f"        convergence: {history}")
    assert queue < 2.0, \
        f"PID failed to converge: final queue={queue:.2f}"

# latency window tests 

def test_latency_window_percentiles():
    w = LatencyWindow(maxlen=100)
    for v in range(1, 101):
        w.record(float(v))
    s = w.stats()
    assert s["n"]   == 100
    assert s["min"] == 1.0
    assert s["max"] == 100.0
    assert 94 <= s["p95"] <= 96
    print(f"        window stats: {s}")

def test_latency_window_rolling():
    """Old values should be evicted when window is full."""
    w = LatencyWindow(maxlen=5)
    for v in range(10):
        w.record(float(v))
    s = w.stats()
    assert s["min"] == 5.0, f"oldest values not evicted: min={s['min']}"
    print(f"        rolling eviction: min={s['min']} (expected 5.0)")

# anomaly detector tests 

def test_anomaly_not_triggered_on_normal():
    det = AnomalyDetector(window=50, threshold=3.0)
    for _ in range(50):
        det.check(100.0 + (hash(str(_)) % 10 - 5))  # values near 100 +/- 5
    is_anom, z = det.check(103.0)
    assert not is_anom, f"normal value flagged as anomaly: z={z}"
    print(f"        normal value z={z:.2f}  not flagged")

def test_anomaly_triggered_on_spike():
    det = AnomalyDetector(window=50, threshold=3.0)
    for i in range(50):
        det.check(100.0)   # stable baseline
    is_anom, z = det.check(500.0)   # massive spike
    assert is_anom, f"spike not detected: z={z}"
    print(f"        spike z={z:.2f}  correctly flagged as anomaly")

def test_anomaly_count_increments():
    det = AnomalyDetector(window=30, threshold=2.0)
    for _ in range(30):
        det.check(10.0)
    before = det.anomaly_count
    det.check(100.0)
    assert det.anomaly_count == before + 1
    print(f"        anomaly_count incremented to {det.anomaly_count}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  M5 test suite  PID + metrics")
    print("=" * 60)
    run("positive error  positive output (scale up)",   test_pid_positive_error_gives_positive_output)
    run("zero error  zero output",                      test_pid_zero_error_gives_zero_output)
    run("integral accumulates over time",                test_pid_integral_accumulates)
    run("derivative dampens on improving error",         test_pid_derivative_dampens)
    run("output clamped to [min, max]",                  test_pid_output_clamped)
    run("anti-windup clamps integral",                   test_pid_anti_windup)
    run("PID converges queue to target",                 test_pid_convergence)
    run("latency window percentiles correct",            test_latency_window_percentiles)
    run("latency window rolls old values off",           test_latency_window_rolling)
    run("anomaly not triggered on normal values",        test_anomaly_not_triggered_on_normal)
    run("anomaly triggered on spike",                    test_anomaly_triggered_on_spike)
    run("anomaly count increments correctly",            test_anomaly_count_increments)
    print("=" * 60)
    print(f"  {passed} passed   {failed} failed")
    print("=" * 60 + "\n")
    if failed:
        raise SystemExit(1)


