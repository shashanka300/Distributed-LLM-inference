# metrics/pid.py
import time
import threading
import collections
import logging

logger = logging.getLogger("pid")


class PIDController:
    """
    Discrete PID controller for autoscaling.

    Error signal:  e(t) = target - measured
    Control output: u(t) = Kp*e + Ki*integral + Kd*derivative

    For autoscaling:
      measured  = current cluster queue depth
      target    = desired queue depth (e.g. 0 keeps queue empty)
      output    = how many workers to add (positive) or remove (negative)

    Tuning guide (Ziegler-Nichols heuristic):
      Start with Ki=Kd=0, increase Kp until the system oscillates,
      then set Kp=0.6*Kp_critical, Ki=2*Kp/T, Kd=Kp*T/8
      where T is the oscillation period.

    For this project, conservative defaults work well:
      Kp=0.5  Ki=0.1  Kd=0.05
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        target: float = 0.0,       # desired queue depth
        output_min: float = -2.0,  # max workers to remove per tick
        output_max: float = 2.0,   # max workers to add per tick
        integral_max: float = 10.0 # anti-windup clamp on integral term
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.output_min  = output_min
        self.output_max  = output_max
        self.integral_max = integral_max

        self._integral    = 0.0
        self._last_error  = 0.0
        self._last_time   = time.perf_counter()
        self._lock        = threading.Lock()

        # history for plotting / debugging
        self._history: collections.deque[dict] = collections.deque(maxlen=200)

    def update(self, measured: float) -> float:
        with self._lock:
            now = time.perf_counter()
            dt  = now - self._last_time
            if dt <= 0:
                dt = 1e-6

            # FIXED: positive error means queue too deep = need more workers
            e = measured - self.target

            p = self.kp * e

            self._integral = max(
                -self.integral_max,
                min(self.integral_max, self._integral + e * dt)
            )
            # leak term: decay integral when error is near zero
            # prevents persistent bias after load subsides
            if abs(e) < 0.5:
                self._integral *= 0.95

            i = self.ki * self._integral
            d = self.kd * (e - self._last_error) / dt

            u = max(self.output_min, min(self.output_max, p + i + d))

            self._last_error = e
            self._last_time  = now

            record = {
                "ts": now, "measured": measured, "target": self.target,
                "error": round(e, 3), "p": round(p, 3),
                "i": round(i, 3), "d": round(d, 3),
                "output": round(u, 3), "integral": round(self._integral, 3),
            }
            self._history.append(record)
            return u

    def reset(self):
        with self._lock:
            self._integral   = 0.0
            self._last_error = 0.0
            self._last_time  = time.perf_counter()

    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    def stats(self) -> dict:
        with self._lock:
            last = self._history[-1] if self._history else {}
            return {
                "kp": self.kp, "ki": self.ki, "kd": self.kd,
                "target":   self.target,
                "integral": round(self._integral, 3),
                "last":     last,
                "ticks":    len(self._history),
            }


