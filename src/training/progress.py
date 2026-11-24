from __future__ import annotations

import sys
import time


def hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class Progress:
    """Counts runs (one model x horizon x learning rate x seed). ``status`` describes the current work item."""

    def __init__(self, total: int, desc: str, every: float = 0.1):
        self.total, self.desc, self.done = max(int(total), 1), desc, 0
        self.t0 = time.perf_counter()
        self.tty = sys.stdout.isatty()
        self._next = every
        self._every = every
        self.bar = None
        if self.tty:
            from tqdm import tqdm

            self.bar = tqdm(total=self.total, desc=desc, unit=" runs", dynamic_ncols=True, smoothing=0.05)
        else:
            print(f"  {desc}: {self.total:,} runs", flush=True)

    def status(self, text: str) -> None:
        if self.bar is not None:
            self.bar.set_postfix_str(text, refresh=True)
        self._status = text

    def update(self, n: int) -> None:
        self.done += n
        if self.bar is not None:
            self.bar.update(n)
            return
        frac = self.done / self.total
        if frac >= self._next or self.done >= self.total:
            el = time.perf_counter() - self.t0
            eta = el / frac * (1 - frac) if frac > 0 else 0.0
            print(f"  {self.desc}: {frac:4.0%} ({self.done:,}/{self.total:,} runs), elapsed {hms(el)}, "
                  f"remaining ~{hms(eta)}  [{getattr(self, '_status', '')}]", flush=True)
            while self._next <= frac:
                self._next += self._every

    def close(self) -> float:
        if self.bar is not None:
            self.bar.close()
        return time.perf_counter() - self.t0
