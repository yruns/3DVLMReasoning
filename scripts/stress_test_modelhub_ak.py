"""Stress-test a single ModelHub AK against gemini-2.5-pro.

Runs a series of (concurrency, duration) steps and reports for each:
- successful requests
- error counts split by HTTP status / error code
- p50 / p95 / p99 latency
- effective QPS

Usage:
    python scripts/stress_test_modelhub_ak.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

AK = "SzeibpIBnTrgVVyP1hY3VGRwrVEEXGgj_GPT_AK"
ENDPOINT = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl"
MODEL = "gemini-2.5-pro"
PAYLOAD = {
    "stream": False,
    "model": MODEL,
    "max_tokens": 64,
    "messages": [{"content": "reply OK", "role": "user"}],
    "thinking": {"budget_tokens": 1024},
}
HEADERS_BASE = {"Content-Type": "application/json"}


def _one_call(client: httpx.Client, idx: int) -> tuple[bool, float, str]:
    headers = dict(HEADERS_BASE)
    headers["X-TT-LOGID"] = f"stress_{int(time.time()*1000)}_{idx}"
    t0 = time.time()
    try:
        r = client.post(
            ENDPOINT,
            params={"ak": AK},
            headers=headers,
            json=PAYLOAD,
            timeout=60.0,
        )
        dt = time.time() - t0
        if r.status_code == 200:
            return True, dt, "200"
        # Try parse error code
        try:
            err = r.json().get("error", {})
            tag = f"{r.status_code}/{err.get('code','?')}"
        except Exception:
            tag = f"{r.status_code}/parse"
        return False, dt, tag
    except httpx.TimeoutException:
        return False, time.time() - t0, "timeout"
    except Exception as e:
        return False, time.time() - t0, f"exc/{type(e).__name__}"


def run_step(concurrency: int, duration_s: float) -> dict:
    """Drive `concurrency` workers continuously for `duration_s` seconds."""
    end_t = time.time() + duration_s
    results: list[tuple[bool, float, str]] = []
    results_lock = threading.Lock()

    def worker_loop(client: httpx.Client, wid: int) -> None:
        local_idx = 0
        while time.time() < end_t:
            ok, dt, tag = _one_call(client, wid * 10000 + local_idx)
            with results_lock:
                results.append((ok, dt, tag))
            local_idx += 1

    t0 = time.time()
    with httpx.Client(http2=False) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(worker_loop, client, i) for i in range(concurrency)]
            for f in futs:
                f.result()
    elapsed = time.time() - t0

    n = len(results)
    n_ok = sum(1 for r in results if r[0])
    n_err = n - n_ok
    err_counter = Counter(r[2] for r in results if not r[0])
    lat_ok = sorted(r[1] for r in results if r[0])

    def pct(p):
        if not lat_ok:
            return 0.0
        i = max(0, min(len(lat_ok) - 1, int(round(p * (len(lat_ok) - 1)))))
        return lat_ok[i]

    return {
        "concurrency": concurrency,
        "duration_s": round(elapsed, 2),
        "total": n,
        "ok": n_ok,
        "err": n_err,
        "qps": round(n_ok / elapsed, 2) if elapsed else 0.0,
        "p50": round(pct(0.50), 2),
        "p95": round(pct(0.95), 2),
        "p99": round(pct(0.99), 2),
        "errors_by_tag": dict(err_counter.most_common()),
    }


def main() -> None:
    print(f"AK ...{AK[-12:]}  endpoint={ENDPOINT}")
    print(f"payload model={MODEL} max_tokens=64 budget=1024")
    print()
    print(
        f"{'N':>4} {'dur':>5} {'tot':>4} {'ok':>4} {'err':>4} {'qps':>6} "
        f"{'p50':>5} {'p95':>5} {'p99':>5}  errors"
    )
    print("-" * 80)
    # Step plan: ramp concurrency. 30s per step for headline N values.
    steps = [
        (1, 20),
        (4, 30),
        (8, 30),
        (16, 30),
        (32, 30),
        (64, 30),
    ]
    all_results = []
    for c, d in steps:
        r = run_step(c, d)
        all_results.append(r)
        print(
            f"{r['concurrency']:>4} {r['duration_s']:>5} {r['total']:>4} "
            f"{r['ok']:>4} {r['err']:>4} {r['qps']:>6} "
            f"{r['p50']:>5} {r['p95']:>5} {r['p99']:>5}  {r['errors_by_tag']}"
        )
        sys.stdout.flush()
        # Cooldown if errors spike
        if r["err"] > 0 and r["err"] / max(1, r["total"]) > 0.5:
            print("  >>> >50% errors, stopping ramp")
            break
        time.sleep(2)
    print()
    print("=== JSON dump ===")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
