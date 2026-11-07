"""Test doubles for the runner/CLI tests. THESE ARE FAKES: a controlled clock and scripted retrievers.
They verify scheduling, timing boundaries, exports and failure handling only; they say nothing about real
retrieval quality or real latency."""

from __future__ import annotations


class FakeClock:
    """Controlled nanosecond clock: returns a fixed 'now' that only moves when a fake retriever advances it."""

    def __init__(self, start=1_000_000_000, events=None):
        self.now, self.reads, self.events = start, 0, events

    def __call__(self):
        self.reads += 1
        if self.events is not None:
            self.events.append(("clock",))
        return self.now

    def advance(self, ns):
        self.now += ns


def default_ranking(name, doc_ids):
    ids = sorted(doc_ids, reverse=(name == "embedding"))
    return [(d, float(len(ids) - i)) for i, d in enumerate(ids)]


class FakeRetriever:
    """Scripted retriever. search() advances the fake clock by cost(query, call_number) and returns a scripted list."""

    BASE = {"bm25": 1000, "embedding": 5000}

    def __init__(self, name, doc_ids, clock, *, events=None, rankings=None, fail=None, drift_after=None,
                 reverse_calls=None, jitter_calls=None):
        self.name, self._doc_ids, self.clock, self.events = name, tuple(doc_ids), clock, events
        self.rankings = rankings or {}
        self.fail = fail or {}      # (query_text, call_number or None) -> exception to raise
        self.drift_after = drift_after  # call_number after which results are reversed (consistency test)
        self.reverse_calls = set(reverse_calls or ())  # call numbers (per query) whose ORDER is reversed
        self.jitter_calls = set(jitter_calls or ())    # call numbers whose scores get +1e-9 (same order)
        self.counts: dict[str, int] = {}

    @property
    def doc_ids(self):
        return self._doc_ids

    def cost(self, n):
        return self.BASE[self.name] + 10 * n

    def search(self, query_text, k):
        n = self.counts[query_text] = self.counts.get(query_text, 0) + 1
        if self.events is not None:
            self.events.append(("search", self.name, query_text, k))
        exc = self.fail.get((query_text, n)) or self.fail.get((query_text, None))
        if exc is not None:
            raise exc
        self.clock.advance(self.cost(n))
        result = self.rankings.get(query_text)
        if result is None:
            result = default_ranking(self.name, self._doc_ids)
        result = list(result)
        if self.drift_after is not None and n > self.drift_after:
            result = list(reversed(result))
        if n in self.reverse_calls:
            result = list(reversed(result))
        if n in self.jitter_calls:
            result = [(d, sc + 1e-9) for d, sc in result]
        return result[:k]

    def describe(self):
        self.clock.advance(10**9)  # outside the timed region: must never leak into any measured sample
        info = {"name": self.name, "fake": True, "n_documents": len(self._doc_ids)}
        if self.name == "embedding":
            info["encoder"] = {"model_id": "FAKE-encoder", "revision": "fake"}
        return info


def make_builders(clock, doc_ids, *, events=None, per_method=None):
    """[(name, builder)] for the two methods. A builder reads the clock twice, like a real timed setup."""
    per_method = per_method or {}

    def make(name):
        def build(clk):
            t0 = clk()
            retriever = FakeRetriever(name, doc_ids, clock, events=events, **per_method.get(name, {}))
            return retriever, {"fake_build_ns": clk() - t0}
        return build

    return [("bm25", make("bm25")), ("embedding", make("embedding"))]


def fake_run(dataset, dataset_dir, *, warmup=3, repetitions=20, ks=(1, 3, 5), per_method=None, dataset_info=None):
    """Run the real runner over FAKE retrievers with a controlled clock; return the run document."""
    from retrieval_eval.runner import make_config, run_evaluation

    clock = FakeClock()
    cfg = make_config(dataset_dir=dataset_dir, ks=ks, warmup=warmup, repetitions=repetitions)
    return run_evaluation(cfg, dataset, make_builders(clock, dataset.doc_ids, per_method=per_method), clock=clock,
                          dataset_info=dataset_info or {})
