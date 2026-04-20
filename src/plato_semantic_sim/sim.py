"""Semantic similarity scoring — Jaccard, cosine, Levenshtein, BM25, weighted."""
import math
import re
from collections import Counter
from typing import Optional

class SemanticSim:
    def __init__(self, stop_words: list[str] = None):
        self.stop_words = set(stop_words or [
            "the","a","an","is","are","was","were","be","been","being",
            "in","on","at","to","for","of","with","by","from","and",
            "or","but","not","it","this","that","as","if","has","had",
            "do","does","did","will","would","can","could","should"])

    def tokenize(self, text: str) -> list[str]:
        return [w for w in re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
                if w not in self.stop_words]

    def jaccard(self, a: str, b: str) -> float:
        sa, sb = set(self.tokenize(a)), set(self.tokenize(b))
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def cosine(self, a: str, b: str) -> float:
        va, vb = Counter(self.tokenize(a)), Counter(self.tokenize(b))
        dot = sum(va[k] * vb[k] for k in va if k in vb)
        ma = math.sqrt(sum(v ** 2 for v in va.values()))
        mb = math.sqrt(sum(v ** 2 for v in vb.values()))
        return dot / (ma * mb) if ma and mb else 0.0

    def levenshtein(self, a: str, b: str) -> float:
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
        max_len = max(m, n)
        return 1.0 - dp[n] / max_len if max_len else 1.0

    def bm25(self, query: str, doc: str, avg_dl: float = 50.0, k1: float = 1.5, b: float = 0.75) -> float:
        q_tokens = self.tokenize(query)
        d_tokens = self.tokenize(doc)
        dl = len(d_tokens) if d_tokens else 1
        score = 0.0
        tf = Counter(d_tokens)
        for qt in q_tokens:
            if qt not in tf:
                continue
            term_tf = tf[qt]
            idf = math.log(1 + (avg_dl - term_tf + 0.5) / (term_tf + 0.5) + 1)
            tf_component = (term_tf * (k1 + 1)) / (term_tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_component
        return min(score / max(len(q_tokens), 1), 1.0)

    def weighted(self, a: str, b: str, domain_match: bool = False) -> float:
        j = self.jaccard(a, b)
        c = self.cosine(a, b)
        l = self.levenshtein(a, b)
        domain_bonus = 0.15 if domain_match else 0.0
        return min(1.0, j * 0.3 + c * 0.3 + l * 0.25 + domain_bonus)

    def nearest(self, query: str, candidates: list[str], top_n: int = 5,
                method: str = "weighted") -> list[tuple[str, float]]:
        scorer = {"jaccard": self.jaccard, "cosine": self.cosine,
                  "levenshtein": self.levenshtein, "bm25": self.bm25,
                  "weighted": self.weighted}.get(method, self.weighted)
        scored = [(c, scorer(query, c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def batch_similarity(self, pairs: list[tuple[str, str]],
                         method: str = "cosine") -> list[float]:
        scorer = {"jaccard": self.jaccard, "cosine": self.cosine,
                  "levenshtein": self.levenshtein, "weighted": self.weighted}.get(method, self.cosine)
        return [scorer(a, b) for a, b in pairs]
