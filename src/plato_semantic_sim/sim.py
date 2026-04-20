"""Semantic similarity scoring."""
import math, re
from collections import Counter

class SemanticSim:
    def __init__(self, stop_words: list[str] = None):
        self.stop_words = set(stop_words or ["the","a","an","is","are","was","were","be","been",
            "being","in","on","at","to","for","of","with","by","from","and","or","but","not","it","this","that"])

    def tokenize(self, text: str) -> list[str]:
        return [w for w in re.findall(r'\b\w+\b', text.lower()) if w not in self.stop_words and len(w) > 1]

    def jaccard(self, a: str, b: str) -> float:
        sa, sb = set(self.tokenize(a)), set(self.tokenize(b))
        if not sa and not sb: return 1.0
        if not sa or not sb: return 0.0
        return len(sa & sb) / len(sa | sb)

    def cosine(self, a: str, b: str) -> float:
        va, vb = Counter(self.tokenize(a)), Counter(self.tokenize(b))
        dot = sum(va[k]*vb[k] for k in va if k in vb)
        ma, mb = math.sqrt(sum(v**2 for v in va.values())), math.sqrt(sum(v**2 for v in vb.values()))
        return dot/(ma*mb) if ma and mb else 0.0

    def weighted(self, a: str, b: str, da: str = "", db: str = "") -> float:
        return min(1.0, self.jaccard(a,b)*0.4 + self.cosine(a,b)*0.4 + (0.2 if da and db and da==db else 0.0))

    def nearest(self, query: str, candidates: list[str], top_n: int = 5) -> list[tuple[str, float]]:
        scored = sorted([(c, self.weighted(query, c)) for c in candidates], key=lambda x: x[1], reverse=True)
        return scored[:top_n]
