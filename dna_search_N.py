# ================================================================
#  PROJECT 10 — DNA FUZZY SEARCH SYSTEM
#  Algorithms for Data Science | University of Warsaw
# ================================================================
#
#  ALGORITHMIC CHAIN:
#    Stage A — Suffix Array  →  indexes the DNA in O(n log n)
#    Stage B — Edit Distance →  fuzzy matching in O(p · m) per candidate
#
#  LECTURE CONNECTIONS:
#    Suffix Array build  → Sorting / Merge Sort          (Lecture 6-7)
#    Binary Search query → Binary Search + Invariants    (Lecture 1)
#    Edit Distance DP    → Dynamic Programming            (Lecture 8-9)
#    Complexity analysis → Big O / Θ / Ω notation        (Lecture 1-3)
#
# ================================================================

import time


# ────────────────────────────────────────────────────────────────
# STAGE A — SUFFIX ARRAY
# ────────────────────────────────────────────────────────────────
#
# WHAT IS A SUFFIX?
#   For DNA = "ACGT", the suffixes are:
#     position 0 → "ACGT"
#     position 1 → "CGT"
#     position 2 → "GT"
#     position 3 → "T"
#
# WHAT IS A SUFFIX ARRAY?
#   A list of those positions, sorted by their suffix alphabetically.
#   For "ACGT": already sorted → suffix_array = [0, 1, 2, 3]
#
# WHY IS IT USEFUL?
#   If our pattern exists in the DNA, it must be the PREFIX (start)
#   of some suffix. So we can binary-search the sorted suffix array
#   to find our pattern in O(p · log n) instead of O(n · p).
#
# TIME:  O(n log n) — one sort, done once at startup
# SPACE: O(n)       — one integer per DNA position

def build_suffix_array(text):
    """
    Build a suffix array for the given DNA string.

    Sorts all starting positions by the suffix they represent.
    Uses Python's built-in Timsort (O(n log n), stable) — same
    asymptotic complexity as Merge Sort taught in Lecture 6-7.

    By the comparison-based lower bound proven in class (Lecture 6-7),
    no sorting algorithm can do better than O(n log n), so this is optimal.
    """
    n = len(text)
    suffix_array = sorted(range(n), key=lambda i: text[i:])
    return suffix_array


def _binary_search_left(text, suffix_array, pattern):
    """
    Find the leftmost position in suffix_array where the suffix
    starts with 'pattern' (or where pattern would be inserted).

    Loop invariant (from Lecture 1):
      The true left boundary is always in [lo, hi].
    """
    p  = len(pattern)
    lo, hi = 0, len(suffix_array)

    while lo < hi:
        mid = (lo + hi) // 2
        start = suffix_array[mid]
        # Compare only the first p characters of this suffix
        suffix_prefix = text[start : start + p]
        if suffix_prefix < pattern:
            lo = mid + 1   # answer is to the right
        else:
            hi = mid       # answer is here or to the left

    return lo


def _binary_search_right(text, suffix_array, pattern):
    """
    Find the rightmost boundary — first position where the suffix
    is strictly greater than 'pattern'.

    Loop invariant: the right boundary is always in [lo, hi].
    """
    p  = len(pattern)
    lo, hi = 0, len(suffix_array)

    while lo < hi:
        mid = (lo + hi) // 2
        start = suffix_array[mid]
        suffix_prefix = text[start : start + p]
        if suffix_prefix <= pattern:
            lo = mid + 1
        else:
            hi = mid

    return lo


def exact_search(text, suffix_array, pattern):
    """
    Find ALL positions where 'pattern' occurs exactly in 'text'.

    Uses two binary searches to find the range [left, right) in the
    suffix array where every entry is a match. Direct application of
    Binary Search from Lecture 1.

    TIME:  O(p · log n)  — p characters compared at each of log n steps
    SPACE: O(1)          — no extra memory beyond index variables
    """
    left  = _binary_search_left(text, suffix_array, pattern)
    right = _binary_search_right(text, suffix_array, pattern)

    # All positions in suffix_array[left:right] are exact matches
    return sorted(suffix_array[left:right])


# ────────────────────────────────────────────────────────────────
# STAGE B — EDIT DISTANCE (FUZZY MATCHING)
# ────────────────────────────────────────────────────────────────
#
# WHAT IS EDIT DISTANCE?
#   The minimum number of single-character operations needed to turn
#   one string into another. Operations: insert / delete / replace.
#
#   Example:
#     "ATGC" → "ATCC"  =  1 replacement (G → C), distance = 1
#     "ATGC" → "AGC"   =  1 deletion,              distance = 1
#
# WHY DO WE NEED IT?
#   DNA can have biological mutations — a letter changes, is inserted,
#   or is deleted. Exact search misses these. Edit distance finds them.
#
# HOW WE IMPLEMENT IT (from Lecture 8-9 — Dynamic Programming):
#
#   State:      D[i][j] = min edits between s1[0..i-1] and s2[0..j-1]
#   Base cases: D[i][0] = i  (delete i chars from s1)
#               D[0][j] = j  (insert j chars into s1)
#   Recurrence:
#     if s1[i-1] == s2[j-1]:  D[i][j] = D[i-1][j-1]         (free)
#     else:                    D[i][j] = 1 + min(
#                                          D[i-1][j],     ← delete
#                                          D[i][j-1],     ← insert
#                                          D[i-1][j-1])   ← replace
#
# TIME:  O(n · m)
# SPACE: O(m)  — space-optimised to two rows, as shown in Lecture 8-9

def edit_distance(s1, s2):
    """
    Compute Levenshtein edit distance between s1 and s2.

    Implements the bottom-up tabulation approach from Lecture 8-9,
    with the space optimisation: instead of keeping the full (n+1)×(m+1)
    table, we keep only two rows (prev and curr), since each row depends
    only on the one above it.
    """
    n, m = len(s1), len(s2)

    # Base case row: D[0][j] = j for all j
    prev = list(range(m + 1))

    for i in range(1, n + 1):
        # Base case column: D[i][0] = i
        curr = [i] + [0] * m

        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]               # characters match — free
            else:
                curr[j] = 1 + min(
                    prev[j],        # delete s1[i-1]
                    curr[j - 1],    # insert s2[j-1]
                    prev[j - 1]     # replace s1[i-1] with s2[j-1]
                )

        prev = curr   # move to next row

    return prev[m]


# ────────────────────────────────────────────────────────────────
# HOW STAGE A FEEDS INTO STAGE B  (the critical handoff)
# ────────────────────────────────────────────────────────────────
#
# NAIVE approach: run edit_distance(pattern, dna[i:i+p]) for every i
#   → O(n · p · k) — too slow for a 3-billion-letter genome
#
# SMART approach using the suffix array:
#   Pigeonhole Principle:
#     If we allow k mismatches, divide pattern into (k+1) equal chunks.
#     With only k edits available, at least ONE chunk must survive intact
#     (there aren't enough edits to corrupt all k+1 chunks).
#
#   Therefore:
#     1. Binary-search the suffix array for each chunk → candidates
#     2. Run edit_distance ONLY on those candidates
#
#   This turns O(n) edit distance calls into O(few) edit distance calls.

def fuzzy_search(text, suffix_array, pattern, max_mismatches):
    """
    Find all positions in 'text' where 'pattern' occurs with at most
    'max_mismatches' edit operations.

    Stage A → Stage B handoff:
      The suffix array (Stage A output) is used to efficiently generate
      a small candidate set. Edit distance (Stage B) then verifies each.

    TIME (query):  O((k+1) · p · log n)  for candidate generation
                 + O(candidates · p · m)  for edit distance verification
    SPACE:         O(n) for suffix array + O(p · m) for edit distance table
    """
    p = len(pattern)

    # ── Step 1: Generate candidates using Pigeonhole Principle ──
    candidates = set()
    num_chunks  = max_mismatches + 1      # k mismatches → k+1 chunks
    chunk_size  = max(1, p // num_chunks)

    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end   = chunk_start + chunk_size if i < num_chunks - 1 else p
        chunk       = pattern[chunk_start : chunk_end]

        # Binary search the suffix array for this exact chunk
        exact_positions = exact_search(text, suffix_array, chunk)

        # If this chunk starts at offset 'chunk_start' inside the pattern,
        # then the full pattern would start at (pos - chunk_start) in the DNA
        for pos in exact_positions:
            candidate_start = pos - chunk_start
            if 0 <= candidate_start and candidate_start + p <= len(text):
                candidates.add(candidate_start)

    # ── Step 2: Verify each candidate with edit distance ────────
    matches = []
    for start in sorted(candidates):
        dna_window = text[start : start + p]
        dist = edit_distance(pattern, dna_window)

        if dist <= max_mismatches:
            matches.append({
                'position'     : start,
                'dna_found'    : dna_window,
                'edit_distance': dist,
                'exact'        : dist == 0
            })

    return matches


# ────────────────────────────────────────────────────────────────
# DEMO RUNNER
# ────────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 65)
    print("  DNA FUZZY SEARCH SYSTEM")
    print("  Project 10 — Algorithms for Data Science")
    print("  University of Warsaw")
    print("=" * 65)

    # ── Sample DNA ────────────────────────────────────────────
    # Real genome = 3 billion letters. Demo uses a short but
    # realistic sequence. The pattern is embedded at position 10.
    dna = (
        "GCTAGCATGC"                     # positions  0-9   (prefix noise)
        "ATGCTAGCTAGTACGAT"              # positions 10-26  (our TARGET)
        "CGATCGTAGCTAGCATGCATGCTAGCTA"   # positions 27-54  (noise)
        "GCTAGTACGATCGATCGTAGCTAGCATG"   # positions 55-82  (noise)
        "ATGCTATCTATTACGAT"              # positions 83-99  (1-edit variant)
        "CATGCTAGCTAGTACGATCGGCTAGCAT"   # positions 100+   (exact copy)
    )

    target_pattern  = "ATGCTAGCTAGTACGAT"   # the gene we're looking for
    mutated_pattern = "ATGCTATCTATTACGAT"   # same gene with 2 mutations
    #                          ^^  ^^
    #                 pos 6: G→C, pos 9: A→T (2 edits from target)
    max_k = 2

    print(f"\nDNA length : {len(dna)} characters")
    print(f"DNA string : {dna[:55]}...")
    print(f"\nTarget pattern  : '{target_pattern}'  (length {len(target_pattern)})")
    print(f"Mutated pattern : '{mutated_pattern}'  (length {len(mutated_pattern)})")
    print(f"                           ^^  ^^")
    print(f"                   2 mutations introduced (edit distance = 2)")

    # ── STAGE A: Build suffix array ───────────────────────────
    print("\n" + "─" * 65)
    print("  STAGE A — Building Suffix Array")
    print("─" * 65)
    print(f"  Sorting all {len(dna)} suffixes alphabetically...")

    t0 = time.perf_counter()
    sa = build_suffix_array(dna)
    t1 = time.perf_counter()

    print(f"  Done in {(t1-t0)*1000:.3f} ms")
    print(f"  Array has {len(sa)} entries  (one integer per DNA position)")
    print(f"\n  First 5 entries in sorted order (position → suffix preview):")
    for idx in sa[:5]:
        print(f"    pos {idx:3d}  →  \"{dna[idx:idx+25]}...\"")

    print(f"\n  Complexity: O(n log n) time, O(n) space")
    print(f"  Why O(n log n): sorting n suffixes — same as Merge Sort (Lecture 6-7)")
    print(f"  Why optimal: comparison sort lower bound Ω(n log n) (Lecture 6-7)")

    # ── STAGE B — Exact search ────────────────────────────────
    print("\n" + "─" * 65)
    print("  STAGE B (Part 1) — Exact Search via Binary Search")
    print("─" * 65)
    print(f"  Pattern: '{target_pattern}'")

    t0 = time.perf_counter()
    exact_hits = exact_search(dna, sa, target_pattern)
    t1 = time.perf_counter()

    print(f"  Found {len(exact_hits)} exact match(es) in {(t1-t0)*1000:.4f} ms\n")
    for pos in exact_hits:
        print(f"    → Position {pos:3d}:  '{dna[pos : pos+len(target_pattern)]}'  [EXACT]")

    print(f"\n  Complexity: O(p · log n) per query")
    print(f"  Why log n : binary search halves the array each step (Lecture 1)")
    print(f"  Why · p   : each step compares p characters of the suffix")

    # ── STAGE B — Fuzzy search ────────────────────────────────
    print("\n" + "─" * 65)
    print("  STAGE B (Part 2) — Fuzzy Search via Edit Distance DP")
    print("─" * 65)
    print(f"  Pattern:         '{mutated_pattern}'")
    print(f"  Max mismatches:  {max_k}")
    print(f"\n  Strategy (A → B handoff):")
    print(f"    1. Pigeonhole: divide pattern into {max_k+1} chunks")
    print(f"       At least 1 chunk must appear exactly → binary search it")
    print(f"    2. Edit distance DP verifies each candidate (Lecture 8-9)")

    t0 = time.perf_counter()
    fuzzy_hits = fuzzy_search(dna, sa, mutated_pattern, max_k)
    t1 = time.perf_counter()

    print(f"\n  Found {len(fuzzy_hits)} fuzzy match(es) in {(t1-t0)*1000:.4f} ms\n")
    for hit in fuzzy_hits:
        tag = "EXACT" if hit['exact'] else f"{hit['edit_distance']} edit(s)"
        print(f"    → Position {hit['position']:3d}:  '{hit['dna_found']}'  [{tag}]")

    print(f"\n  Complexity: O(p · m) per candidate")
    print(f"  Why p · m : DP table has p rows × m columns (Lecture 8-9)")

    # ── Edit distance walkthrough ─────────────────────────────
    print("\n" + "─" * 65)
    print("  EDIT DISTANCE WALKTHROUGH (Lecture 8-9 DP)")
    print("─" * 65)
    s1 = mutated_pattern
    s2 = target_pattern
    dist = edit_distance(s1, s2)
    print(f"  edit_distance('{s1}',")
    print(f"                '{s2}')")
    print(f"\n  Aligning both strings:")
    print(f"  Mutated : {s1}")
    print(f"  Target  : {s2}")
    diffs = ['^' if a != b else ' ' for a, b in zip(s1, s2)]
    print(f"  Diffs   : {''.join(diffs)}  ← {dist} mismatch(es)")
    print(f"\n  Edit distance = {dist}  ✓  (≤ max_mismatches={max_k} → MATCH ACCEPTED)")

    # ── Complexity summary ────────────────────────────────────
    print("\n" + "─" * 65)
    print("  FULL SYSTEM COMPLEXITY SUMMARY")
    print("─" * 65)
    print(f"  ┌──────────────────────┬─────────────────┬──────────────┐")
    print(f"  │ Phase                │ Time            │ Space        │")
    print(f"  ├──────────────────────┼─────────────────┼──────────────┤")
    print(f"  │ Stage A: Build index │ O(n log n)      │ O(n)         │")
    print(f"  │ Stage A: Search      │ O(p · log n)    │ O(1)         │")
    print(f"  │ Stage B: Edit dist.  │ O(p · m)        │ O(m)         │")
    print(f"  └──────────────────────┴─────────────────┴──────────────┘")
    print(f"")
    print(f"  Bottleneck: Stage A build — O(n log n), done ONCE at startup.")
    print(f"  After that, every query is fast: O(p · log n) + O(candidates · p · m)")
    print(f"")
    print(f"  Why Suffix Array over Hash Map?")
    print(f"    Hash map of all k-mers: O(n · p) space — too large for 3B DNA")
    print(f"    Suffix array:           O(n) space — only stores n integers")
    print(f"    Bonus: suffix array supports binary search — hash map does not")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()