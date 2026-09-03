# Superseded sqlparse harvest (moved 2026-09-03)

These labels were produced against a checkout that did not match the screened
revision. The subject suite truncates its own tracked fixture
`tests/files/function.sql` when it runs, and the damage was already on disk when
the baseline screen ran. With the fixture empty exactly one test fails, so the
screen recorded

    sqlparse/tests/test_split::test_split_create_function[function.sql]
      -> "failing_on_clean_checkout:FAILED"

which is a false description of a clean checkout, and the labelled universe was
506 tests instead of 507. All 250 mutant runs then executed against the damaged
tree. No test shows mid-run kill drift (a persistent-kill scan found zero
suspects), so the kill lists here are not believed to be wrong -- but they are
one test short of the universe and they cannot be traced to the pinned SHA
60cdc649726bf1bc4f1b336050560b336da715ec as it actually stands.

Kept for comparison, not for use. Recorded totals: 237 harvested, 4 broke_suite,
9 timed_out, 15964 kills, universe 506, 40.2 min.
