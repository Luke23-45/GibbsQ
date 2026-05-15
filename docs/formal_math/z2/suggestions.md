# Archived External Suggestions

This file is retained only as an archive pointer. It is not an authoritative
theorem-status note for the `z2` package.

The original external suggestion memo mixed correct observations with
overclaims. In particular, it correctly identified the boundary mismatch in the
CTMC generator for the potential \(H\), but it also proposed stochastic-scaling
claims that do not certify the original fixed-parameter CTMC as written.

The authoritative review of those suggestions is:

- [09_review_of_suggestions.md](./09_review_of_suggestions.md)

Use that reviewed file, not the archived memo text, when writing thesis,
publication, abstract, or defense material.

The safe retained conclusions are:

- the exact CTMC generator calculation for \(H\) in
  [08_ctmc_generator_analysis.md](./08_ctmc_generator_analysis.md) is correct;
- the boundary mismatch between the reflected ODE and the CTMC generator is
  real;
- the current \(H\)-based calculation does not close a direct CTMC stability
  proof;
- the temperature-scaling bridge suggested in the original memo does not, by
  itself, prove stability of the original fixed-parameter CTMC;
- any direct CTMC certification must be judged from files `10` and `11`, not
  from the superseded suggestion memo.
