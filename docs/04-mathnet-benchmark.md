# MathNet 100 Benchmark

This benchmark compares two generation paths on the same 100 sampled MathNet
United States seed problems.

- **Direct baseline**: seed question + solution -> direct generation.
- **atomicmath**: seed question + solution -> hinge extraction -> mutation prompt
  with global memory -> generation.

Both outputs are evaluated with the same seed-relative comparison judge. This is
the apples-to-apples metric used for the external acceptance rate.

## Shared Comparison Judge

| Metric | Direct baseline | atomicmath |
| --- | ---: | ---: |
| Total seeds | 100 | 100 |
| Generated | 96 | 95 |
| Judged | 96 | 94 |
| Accepted | 35 | 34 |
| Accepted rate | 35.0% | 34.0% |
| Correctness rate | 96.9% | 88.3% |
| Generation success rate | 96.0% | 95.0% |
| Judge success rate | 96.0% | 94.0% |
| Mean MinHash overlap | 0.047 | 0.017 |
| Mean embedding cosine | 0.603 | 0.612 |
| Mean depth score | 0.482 | 0.564 |
| Mean contest score | 0.628 | 0.678 |
| Mean novelty score | 0.659 | 0.482 |
| Mean seed alignment | 0.628 | 0.790 |
| Mean non-stitched score | 0.981 | 0.990 |
| Mean solution economy | 0.889 | 0.880 |
| Mean routine score | 0.690 | 0.652 |

## Failure Breakdown

| Failure kind | Direct baseline | atomicmath |
| --- | ---: | ---: |
| accepted | 35 | 34 |
| generation/error | 4 | 6 |
| incorrect | 3 | 11 |
| near_paraphrase | 6 | 23 |
| routine | 17 | 8 |
| weak_quality | 35 | 18 |

## Atomicmath Internal Judge

atomicmath also reports its own hinge-aware internal judge. This judge measures
whether the generated candidate preserves and transforms the seed hinge.

| Internal atomicmath metric | Value |
| --- | ---: |
| Accepted | 82 / 100 |
| Rejected | 18 / 100 |
| Internal accepted rate | 82.0% |
| Mean hinge preservation | 0.945 |
| Mean mutation quality | 0.883 |
| Mean sharpness | 0.827 |
| Mean atomic novelty | 0.800 |
| Mean atomic non-stitched score | 0.996 |
| Mean atomic solution economy | 0.898 |

## Interpretation

Under the shared comparison judge, atomicmath is currently similar to direct
prompting on external accepted rate. Its quality profile is different:

- stronger depth and contest scores;
- stronger seed alignment;
- lower routine score;
- lower MinHash overlap;
- stronger non-stitched structure.

The main regression is that the generated candidates need stricter correctness
repair and stronger anti-sibling pressure before publication. The internal judge
accepts many candidates that the shared comparison judge rejects, so a key next
step is aligning the hinge-aware judge with external quality criteria.

## Datasets

- [`vibhuiitj/mathnet-direct-baseline_100`](https://huggingface.co/datasets/vibhuiitj/mathnet-direct-baseline_100)
- [`vibhuiitj/mathnet-atomicmath_100`](https://huggingface.co/datasets/vibhuiitj/mathnet-atomicmath_100)
