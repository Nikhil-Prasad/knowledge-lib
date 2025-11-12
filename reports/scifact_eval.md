# SciFact Retrieval Evaluation
Generated: 2025-11-11 15:54:47

- Queries: `/Users/nikhilprasad/crown/knowledge-lib/evals/datasets/beir/eval/scifact/queries.jsonl`
- Qrels: `/Users/nikhilprasad/crown/knowledge-lib/evals/datasets/beir/eval/scifact/qrels/test.tsv`
- Collection: `ALL`
- K: `10`  •  Total evaluated: `300`

## Summary
Method | HIT@k | MRR@k | Recall@k
--- | ---: | ---: | ---:
FTS | 35.3% | 0.203 | 0.342
ANN | 81.0% | 0.676 | 0.796
HYBRID | 75.0% | 0.532 | 0.726

## Sample Queries (first 10)
- QID `1`: 0-dimensional biomaterials show inductive properties.
  - expected: ['1d7574fc-cef9-494c-b46d-b060c59ac257']
  - FTS: no hit in top-10, recall@10=0.00
  - ANN: no hit in top-10, recall@10=0.00
  - HYBRID: no hit in top-10, recall@10=0.00
- QID `3`: 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.
  - expected: ['047b7627-07e5-418e-88fb-2a30550669eb']
  - FTS: no hit in top-10, recall@10=0.00
  - ANN: no hit in top-10, recall@10=0.00
  - HYBRID: rank=6, recall@10=1.00
- QID `5`: 1/2000 in UK have abnormal PrP positivity.
  - expected: ['d75fcfd4-c419-4ba2-9dcd-3321cee57de5']
  - FTS: no hit in top-10, recall@10=0.00
  - ANN: rank=1, recall@10=1.00
  - HYBRID: rank=1, recall@10=1.00
- QID `13`: 5% of perinatal mortality is due to low birth weight.
  - expected: ['584432cf-c7bd-4322-8536-d62edf33372a']
  - FTS: no hit in top-10, recall@10=0.00
  - ANN: no hit in top-10, recall@10=0.00
  - HYBRID: no hit in top-10, recall@10=0.00
- QID `36`: A deficiency of vitamin B12 increases blood levels of homocysteine.
  - expected: ['5e6f6db4-eaed-4d73-aff2-69db4903a306', '8a487756-d7a2-4cf9-97e8-004c9143a47d']
  - FTS: no hit in top-10, recall@10=0.00
  - ANN: no hit in top-10, recall@10=0.00
  - HYBRID: no hit in top-10, recall@10=0.00
- QID `42`: A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects.
  - expected: ['a2a77a03-13ae-4df4-a209-1b64fc235fc2']
  - FTS: no hit in top-10, recall@10=0.00
  - ANN: rank=1, recall@10=1.00
  - HYBRID: rank=1, recall@10=1.00
- QID `48`: A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.
  - expected: ['d75fcfd4-c419-4ba2-9dcd-3321cee57de5']
  - FTS: no hit in top-10, recall@10=0.00
  - ANN: rank=3, recall@10=1.00
  - HYBRID: rank=7, recall@10=1.00
- QID `49`: ADAR1 binds to Dicer to cleave pre-miRNA.
  - expected: ['8c9755f6-6ab9-4557-88e4-ae48bf887d8f']
  - FTS: rank=1, recall@10=1.00
  - ANN: rank=1, recall@10=1.00
  - HYBRID: rank=1, recall@10=1.00
- QID `50`: AIRE is expressed in some skin tumors.
  - expected: ['25eea91d-8c4e-4d35-b54b-d68175e5fd3c']
  - FTS: rank=1, recall@10=1.00
  - ANN: rank=1, recall@10=1.00
  - HYBRID: rank=1, recall@10=1.00
- QID `51`: ALDH1 expression is associated with better breast cancer outcomes.
  - expected: ['95902f5c-54e8-4598-bcf9-8950a6f142a4']
  - FTS: no hit in top-10, recall@10=0.00
  - ANN: rank=1, recall@10=1.00
  - HYBRID: rank=6, recall@10=1.00
