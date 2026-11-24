# Datasets

Data payloads are not tracked in this repository. The sidecar reads its candidate catalog from
`Datasets/v2/catalog_snapshot.json`, relative to the repository root:

```text
Datasets/
└── v2/
    └── catalog_snapshot.json   # candidate capabilities and knowledge entries
```

`catalog_snapshot.json` describes the capabilities and knowledge entries that the sidecar renders as
candidate cards. It is installation-specific: generate it from your own Bioflow tool library and
knowledge base, and keep it out of git. A missing snapshot means the sidecar cannot score candidates
and the runtime falls back to browsing.

Training and evaluation data for the decision heads are also not included. The splits, row counts and
SHA-256 values used for the released checkpoint are recorded in the review package that accompanies
the paper.
