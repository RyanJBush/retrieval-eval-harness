# Synthetic fixture (developer fixture, not a benchmark)

Provenance: hand-written for this project on 2026-11-06 by Claude (implementation owner), to the
scenario list in handoff 01 section 2. English, 20 documents (`d01`–`d20`), 8 queries (`q01`–`q08`),
160 qrel rows: every query–document pair has an explicit 0/1 judgment (complete judgments).

**Freeze status:** judgments were written with the documents and queries, before any retriever
(BM25 or embedding) was run on this fixture. No retrieval output has been inspected. Any later
label change must be recorded here and in `EXPECTED_POSITIVES` in `tests/test_data.py`.

**Relevance rule:** a document is relevant (1) if a person asking the query would find their
question answered by it; topical overlap or shared vocabulary alone is nonrelevant (0).

| Query | Text | Relevant | Scenario | Rationale |
|---|---|---|---|---|
| q01 | `E4107` | d01 | exact identifier | d01 is the only document defining error E4107 and its fix. |
| q02 | how do I get reimbursed for a business trip | d10 | paraphrase | d10 describes repayment of travel costs via receipts; wording ("repaid", "travel costs") differs from the query. d09 (purchase approval) does not answer it. |
| q03 | regain access to my account | d02, d03 | paraphrase, multiple relevant | d02 (password reset) and d03 (account recovery after losing email) are two distinct routes to regaining access. |
| q04 | how do pythons catch their prey | d07 | ambiguous vocabulary | d07 is about the snake; d06 (Python virtualenv) shares the term but is nonrelevant. |
| q05 | how do researchers estimate jaguar territory | d05 | ambiguous vocabulary | d05 is the animal study; d04 (Jaguar car rental fleet) shares the term but is nonrelevant. |
| q06 | is the path by the river bank open in spring | d14 | ambiguous vocabulary | d14 says the riverside footpath closes April–May, so it is judged to answer the query by reporting that closure; it is not read as a rule for every spring date or location. d15 (financial bank) is nonrelevant. |
| q07 | analytics queries time out, how to fix | d18, d19 | multiple relevant | d18 explicitly gives a fix: add a date filter to the WHERE clause to address missing partition filters. d19 does not give the same instruction; it identifies scanning the full events table instead of a single day's partition as the cause of slow dashboards, which is actionable diagnostic guidance that implies a remedy. Both are judged relevant for this developer fixture. This is a human relevance judgment, not a metric result; the independent review (handoff 02) kept both labels. |
| q08 | what is the policy on pet dogs in the office | none | no relevant document | No document mentions pets; all 20 judgments are explicit 0. |

Distractor/noise documents with no positive judgment for any query: d08, d11, d12, d13, d16, d17, d20.
d11 and d12 (API key rotation) are unused by current queries; they were kept as corpus noise.

These scenario types do not promise a win for either retriever. Synthetic results support a
correctness/demo claim only, not a claim about real-world retrieval quality.

## Checksums

`FIXTURE.sha256` is the **authoritative** manifest for the three fixture data files (`sha256sum -c FIXTURE.sha256`
from this directory). `SHA256SUMS` is a legacy file of unknown origin that does **not** match the current fixture; it is
kept unchanged as evidence and is not authoritative. See `CHECKSUMS.md`.
