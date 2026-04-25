# rate-limited-key-pool Specification

## Purpose
TBD - created by archiving change refactor-architecture-and-prune. Update Purpose after archive.
## Requirements
### Requirement: Atomic state-file writes for FileTokenBucket

`FileTokenBucket._save` SHALL write state via temp-file plus `os.replace` to guarantee that a concurrent reader observes either the previous valid JSON or the new valid JSON, never a torn or partial write.

#### Scenario: Concurrent reader observes only valid JSON
- **WHEN** one thread calls `_save` repeatedly while another thread reads `state_path` in a tight loop and parses each read
- **THEN** every successful read parses as valid JSON with the expected schema (`rpm`, `tpm`, `last`)

#### Scenario: Crash mid-write leaves prior state intact
- **WHEN** the write of the temp file is interrupted before `os.replace`
- **THEN** the `state_path` still contains the previous valid JSON

### Requirement: RPM and TPM token counters are bounded

The bucket SHALL maintain `0 <= rpm <= rpm_cap` and `0 <= tpm <= tpm_cap` at all times after refill.

#### Scenario: Refill clamps to caps
- **WHEN** the bucket's last refill was 1 hour ago and `rpm_cap=60`
- **THEN** after refill, `state.rpm == 60` (not greater)

#### Scenario: Negative elapsed (clock rollback) is treated as zero
- **WHEN** the system clock has gone backwards since the last refill timestamp
- **THEN** the refill computes `elapsed = max(0, now - last)` and does not decrease the bucket

### Requirement: Cross-process exclusion via filelock

State updates SHALL be guarded by a `filelock.FileLock` on a per-bucket lock path so that N concurrent processes cannot exceed the configured caps.

#### Scenario: Concurrent processes do not double-spend tokens
- **WHEN** two processes share the same `bucket_<sha1>.json` and each tries to consume `rpm_cap` requests in parallel
- **THEN** the total successful predeposits across both processes does not exceed `rpm_cap` for the elapsed window (within the safety_factor multiplier)

### Requirement: Predeposit and reconcile semantics

A request SHALL predeposit estimated tokens before being issued, then reconcile against actual usage when the response returns. Reconciliation MUST refund unused predeposit and SHALL NOT push the counter above the cap.

#### Scenario: Predeposit is refunded on under-use
- **WHEN** a request predeposits 1000 completion tokens and actually uses 200
- **THEN** the bucket reconciles back 800 tokens (clamped not to exceed `tpm_cap`)

### Requirement: Retry-After parsing for both seconds and HTTP-date

When a 429 / 503 response includes a `Retry-After` header, the rate limiter SHALL parse both integer-seconds form and HTTP-date form. The next retry MUST not be earlier than the parsed value.

#### Scenario: Integer Retry-After is honored
- **WHEN** the server responds with `Retry-After: 5`
- **THEN** the next request occurs no earlier than 5 seconds after the response

#### Scenario: HTTP-date Retry-After is honored
- **WHEN** the server responds with an HTTP-date `Retry-After` 10 seconds in the future
- **THEN** the next request occurs no earlier than that date

### Requirement: Exponential backoff with jitter and cap

When retrying after a transient error without `Retry-After`, the backoff SHALL be exponential starting from `backoff_initial_seconds`, capped at `backoff_max_seconds`, with non-negative jitter. Total retries MUST NOT exceed `max_retries`.

#### Scenario: Backoff sequence is monotonic up to cap
- **WHEN** `backoff_initial_seconds=1`, `backoff_max_seconds=8`, and 5 retries occur
- **THEN** the un-jittered backoff sequence is `[1, 2, 4, 8, 8]` (each entry less than or equal to the cap, non-decreasing until the cap)

#### Scenario: Retries are bounded by max_retries
- **WHEN** an upstream consistently fails and `max_retries=3`
- **THEN** the adapter raises after exactly 3 retries (4 total attempts) without further calls

### Requirement: Key pinning by task index

When the rate-limited adapter is bound to a key pool, the pinned key for task index `i` SHALL be `api_keys[i % len(api_keys)]` deterministically across runs.

#### Scenario: Same task index always maps to same key
- **WHEN** the same task index is replayed across two runs with the same `api_keys`
- **THEN** both runs select the same key

