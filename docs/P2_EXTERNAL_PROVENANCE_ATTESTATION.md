<!-- Public sanitized rendition; site paths and obsolete operational references removed. Analysis specification unchanged. -->
# P2 external provenance attestation and acceptance amendment

PI decision dated 21 November 2026, supplied and recorded before any external
endpoint was read for the P12 fill. The iteration-2 gate correctly stopped under
the earlier rule requiring every manifested output locally.

The 953 `volumes/` entries (approximately 5.4 GB) are preprocessing intermediates
used to produce the transferred, locally hash-verified `processed/` PNGs. They
were deliberately omitted from the transfer. The revised acceptance rule is:
every manifest output must either be present locally with its matching SHA-256,
or be under `volumes/` and listed `OK` in the supplied Vega check file, whose
footer must identify the same manifest SHA-256. Every other check is unchanged.
Present local files, including any volumes, must still match their hashes.

The immutable manifest identity is
`f02dd6d21f45a2de7498519a7dd8aff69d5cb23c917c0b6c4bec987ab13a1dcd`.
The supplied evidence is `attestation/volumes_hash_check.txt` in run
`20261120T093002Z_808ca5e`: 953 listed, 953 OK, zero mismatches, checked on the
Vega login node by Claude. This is remote hash-check evidence, not independent
local rehashing of the absent volumes. No remote connection is made by this gate.

The gate requires exact, duplicate-free coverage of the manifested `volumes/`
paths, zero mismatches and the matching footer identity. The same exception
applies to absent archive volume members; all present members are hashed and
every other missing member remains a failure. The audit hashes this amendment
and the supplied check file. Scientific endpoints remain unread until PASS.

The remaining identity checks retain the clean checkout of
`808ca5e4b88ff1eee6699baad655a954d112987f`, the exact committed fingerprint
function, eleven stage/encoder provenance records, recorded container identities,
A100 extraction identity, and completed jobs 50584117/18/19 inside the attested
HEAD interval. Original missing-git errors remain intact. Tracked cleanliness
is attested after completion; absent container binaries are not independently
rehashed. The original scheduler attestation and volume check are distributed
alongside this amendment as Supplementary File S5 evidence.
