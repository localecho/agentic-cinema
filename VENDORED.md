# Vendored file provenance

| File in this repo | Vendored from (private) | As of |
|---|---|---|
| `web/telemetry.js` | `~/ascii-cinema/js/telemetry.js` | 2026-08-31 |
| `web/schedule.js` | `~/ascii-cinema/js/schedule.js` | 2026-08-31 |
| `web/theatre.js` | `~/ascii-cinema/js/theatre.js` | 2026-08-31 |
| `web/decoder.js` | `~/ascii-cinema/js/decoder.js` | 2026-08-31 |
| `web/index.html` | `~/ascii-cinema/index.html` | 2026-08-31 |
| `pipeline/sam2_matte.py` | `~/cowork/cv-skills/sam2_matte.py` | 2026-08-31 |
| `pipeline/ingest_archive_template.py` | `~/blueduck-grants/datamoshing/whatsapp_dive_pipeline.py` | 2026-08-31 |

These are direct copies at the moment of vendoring. Sync manually if the
private source repos change — no automated pull mechanism is set up.
`web/telemetry.js`'s sink URL still needs to be repointed to this project's
own `/telemetry` endpoint before the front end is wired up (Day 2 of the
build plan); `pipeline/ingest_archive_template.py` needs a shot-detection
pre-pass added per the source project's own documented gap (skip title-card
frames before seeding SAM2).
