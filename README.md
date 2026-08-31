# Agentic Cinema

A live, ten-channel ASCII television station with a Gemini-powered "Station
Director" agent. Viewer dwell/flip telemetry streams into ClickHouse Cloud;
the agent queries it, diagnoses underperforming channels and dead-air
dayparts, and issues deterministic programming directives — reordering
lineups, scheduling interstitials, and commissioning new archive films for
ingest through a computer-vision pipeline that converts public-domain cinema
into broadcast-ready ASCII.

Built for the [Devpost "Agentic Cinema" hackathon](https://agentic-cinema.devpost.com)
(ClickHouse partner track), submission deadline 2026-09-09.

## Architecture

```
Browser (station, Vercel)          Cloud Run service (GCP)         ClickHouse Cloud
  telemetry.js  ---POST--->  /telemetry  ---insert--->  events table
  director.js   <---GET----  /programming <---select---  directives table
                              /director/run
                                 |
                              Gemini 2.5 Flash via ADK (Vertex AI Agent Builder)
```

- `web/` — the station front end (ASCII scheduler, telemetry, theatre rendering),
  vendored from a private sibling project and adapted to point at this
  project's own telemetry sink.
- `service/` — the Station Director: a FastAPI service running an ADK agent
  that queries ClickHouse and writes back programming directives.
- `pipeline/` — the computer-vision archive-ingest lane (SAM2 subject
  matting), vendored from a private sibling project, for commissioning new
  films onto a channel.

## Status

Scaffold stage. Day 1 of the build plan (see the project's own planning notes)
verifies both partner API integrations — Gemini via Vertex AI and a real
ClickHouse insert/select — end to end before anything else is built.

## Provenance

`web/` and `pipeline/` vendor code from the author's private repositories
(`ascii-cinema`, `blueduck-grants/datamoshing`, `cowork/cv-skills`). This repo
is the public, MIT-licensed face of that work for the hackathon submission —
see each vendored file's origin noted in `VENDORED.md`.

## License

MIT — see `LICENSE`.
