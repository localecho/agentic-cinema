"""Station Director service: telemetry ingest, programming directives, agent trigger.

Day 1 scope (per the build plan): prove both partner integrations work end to
end before building anything else.
  - POST /telemetry  -> insert a real row into ClickHouse Cloud
  - GET  /programming -> select the latest directive from ClickHouse Cloud
  - POST /director/run -> call Gemini via ADK, have it choose a directive, write it back

None of this is implemented yet. Filled in during the build week.
"""

from fastapi import FastAPI

app = FastAPI(title="Agentic Cinema — Station Director")


@app.get("/health")
def health():
    return {"status": "scaffold"}
