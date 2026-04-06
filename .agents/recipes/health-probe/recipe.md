---
name: health-probe
description: Verify the inference API and Claude CLI are operational
trigger: schedule
tool: claude-code
timeout_minutes: 3
max_turns: 1
permissions:
  contents: read
---

# Health Probe

Reply with exactly: HEALTH_CHECK_OK

Do not use any tools. Do not read any files. Just reply with the text above.
