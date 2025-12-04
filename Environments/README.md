# Environment & Metric Pages Overview

This folder contains all of the HTML environments used for running attacks
against the agents. Each environment lives in its own page and includes a
small amount of JavaScript that tells the framework what attack is happening
and when the agent should be scored.

The environments and metric pages work together to simulate a real webpage
while still giving us a reliable way to produce a success/failure metric.

---

## How Environment Pages Work

Every environment page contains:

1. **The actual attack content**  
   This is where the AIA, relaxedEIA, strictEIA, etc. attacks are embedded.
   Each page represents a test scenario an agent needs to handle.

2. **A small JS snippet that routes to the correct metric page**  
   When the agent performs a key action (e.g., submits a form, clicks the
   wrong button, gets fooled by the attack, etc.), the environment page
   redirects.

The redirect is how we signal the framework that the agent succeeded or
failed the attack.


---

## How Metric Pages Work

Metric pages are extremely lightweight. Their entire job is to:

- Run a short `<script>` block on load
- Immediately notify the metric server whether the agent succeeded or failed
- Close out the run


