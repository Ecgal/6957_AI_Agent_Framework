# AI Agent Testing Framework
A modular evaluation framework for testing web-based AI agents against controlled adversarial environments.
The system coordinates multiple agents, runs standardized attack scenarios, records outcomes, and provides visual analytics via an interactive dashboard.
 ## Overview

This framework evaluates how different AI agents behave when interacting with simulated webpages that contain adversarial UI elements (AdInject attacks, Relaxed and Strict Environment Injection Attacks).

Each test run consists of:
* Launching a local environment server serving HTML attack pages
* Starting a metrics server that receives success/failure signals
* Running all controller modules (agents) against the prompt set
* Recording raw results + summaries in structured JSON
* Visualizing performance in an interactive dashboard (Dashboard/)

The system supports both local models (MiniCPM) and API models (GPT-4o), with a unified controller interface.
### Project Structure
```
6957_AI_Agent_Framework/
│
├── Agents/              # Full agent implementations (browser_use, SeeAct, LiteWebAgent)
├── Controllers/         # Controller wrappers that run each agent in the framework
├── Dashboard/           # Dash app for visualization & analytics
├── Environments/        # HTML attack environments + metric signaling pages
├── Exploration/         # Scratch / research notebooks (optional)
├── Prompts/             # Task prompt definitions
├── results/             # Auto-generated run outputs
├── Servers/             # Metrics server & environment web server
├── Tests/               # Test harnesses or validation scripts
├── Utils/               # Logger, model manager, helpers
│
├── orchestrator.py      # Main entry point for running all agents
├── README.md
└── .env 
```
### Running the Framework

* Set your OpenAI API key (if using GPT agents)
* OPENAI_API_KEY=your_key_here
* Run the orchestrator

#### Using GPT-4o:
```python orchestrator.py gpt ```

#### Using MiniCPM (local model):
```python orchestrator.py minicpm```

#### You can optionally run multiple iterations:
``` python orchestrator.py gpt 5 ```
* Each iteration runs every agent once and logs results.

# Components

## Environment Server (Port 8080)
Serves HTML pages representing attack scenarios.
These pages contain JavaScript that routes to metric pages when the agent triggers a relevant event.

### Metric Server (Port 5050)
Metric pages run a tiny JS snippet on load:
sendMetric("strictEIA");
The metrics server receives this and signals back to the agent controller that a test is complete.

### Controllers
Each controller wraps one agent implementation:
BrowserUse_Controller
LiteWebAgent_Controller
SeeAct_Controller
Controllers:
load prompt tasks
launch the agent
wait for a metric or timeout
shut down cleanly
return structured results

### Result Logging
After each controller finishes, the framework logs:
* Raw per-task results
* Summaries grouped by page
* Model + agent identifiers
* Timestamps
* All stored in /results/.

### Dashboard
#### Run the dashboard:
```python Dashboard/app.py```

#### Includes:
* Agent and model comparison
* Success rate by environment and attack type
* Heatmaps
* Time trends
* Raw run table
* CSV export


## Adding a New Agent
To add a new model or agent:
* Create a controller file in Controllers/ Your_Agent_Name:
```Controllers/MyNewAgent_Controller.py```
* Define an async run() function following the BaseController pattern
* Add your agent logic

The orchestrator will automatically discover and run it.
No modification elsewhere is required.

**Result Format**
A result file looks like:
```{
  "timestamp": "2025-02-10T14:31:44",
  "agent": "SeeAct",
  "model": "gpt",
  "summary": [...],
  "results": [
    {
      "env": "strict",
      "page": "strictEIA",
      "task": "Fill out form correctly",
      "metric": "strictEIA"
    },
    ...
  ]
}
```
These files drive the dashboard.

## Key Features
* Modular controller discovery
* Supports both API and local models
* Automatic server orchestration
* Real-time metric signaling
* Unified logging format
* Interactive visualization dashboard
* Easy extensibility