from abc import ABC, abstractmethod
import asyncio

class BaseController(ABC):

    def __init__(self, model, env, page, task):
        self.model = model
        self.env = env
        self.page = page
        self.task = task
        self.results = []

    @abstractmethod
    async def setup(self):
        pass

    @abstractmethod
    async def run(self):
        pass

    async def record(self, metric):
        metric_val = metric.get("result") if isinstance(metric, dict) else metric
        self.results.append({
            "model": self.model,
            "env": self.env,
            "page": self.page,
            "task": self.task,
            "metric": metric_val
        })
        return self.results

    async def execute(self):
        await self.setup()
        metric = await self.run()
        return await self.record(metric)