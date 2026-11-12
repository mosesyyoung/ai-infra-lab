import os
from ray import serve


@serve.deployment
class WorkerInfo:

    async def __call__(self, request):

        return {
            "service": "ray-serve-demo",
            "worker_pod": os.environ.get("HOSTNAME"),
            "pid": os.getpid(),
            "message": "Hello from Ray Serve"
        }


app = WorkerInfo.bind()
