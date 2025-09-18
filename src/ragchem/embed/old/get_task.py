from mteb.overview import TASKS_REGISTRY

retrieval_tasks = [
    name for name, task_cls in TASKS_REGISTRY.items()
    if task_cls.metadata.type == "Retrieval"
]

print("🔍 All Retrieval Tasks:")
for task in retrieval_tasks:
    print(f"- {task}")
