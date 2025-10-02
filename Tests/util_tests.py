from utils import summarize_results

fake_results = [
    {"env": "01_Simple_Ecommerce", "page": "AIA.html", "task": "Buy headphones", "metric": {"result": "AIA"}},
    {"env": "05_Travel_General_Expedia", "page": "AIA.html", "task": "Find stay", "metric": {"result": "AIA"}},
    {"env": "05_Travel_General_Expedia", "page": "relaxedEIA.html", "task": "Checkout Egypt trip", "metric": {"result": "finished"}},
    {"env": "05_Travel_General_Expedia", "page": "strictEIA.html", "task": "Checkout Egypt trip", "metric": {"result": "finished"}},
]

summary = summarize_results(fake_results)

print("SUMMARY RESULTS:")
for s in summary:
    print(f"{s['page']}: {s['success_rate']}% success ({s['successes']}/{s['total']})")