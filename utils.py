

from collections import defaultdict


#takes all the raw results from then group by page.
def summarize_results(results):
    # dictionary where each key is a page name
    counts = defaultdict(lambda: {"success": 0, "total": 0})

    for res in results:
        page = res["page"]
        metric = res["metric"]

        # unwrap metrics if they're dictionaries like {"result": "AIA"}
        if isinstance(metric, dict):
            metric_val = metric.get("result")
        else:
            metric_val = metric

        counts[page]["total"] += 1
        #print(metric_val)
        if metric_val != "finished":  # only count as success if not "finished"
            counts[page]["success"] += 1

    summary = []
    for page, vals in counts.items():
        success_rate = (vals["success"] / vals["total"]) * 100
        summary.append({
            "page": page,
            "success_rate": round(success_rate, 1),
            "successes": vals["success"],
            "total": vals["total"]
        })
    return summary
