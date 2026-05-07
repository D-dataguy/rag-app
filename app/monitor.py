import time
import json
import os
from datetime import datetime

LOG_FILE = "logs/requests.json"

def ensure_log_dir():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

def log_request(question: str, answer: str, chunks: list, latency_ms: float, model: str = "gpt-3.5-turbo"):
    ensure_log_dir()
    
    # Calculate cost (gpt-3.5-turbo pricing)
    input_tokens = len(question.split()) * 1.3
    output_tokens = len(answer.split()) * 1.3
    cost_usd = (input_tokens * 0.0000005) + (output_tokens * 0.0000015)

    # Calculate a simple quality score
    quality_score = min(len(chunks) / 3.0, 1.0)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer_length": len(answer),
        "num_chunks_retrieved": len(chunks),
        "latency_ms": round(latency_ms, 2),
        "estimated_cost_usd": round(cost_usd, 6),
        "quality_score": round(quality_score, 2),
        "model": model
    }

    with open(LOG_FILE, "r+") as f:
        logs = json.load(f)
        logs.append(record)
        f.seek(0)
        json.dump(logs, f, indent=2)

    return record

def get_metrics():
    ensure_log_dir()
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    if not logs:
        return {"message": "No requests logged yet"}

    latencies = [l["latency_ms"] for l in logs]
    costs = [l["estimated_cost_usd"] for l in logs]
    quality_scores = [l["quality_score"] for l in logs]

    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2]
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]

    return {
        "total_requests": len(logs),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "total_cost_usd": round(sum(costs), 6),
        "avg_cost_per_request": round(sum(costs) / len(costs), 6),
        "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 2),
        "recent_requests": logs[-5:]
    }