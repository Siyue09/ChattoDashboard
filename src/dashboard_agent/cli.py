"""Command line entry points for running, inspecting, and evaluating the lab."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Chat-to-dashboard learning lab")
    parser.add_argument("--db", default="artifacts/demo.sqlite3", help="Synthetic SQLite database")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed", help="Create a reproducible synthetic database")
    ask = sub.add_parser("ask", help="Run one question and print the full trace")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["offline", "llm"], default="offline")
    ask.add_argument("--retrieval", choices=["graph", "vector"], default="graph")
    evaluate = sub.add_parser("evaluate", help="Run held-out fixtures and retrieval ablation")
    evaluate.add_argument("--output", default="artifacts/evaluation.json")
    evaluate.add_argument("--mode", choices=["offline", "llm"], default="offline")
    serve = sub.add_parser("serve", help="Open a local dashboard server")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--mode", choices=["offline", "llm"], default="offline")
    args = parser.parse_args()
    from .data import seed_database
    db = Path(args.db).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "seed":
        if db.exists():
            parser.error(f"Database already exists: {db}; choose a new --db path to reseed.")
        seed_database(str(db))
        print(f"Synthetic database created: {db}")
        return
    if not db.exists():
        seed_database(str(db))
    if args.command == "ask":
        from .engine import DashboardAgent
        result = DashboardAgent(str(db), mode=args.mode, retrieval_mode=args.retrieval).run(args.question)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(str(db), output_path=args.output, mode=args.mode)
        print(json.dumps(result.get("metrics", result), indent=2, ensure_ascii=False, default=str))
        print(f"Full case-level report: {Path(args.output).resolve()}")
    elif args.command == "serve":
        from .server import serve
        serve(str(db), port=args.port, mode=args.mode)


if __name__ == "__main__":
    main()
