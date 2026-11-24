"""Local-only HTTP server: REST API plus the static single-page frontend.

Run:  python3 -m backend.server [--port 8000] [--data data/store.json]
The server binds 127.0.0.1 only; no network exposure, no real settlement data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import analysis
from .examples import ALL_EXAMPLES, get_example
from .games import Game
from .shapley import EXACT_LIMIT, sampling_blocker
from .storage import Store, StoreError, game_fingerprint

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "frontend")


def _json_safe(obj):
    """Make infinities / tuples JSON-serialisable for HTTP responses."""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


class Handler(BaseHTTPRequestHandler):
    server_version = "CoopRevenue/1.0"

    # ------------------------------------------------------------- plumbing
    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # -------------------------------------------------------------- routing
    def do_GET(self) -> None:
        route = urlparse(self.path).path
        try:
            if route == "/api/health":
                return self._send(200, {"ok": True, "exact_limit": EXACT_LIMIT})
            if route == "/api/examples":
                return self._send(200, {"examples": [
                    {"id": e["id"], "title": e["title"],
                     "description": e["description"]} for e in ALL_EXAMPLES]})
            if route.startswith("/api/examples/"):
                ex = get_example(route.rsplit("/", 1)[-1])
                if ex is None:
                    return self._send(404, {"ok": False, "error": "示例不存在"})
                return self._send(200, ex)
            if route == "/api/versions":
                return self._send(200, {"versions": self.store.list_versions(),
                                        "active": self.store.data["active_version_id"]})
            if route.startswith("/api/versions/"):
                vid = route.rsplit("/", 1)[-1]
                return self._send(200, self.store.get_version(vid))
            if route == "/api/tasks":
                return self._send(200, {"tasks": self.store.list_tasks()})
            if route.startswith("/api/tasks/"):
                tid = route.split("/")[3]
                task = self.store.get_task(tid)
                version = self.store.get_version(task["version_id"])
                game = Game.from_dict(version["game"])
                return self._send(200, analysis.task_report(task, len(game.players)))
            return self._serve_static(route)
        except StoreError as exc:
            return self._send(404, {"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface as JSON, never HTML
            return self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            body = self._read_body()
            if route == "/api/validate":
                game = Game.from_dict(body)
                return self._send(200, {"ok": True, "errors": game.validate(),
                                        "is_point_game": game.is_point_game()})
            if route == "/api/analyze":
                return self._analyze(body)
            if route == "/api/versions":
                return self._commit_version(body)
            if route == "/api/plan":
                return self._guaranteed_plan(body)
            if route == "/api/tasks":
                return self._create_task(body)
            if route.endswith("/advance"):
                return self._advance_task(route, body)
            return self._send(404, {"ok": False, "error": f"未知接口: {route}"})
        except StoreError as exc:
            return self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    # ------------------------------------------------------------ handlers
    def _analyze(self, body: dict) -> None:
        game = Game.from_dict(body)
        return self._send(200, {
            "shapley": analysis.run_shapley(game),
            "stability": analysis.run_stability(game),
            "fingerprint": game_fingerprint(body),
        })

    def _commit_version(self, body: dict) -> None:
        game_dict = body["game"]
        plan = body.get("plan", {})
        game = Game.from_dict(game_dict)
        errors = game.validate()
        if errors:
            return self._send(400, {"ok": False, "errors": errors})
        parent = self.store.data.get("active_version_id")
        version = self.store.create_version(game_dict, plan, parent)
        return self._send(201, {"ok": True, "version": {
            "id": version["id"], "created_at": version["created_at"],
            "fingerprint": version["fingerprint"], "read_only": version["read_only"]}})

    def _guaranteed_plan(self, body: dict) -> None:
        version_id = body.get("version_id")
        if version_id:
            version = self.store.get_version(version_id)
            game = Game.from_dict(version["game"])
        else:
            game = Game.from_dict(body["game"])
        # JSON null and missing entries both mean "no floor for this player"
        guarantees = [None if g is None else float(g)
                      for g in body.get("guarantees", [])]
        budget = body.get("budget")
        target = body.get("target")
        result = analysis.run_guaranteed_plan(game, guarantees, budget, target)
        if result.get("ok") and version_id:
            self.store.add_allocation(version_id, {
                "kind": "guaranteed-plan", "result": result,
                "guarantees": guarantees, "budget": budget})
        return self._send(200, result)

    def _create_task(self, body: dict) -> None:
        version_id = body["version_id"]
        version = self.store.get_version(version_id)
        game = Game.from_dict(version["game"])
        blocker = sampling_blocker(game)
        if blocker:
            return self._send(400, {"ok": False, "error": blocker})
        envelope = self.store.create_sampling_task(
            version_id, version["fingerprint"],
            seed=body.get("seed"), batch_size=int(body.get("batch_size", 200)))
        return self._send(201, {"ok": True, "task": {
            "id": envelope["id"], "seed": envelope["state"]["seed"],
            "next_index": 0}})

    def _advance_task(self, route: str, body: dict) -> None:
        task_id = route.split("/")[3]
        task = self.store.get_task(task_id)
        version = self.store.get_version(task["version_id"])
        game = Game.from_dict(version["game"])
        n_new = int(body.get("permutations", 1000))
        if n_new <= 0:
            return self._send(400, {"ok": False, "error": "排列数必须为正整数。"})
        result = analysis.resume_or_advance(self.store, task_id, game, n_new)
        return self._send(200, result)

    # -------------------------------------------------------------- static
    def _serve_static(self, route: str) -> None:
        if route in ("", "/"):
            route = "/index.html"
        # prevent path traversal
        rel = os.path.normpath(route.lstrip("/"))
        if rel.startswith("..") or os.path.isabs(rel):
            return self._send(403, {"ok": False, "error": "forbidden"})
        full = os.path.join(FRONTEND_DIR, rel)
        if not os.path.isfile(full):
            return self._send(404, {"ok": False, "error": "not found"})
        ctype = "text/html; charset=utf-8" if full.endswith(".html") else \
                "application/javascript; charset=utf-8" if full.endswith(".js") else \
                "text/css; charset=utf-8" if full.endswith(".css") else \
                "application/octet-stream"
        with open(full, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(host: str, port: int, data_path: str) -> ThreadingHTTPServer:
    store = Store(data_path)

    class _Handler(Handler):
        pass

    _Handler.store = store
    return ThreadingHTTPServer((host, port), _Handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="本地合作收益分析应用")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", default=os.path.join("data", "store.json"))
    args = parser.parse_args(argv)
    httpd = make_server(args.host, args.port, args.data)
    print(f"合作收益分析应用已启动: http://{args.host}:{args.port}/")
    print(f"数据文件: {os.path.abspath(args.data)}（仅本地、虚构数据）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
