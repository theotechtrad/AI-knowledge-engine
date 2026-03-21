#!/usr/bin/env python3
"""
Integration tests for Knowledge Engine — all chat features + API endpoints.

Usage:
  1. Start the server:  .venv\\Scripts\\python.exe main.py
  2. Run tests:          .venv\\Scripts\\python.exe test.py

Optional:
  set TEST_SERVER=http://127.0.0.1:5500   (default)

Tests are skipped automatically if the server is not reachable.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from typing import Any

# Default: Flask app in main.py runs on 5500
BASE_URL = os.environ.get("TEST_SERVER", "http://127.0.0.1:5500").rstrip("/")
SESSION = "test-session-py"


def _request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: int = 120,
) -> tuple[int, dict[str, Any] | str]:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            code = resp.getcode()
            try:
                return code, json.loads(raw)
            except json.JSONDecodeError:
                return code, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except (urllib.error.URLError, OSError) as e:
        return -1, str(e)


def server_reachable() -> bool:
    code, _ = _request("GET", "/api/stats")
    return code == 200


def chat(message: str, session_id: str = SESSION) -> dict[str, Any]:
    code, data = _request("POST", "/api/chat", {"message": message, "session_id": session_id})
    if code != 200 or not isinstance(data, dict):
        return {"success": False, "error": f"HTTP {code}: {data}"}
    return data


def stats() -> dict[str, Any]:
    code, data = _request("GET", "/api/stats")
    if code != 200 or not isinstance(data, dict):
        return {"success": False}
    return data


def clear_session(session_id: str = SESSION) -> dict[str, Any]:
    code, data = _request("POST", "/api/clear", {"session_id": session_id})
    if code != 200 or not isinstance(data, dict):
        return {"success": False}
    return data


@unittest.skipUnless(
    server_reachable(),
    f"Server not reachable at {BASE_URL} — start with: python main.py",
)
class TestAPIEndpoints(unittest.TestCase):
    """Health and JSON API."""

    def test_stats(self) -> None:
        code, data = _request("GET", "/api/stats")
        self.assertEqual(code, 200)
        assert isinstance(data, dict)
        self.assertTrue(data.get("success"))
        self.assertIn("stats", data)
        s = data["stats"]
        for key in ("knowledge_entries", "mindmaps", "ideas", "roadmaps", "vector_db_count"):
            self.assertIn(key, s)

    def test_clear(self) -> None:
        r = clear_session()
        self.assertTrue(r.get("success"))


@unittest.skipUnless(
    server_reachable(),
    f"Server not reachable at {BASE_URL}",
)
class TestChatFeatures(unittest.TestCase):
    """
    One test per major route_message feature.
    Requires Groq API keys in .env for LLM-backed tools (mindmap, roadmap, etc.).
    """

    @classmethod
    def setUpClass(cls) -> None:
        clear_session()

    def _assert_ok(self, data: dict[str, Any], msg: str = "") -> None:
        self.assertTrue(data.get("success"), msg or data.get("error", data))

    # --- 1. Time / date (tool, no LLM) ---
    def test_time(self) -> None:
        data = chat("What time is it?")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        self.assertIn("response", data)
        self.assertTrue(any(x in data["response"] for x in ("202", "2024", "2025", "2026", "AM", "PM", ":")))

    # --- 2. List categories (tool, RAG) ---
    def test_list_categories(self) -> None:
        data = chat("Please list all my knowledge categories")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        self.assertIn("response", data)

    # --- 3. Mindmap (tool + LLM) ---
    def test_mindmap(self) -> None:
        data = chat("Create a mindmap for: Testing")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        self.assertIn("%%MINDMAP%%", data["response"])
        self.assertIn("%%ENDMINDMAP%%", data["response"])

    # --- 4. Roadmap (tool + LLM) ---
    def test_roadmap(self) -> None:
        data = chat("Create a learning roadmap for: Testing")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        self.assertEqual(data.get("response_type"), "roadmap")
        self.assertIn("ROADMAP_B64_START", data["response"])
        self.assertIn("ROADMAP_B64_END", data["response"])

    # --- 5. Code creator (llm) ---
    def test_code_creator(self) -> None:
        data = chat("Create code for: a function that returns 42")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "llm")
        self.assertIn("response", data)

    # --- 6. Idea expander (tool + LLM) ---
    def test_idea_expander(self) -> None:
        data = chat("Expand this idea: a todo app for students")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        self.assertGreater(len(data.get("response", "")), 50)

    # --- 7. Step-by-step solver (tool + LLM) ---
    def test_step_solver(self) -> None:
        data = chat("Solve step by step: how to stay consistent with studying")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        self.assertIn("STEP", data["response"].upper())

    # --- 8. Summarizer (tool + LLM) ---
    def test_summarizer(self) -> None:
        data = chat(
            "Summarize: Python is a programming language. It is widely used for web and data science."
        )
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        self.assertIn("SUMMARY", data["response"].upper())

    # --- 9. Wikipedia (tool, network) ---
    def test_wikipedia(self) -> None:
        data = chat("Wikipedia Python programming language")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        self.assertIn("Python", data["response"])

    # --- 10. Save + retrieve knowledge (tool, RAG) ---
    def test_save_and_retrieve(self) -> None:
        unique = f"testtopic-{os.getpid()}"
        save_msg = (
            f"Save this knowledge: topic={unique}, content=hello test fact, category=test"
        )
        d1 = chat(save_msg)
        self._assert_ok(d1)
        self.assertEqual(d1.get("route"), "tool")

        d2 = chat(f"Retrieve knowledge about: {unique}")
        self._assert_ok(d2)
        self.assertEqual(d2.get("route"), "tool")
        self.assertIn("hello test fact", d2["response"].lower())

    # --- 11. Image analysis (tool + vision LLM) — optional tiny PNG ---
    def test_image_analysis(self) -> None:
        # 1x1 transparent PNG
        b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        msg = f"Describe this [IMAGE_DATA:{b64}]"
        data = chat(msg)
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "tool")
        # May contain "IMAGE ANALYSIS" or error note if vision not supported
        self.assertIn("response", data)

    # --- 12. Generic chat (llm fallback) ---
    def test_generic_llm(self) -> None:
        data = chat("Say hello in one short sentence.")
        self._assert_ok(data)
        self.assertEqual(data.get("route"), "llm")


def main() -> int:
    if not server_reachable():
        print(
            f"Note: Server not at {BASE_URL} — tests will be skipped.\n"
            "Start with: python main.py\n",
            file=sys.stderr,
        )
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
