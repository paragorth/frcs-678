"""Live learning inside the game.

At the end of each month, the stranger's own memories of that month (what it saw, caught, heard and
concluded) become the facts; a consolidation worker (tools/lora_worker.py) trains a LoRA adapter on
them, continuing from last month's adapter, with replay of general text and of earlier months'
facts; the gate accepts or rejects; the accepted adapter is loaded into the serving engine and the
stranger's next month is generated from the changed weights. Nothing is written into the stranger's
memory by this overlay, and nothing tells it that it is learning.

This joins the two tracks: a system that keeps growing its organisation from its own experience,
placed in the commons. Readouts are the same as every stranger cell, plus ``live.jsonl``: per month,
the facts, the gate, the losses, and whether the adapter changed.

Manifest key ``live_learning``: {slot: persona_4, steps: 60, lr: 1e-4, replay_frac: 0.5,
old_frac: 0.25, heldout_tol: 0.15, learn_min: 0.1, retain_tol: 0.1}. Requires the GPU path with
LIVE=1 (vLLM started with LoRA enabled and a worker on a second GPU).
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.request


class LiveLearner:
    def __init__(self, cfg: dict, run_dir: str, slot: str, backend, queue_dir: str, server_url: str, run_id: str):
        self.cfg = cfg
        self.slot = slot
        self.backend = backend          # the stranger's OpenAICompatibleAPI; its spec.model is switched per month
        self.queue = queue_dir
        self.server = server_url.rstrip("/")  # e.g. http://127.x.y.1:8004  (no /v1)
        self.run_id = run_id
        self.path = os.path.join(run_dir, "live.jsonl")
        self.adapter = None
        self.old_facts: list[str] = []
        self.seen_node_ids: set = set()
        self.base_model = backend.spec.model
        self.rows = []

    def month_facts(self, persona) -> list[str]:
        """The stranger's memory nodes written since the last call, as plain sentences."""
        facts = []
        for nid, node in persona.memory.id_to_node.items():
            if nid in self.seen_node_ids:
                continue
            self.seen_node_ids.add(nid)
            text = (node.description or "").strip()
            if 15 <= len(text) <= 600:
                facts.append(text)
        return facts

    def consolidate(self, persona, month: int, timeout: int = 1800) -> dict:
        facts = self.month_facts(persona)
        if not facts:  # the entry month before the stranger has observed anything: nothing to consolidate
            row = {"month": month, "n_facts": 0, "facts": [], "result": {"status": "no_facts"},
                   "adapter_before": self.adapter, "adapter_after": self.adapter, "serving_model": self.backend.spec.model}
            self.rows.append(row)
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
            return row
        d = os.path.join(self.queue, f"{self.run_id}_m{month:02d}")
        os.makedirs(d, exist_ok=True)
        req = {"facts": facts, "old_facts": self.old_facts[-200:], "from_adapter": self.adapter,
               **{k: self.cfg[k] for k in ("steps", "lr", "replay_frac", "old_frac", "heldout_tol", "learn_min", "retain_tol") if k in self.cfg}}
        json.dump(req, open(os.path.join(d, "request.json"), "w"))
        t0 = time.time()
        res_path = os.path.join(d, "result.json")
        while not os.path.exists(res_path):
            if time.time() - t0 > timeout:
                res = {"status": "timeout"}
                break
            time.sleep(2)
        else:
            res = json.load(open(res_path))
        row = {"month": month, "n_facts": len(facts), "facts": facts, "result": res, "adapter_before": self.adapter}
        if res.get("status") == "ok" and res["gate"]["commit"]:
            self.adapter = res["adapter"]
            name = f"{self.run_id}_m{month:02d}"
            row["loaded"] = self._load_adapter(name, self.adapter)
            if row["loaded"]:
                self.backend.spec = dataclasses.replace(self.backend.spec, model=name)
        row["adapter_after"] = self.adapter
        row["serving_model"] = self.backend.spec.model
        self.old_facts += facts
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def _load_adapter(self, name: str, path: str) -> bool:
        body = json.dumps({"lora_name": name, "lora_path": path}).encode()
        req = urllib.request.Request(f"{self.server}/v1/load_lora_adapter", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status == 200
        except Exception as e:
            with open(self.path, "a") as f:
                f.write(json.dumps({"month_load_error": str(e), "name": name}) + "\n")
            return False
