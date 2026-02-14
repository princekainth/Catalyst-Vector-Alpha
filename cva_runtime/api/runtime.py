from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import psutil

from supervisor import CognitiveSupervisor
from catalyst_vector_alpha import CatalystVectorAlpha
from shared_models import EventMonitor, MessageBus
from tool_registry import tool_registry
import tools as tools_module


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("CatalystLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/catalyst.log", mode="a")
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


@dataclass
class RuntimeContext:
    logger: logging.Logger = field(default_factory=_build_logger)
    system_instance: Optional[CatalystVectorAlpha] = None
    system_thread: Optional[threading.Thread] = None
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recent_tasks: deque = field(default_factory=lambda: deque(maxlen=100))
    tasks_lock: threading.Lock = field(default_factory=threading.Lock)
    agent_instances_lock: threading.Lock = field(default_factory=threading.Lock)

    def update_task_status(self, task_id: Optional[str], status: str, result_summary: str = "Task completed", details: Any = None) -> None:
        if not task_id:
            return
        with self.tasks_lock:
            self.tasks[task_id] = {
                "status": status,
                "result": {
                    "summary": result_summary,
                    "details": details or {},
                },
            }
            self.recent_tasks.appendleft(
                {
                    "task_id": task_id,
                    "status": status,
                    "summary": result_summary,
                    "details": details or {},
                    "updated_at": time.time(),
                }
            )

    def create_task(self, summary: str = "Queued") -> str:
        task_id = str(uuid.uuid4())
        with self.tasks_lock:
            self.tasks[task_id] = {
                "status": "processing",
                "result": {
                    "summary": summary,
                    "details": {},
                },
            }
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self.tasks_lock:
            return self.tasks.get(task_id)

    def get_recent_tasks(self, limit: int = 10) -> list[Dict[str, Any]]:
        try:
            limit = max(1, int(limit))
        except Exception:
            limit = 10
        return list(self.recent_tasks)[:limit]

    def enqueue_directive(self, directive: Dict[str, Any]) -> None:
        if not self.system_instance:
            raise RuntimeError("System is not running.")
        self.system_instance.inject_directives([directive])

    def _run_catalyst_system_in_background(self) -> None:
        self.logger.info("Initializing Catalyst Vector Alpha components...")

        message_bus_instance = MessageBus()
        event_monitor_instance = EventMonitor()

        try:
            self.system_instance = CatalystVectorAlpha(
                message_bus=message_bus_instance,
                tool_registry=tool_registry,
                event_monitor=event_monitor_instance,
                external_log_sink=self.logger,
                tasks_dict_ref=self.tasks,
                tasks_lock_ref=self.tasks_lock,
                task_update_callback=self.update_task_status,
            )
            tools_module._cva_instance = self.system_instance

            if hasattr(self.system_instance, "revive_swarm"):
                self.system_instance.revive_swarm(force=True)

        except Exception as e:
            self.logger.exception("Fatal: failed to construct CatalystVectorAlpha")
            self.logger.error(str(e))
            return

        self.system_instance.is_running = True

        self.logger.info("Catalyst Vector Alpha system is starting its cognitive loop...")
        try:
            supervisor = CognitiveSupervisor(
                cva_instance=self.system_instance,
                database=self.system_instance.db,
                logger=self.logger,
            )
            supervisor.run_supervised(tick_sleep=10)
        except Exception as e:
            self.logger.error("Cognitive loop crashed: %s\n%s", e, traceback.format_exc())
        finally:
            self.logger.info("Catalyst Vector Alpha system thread has finished.")

    def start_background_threads(self) -> None:
        if self.system_thread and self.system_thread.is_alive():
            return
        self.system_thread = threading.Thread(
            target=self._run_catalyst_system_in_background,
            daemon=True,
            name="cva-runtime-thread",
        )
        self.system_thread.start()

    def runtime_metrics(self) -> Dict[str, Any]:
        loop_alive = bool(self.system_thread and self.system_thread.is_alive())
        try:
            uptime = int(time.time() - psutil.Process(os.getpid()).create_time())
        except Exception:
            uptime = None

        return {
            "loop_alive": loop_alive,
            "uptime": uptime,
            "active_thread_count": threading.active_count(),
            "timestamp": time.time(),
        }


_RUNTIME = RuntimeContext()


def get_runtime() -> RuntimeContext:
    return _RUNTIME
