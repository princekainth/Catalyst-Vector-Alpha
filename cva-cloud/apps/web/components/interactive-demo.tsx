"use client";

import { useEffect, useState } from "react";

type DemoStep = {
  label: string;
  detail: string;
  tone: string;
};

const steps: DemoStep[] = [
  { label: "DETECT", detail: "ImagePullBackOff on payments-api", tone: "text-danger" },
  { label: "ANALYZE", detail: "Manifest not found: nginx:fake-tag", tone: "text-warning" },
  { label: "DECIDE", detail: "Fix image tag → nginx:latest", tone: "text-accent" },
  { label: "ACT", detail: "Patch deployment + rollout", tone: "text-white" },
  { label: "VERIFY", detail: "Pod Ready in 12s", tone: "text-accent" },
];

export default function InteractiveDemo() {
  const [running, setRunning] = useState(false);
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!running) return;
    if (index >= steps.length) {
      setRunning(false);
      return;
    }
    const timer = setTimeout(() => {
      setIndex((prev) => prev + 1);
    }, 900);
    return () => clearTimeout(timer);
  }, [running, index]);

  const start = () => {
    setIndex(0);
    setRunning(true);
  };

  const reset = () => {
    setRunning(false);
    setIndex(0);
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-[#0f1621]/80 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-white/50">Interactive demo</p>
          <h3 className="mt-2 text-xl font-display">Simulate a real incident in seconds.</h3>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={start}
            disabled={running}
            className="rounded-md bg-accent px-4 py-2 text-xs font-semibold text-black disabled:opacity-60"
          >
            {running ? "Running..." : "Simulate Incident"}
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-white/10 px-4 py-2 text-xs text-white/60"
          >
            Reset
          </button>
        </div>
      </div>
      <div className="mt-6 space-y-3 text-sm">
        {steps.map((step, i) => {
          const active = running && i === index;
          const complete = i < index;
          return (
            <div
              key={step.label}
              className={`rounded-lg border border-white/10 px-4 py-3 ${
                active ? "bg-black/50" : "bg-black/30"
              }`}
            >
              <div className="flex items-center justify-between text-xs text-white/50">
                <span className={step.tone}>{step.label}</span>
                <span>{complete ? "done" : active ? "now" : "queued"}</span>
              </div>
              <p className="text-white/70">{step.detail}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
