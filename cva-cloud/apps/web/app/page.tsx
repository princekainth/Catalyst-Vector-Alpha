import Link from "next/link";
import InteractiveDemo from "@/components/interactive-demo";

const highlights = [
  { label: "Incidents resolved", value: "2,431", note: "+14% this week" },
  { label: "Median remediation", value: "18s", note: "across 12 clusters" },
  { label: "Human approvals", value: "92%", note: "policy driven" },
];

const liveEvents = [
  {
    title: "ImagePullBackOff",
    description: "payment-service",
    detail: "nginx:fake-tag → nginx:latest",
    tone: "text-danger",
  },
  {
    title: "ConfigMap missing",
    description: "checkout-api",
    detail: "created app-config",
    tone: "text-warning",
  },
  {
    title: "OOMKilled",
    description: "recommender",
    detail: "memory +256Mi approved",
    tone: "text-accent",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[#080b11] text-white">
      <div className="relative overflow-hidden">
        <div className="absolute -top-32 right-0 h-72 w-72 rounded-full bg-accent/20 blur-[120px] glow-pulse" />
        <div className="absolute top-60 -left-24 h-80 w-80 rounded-full bg-warning/20 blur-[140px] float-slow" />
        <div className="absolute bottom-0 right-24 h-64 w-64 rounded-full bg-danger/20 blur-[140px] float-slow" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.08),transparent_55%)]" />

        <section className="relative mx-auto max-w-6xl px-6 pb-16 pt-20">
          <div className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-white/70">
                <span className="h-2 w-2 rounded-full bg-accent" />
                Autonomous remediation with human control
              </div>
              <h1 className="text-4xl font-display leading-tight md:text-5xl">
                CVA Cloud keeps your clusters stable while you sleep.
              </h1>
              <p className="text-base text-white/70 md:text-lg">
                Detect, reason, and resolve Kubernetes failures in seconds. Review each action
                before it ships, or let policy auto-approve safe fixes.
              </p>
              <div className="flex flex-wrap gap-4">
                <Link
                  href="/dashboard"
                  className="rounded-md bg-accent px-5 py-3 text-sm font-semibold text-black"
                >
                  Launch Mission Control
                </Link>
                <Link
                  href="/clusters/new"
                  className="rounded-md border border-white/20 px-5 py-3 text-sm font-semibold text-white"
                >
                  Connect a Cluster
                </Link>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                {highlights.map((item) => (
                  <div key={item.label} className="rounded-lg border border-white/10 bg-white/5 p-4">
                    <p className="text-xs text-white/60">{item.label}</p>
                    <p className="mt-2 text-2xl font-display">{item.value}</p>
                    <p className="text-xs text-white/50">{item.note}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="relative">
              <div className="rounded-2xl border border-white/10 bg-[#0f1621]/80 p-6 shadow-xl">
                <div className="flex items-center justify-between">
                  <p className="text-xs text-white/50">Live Response Feed</p>
                  <span className="rounded-full bg-accent/20 px-3 py-1 text-xs text-accent">
                    live
                  </span>
                </div>
                <div className="mt-6 space-y-4">
                  {liveEvents.map((event, idx) => (
                    <div
                      key={event.title}
                      className="relative rounded-xl border border-white/10 bg-black/30 p-4"
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className={`font-semibold ${event.tone}`}>{event.title}</span>
                        <span className="text-white/40">just now</span>
                      </div>
                      <p className="mt-2 text-sm">{event.description}</p>
                      <p className="text-xs text-white/50">{event.detail}</p>
                      {idx === 0 ? <div className="absolute inset-0 shimmer-line" /> : null}
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-white/10 bg-[#101726]/90 p-5">
                <p className="text-xs text-white/50">Reasoning Trace</p>
                <div className="mt-4 space-y-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span>OBSERVE</span>
                    <span className="text-white/40">0.1s</span>
                  </div>
                  <p className="text-white/70">ImagePullBackOff on payment-service</p>
                  <div className="flex items-center justify-between">
                    <span>DECIDE</span>
                    <span className="text-white/40">95%</span>
                  </div>
                  <p className="text-white/70">Fix image tag → nginx:latest</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

      <section className="mx-auto max-w-6xl px-6 pb-16">
        <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <div className="rounded-2xl border border-white/10 bg-[#0f1621]/80 p-6">
            <div className="flex items-center justify-between text-xs text-white/50">
              <span>Product preview</span>
              <span className="rounded-full bg-white/10 px-3 py-1">Live demo</span>
            </div>
            <div className="mt-6 rounded-xl border border-white/10 bg-black/40 p-4">
              <div className="flex items-center justify-between text-xs text-white/50">
                <span>Mission Control</span>
                <span className="rounded-full bg-accent/20 px-2 py-1 text-[10px] text-accent">
                  live
                </span>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <p className="text-white/50">Active Incidents</p>
                  <p className="mt-2 text-lg font-display">5</p>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <p className="text-white/50">Median Fix</p>
                  <p className="mt-2 text-lg font-display">21s</p>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <p className="text-white/50">Auto-Approve</p>
                  <p className="mt-2 text-lg font-display">68%</p>
                </div>
              </div>
              <div className="mt-4 rounded-lg border border-white/10 bg-[#0c121c]/90 p-4">
                <div className="flex items-center justify-between text-[10px] uppercase text-white/40">
                  <span>Incident Feed</span>
                  <span>last 60s</span>
                </div>
                <div className="mt-3 space-y-3 text-xs">
                  {[
                    {
                      title: "ImagePullBackOff",
                      pod: "checkout-api",
                      detail: "nginx:1.2.3 → nginx:latest",
                      tone: "text-danger",
                    },
                    {
                      title: "CrashLoopBackOff",
                      pod: "payments",
                      detail: "missing env var: STRIPE_KEY",
                      tone: "text-warning",
                    },
                    {
                      title: "ConfigMap missing",
                      pod: "inventory",
                      detail: "created app-config",
                      tone: "text-accent",
                    },
                  ].map((row) => (
                    <div
                      key={row.title}
                      className="rounded-md border border-white/10 bg-black/30 p-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className={`font-semibold ${row.tone}`}>{row.title}</span>
                        <span className="text-white/40">just now</span>
                      </div>
                      <p className="text-white/70">{row.pod}</p>
                      <p className="text-white/40">{row.detail}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-4 rounded-lg border border-white/10 bg-[#0b1019]/80 p-3 text-[10px] text-white/50">
                Reasoning Trace: OBSERVE → ANALYZE → DECIDE → ACT → VERIFY
              </div>
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <p className="text-xs text-white/50">Trusted by modern infra teams</p>
            <div className="mt-6 grid grid-cols-2 gap-4 text-sm text-white/70">
              {[
                "Sentry-ready audit trail",
                "SOC2 aligned workflows",
                "Human approvals built in",
                "Zero-touch rollback hooks",
                "Policy engine for prod",
                "Global incident ledger",
              ].map((item) => (
                <div
                  key={item}
                  className="rounded-lg border border-white/10 bg-black/30 px-3 py-3"
                >
                  {item}
                </div>
              ))}
            </div>
            <div className="mt-6 rounded-lg border border-white/10 bg-black/40 px-4 py-3 text-xs text-white/50">
              Add customer logos or security badges here.
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-16">
        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <InteractiveDemo />
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <p className="text-xs text-white/50">Architecture</p>
            <h3 className="mt-3 text-xl font-display">Agent → SaaS → Approval → Fix → Audit</h3>
            <p className="mt-2 text-sm text-white/70">
              Every decision is tracked end-to-end. Agents execute only what you approve,
              and every action is recorded with evidence.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3 text-xs text-white/60">
              {[
                "CVA Agent",
                "Policy Engine",
                "Human Approval",
                "Remediation",
                "Audit Log",
              ].map((node, idx) => (
                <div key={node} className="flex items-center gap-3">
                  <span className="rounded-full border border-white/10 bg-black/40 px-3 py-1">
                    {node}
                  </span>
                  {idx < 4 ? <span className="text-white/30">→</span> : null}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-20">
        <div className="grid gap-6 md:grid-cols-3">
          {[
            {
              title: "Detect",
              copy: "EventGate monitors every pod and surfaces failures in real time.",
            },
            {
              title: "Decide",
              copy: "LLM reasoning summarizes root cause and the safest fix.",
            },
            {
              title: "Ship",
              copy: "Approve with a click or let policy handle safe remediations.",
            },
          ].map((step) => (
            <div key={step.title} className="rounded-xl border border-white/10 bg-white/5 p-6">
              <p className="text-xs text-white/50">Step</p>
              <h3 className="mt-3 text-lg font-display">{step.title}</h3>
              <p className="mt-2 text-sm text-white/70">{step.copy}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-16">
        <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <p className="text-xs text-white/50">Safety & Trust</p>
            <h3 className="mt-3 text-xl font-display">Every fix is auditable and human-approved.</h3>
            <p className="mt-2 text-sm text-white/70">
              CVA ships only what your policies allow. Every action is logged with a reasoning trail,
              evidence, and rollback metadata.
            </p>
            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              {[
                "Human-in-the-loop approvals",
                "Immutable audit log",
                "One-click rollback hooks",
                "RBAC + tenant isolation",
              ].map((item) => (
                <div
                  key={item}
                  className="rounded-lg border border-white/10 bg-black/30 px-4 py-3 text-sm text-white/70"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-[#0f1621]/80 p-6">
            <p className="text-xs text-white/50">Incident Walkthrough</p>
            <div className="mt-4 space-y-3 text-sm">
              {[
                { label: "OBSERVE", detail: "ImagePullBackOff on payments", time: "0.1s" },
                { label: "ANALYZE", detail: "Manifest not found: nginx:fake-tag", time: "0.6s" },
                { label: "DECIDE", detail: "Fix image tag → nginx:latest", time: "2.4s" },
                { label: "VERIFY", detail: "Pod Ready in 12s", time: "12.0s" },
              ].map((step) => (
                <div
                  key={step.label}
                  className="rounded-lg border border-white/10 bg-black/40 px-4 py-3"
                >
                  <div className="flex items-center justify-between text-xs text-white/60">
                    <span>{step.label}</span>
                    <span>{step.time}</span>
                  </div>
                  <p className="text-white/70">{step.detail}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-24">
        <div className="rounded-2xl border border-white/10 bg-white/5 p-8">
          <div className="flex flex-wrap items-center justify-between gap-6">
            <div>
              <p className="text-xs text-white/50">Pricing preview</p>
              <h3 className="mt-2 text-2xl font-display">Start free, upgrade when you scale.</h3>
              <p className="mt-2 text-sm text-white/70">
                Designed for startups and teams scaling production Kubernetes.
              </p>
            </div>
            <Link
              href="/clusters/new"
              className="rounded-md bg-accent px-5 py-3 text-sm font-semibold text-black"
            >
              Connect your first cluster
            </Link>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {[
              { name: "Free", price: "$0", detail: "1 cluster · 100 fixes" },
              { name: "Pro", price: "$99", detail: "5 clusters · 1k fixes" },
              { name: "Team", price: "$299", detail: "20 clusters · unlimited fixes" },
            ].map((tier) => (
              <div
                key={tier.name}
                className="rounded-xl border border-white/10 bg-black/40 p-5"
              >
                <p className="text-xs text-white/50">{tier.name}</p>
                <p className="mt-2 text-2xl font-display">{tier.price}</p>
                <p className="text-sm text-white/70">{tier.detail}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
