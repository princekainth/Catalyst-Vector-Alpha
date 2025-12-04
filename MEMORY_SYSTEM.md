MEMORY_SYSTEM.md — Catalyst Vector Alpha Memory Architecture
Overview

CVA’s memory system is a three-tier hybrid cognitive architecture combining:

Short-term memory (STM) for raw events

Mid-term compressed memory for episodic abstraction

Long-term semantic memory (LTM) via vector embeddings

Persistent JSONL logs for replay and auditing

This system enables CVA to learn across cycles, recognize patterns, inform planning, and maintain context in a way similar to biological cognition (sensory → episodic → conceptual layers).

Memory is stored locally — no cloud dependency.

1. Short-Term Memory (STM) — “Raw Experience Buffer”
Structure

CVA uses a deque with a fixed max length (default: 100 entries) to store raw memories:

self.memories = deque(maxlen=100)

Contents

Each memory entry contains:

timestamp

type (TaskOutcome, ToolUse, IntentAdaptation, etc.)

content (tool result, task details, message payload)

related task/event IDs

cycle reference

source agent

When Agents Create STM Memories

Agents log raw memories whenever they:

execute tasks

use tools

adapt intent

send/receive messages

detect system alerts

reflect on state

Purpose

STM acts as working memory:

fuels reflection

helps detect stagnation

lets agents recall what happened in the last N cycles

provides raw input for compression into higher-level episodes

STM is temporal, noisy, and high-granularity.

2. Compressed Memory — “Episodic Abstraction Layer”
Trigger

When the STM buffer fills or at scheduled intervals:

summarize_and_compress_memories(...)

Process

Concatenate a batch of raw memories

Use LLM summarization to generate one coherent episode summary

Generate an embedding for vector search

Create a CompressedMemory entry:

{
   "timestamp": "...",
   "type": "CompressedMemory",
   "summary": "...",
   "embedding": [...],
   "original_memory_count": 23
}


Store compressed memory in:

in-memory compressed deque

ChromaDB (long-term)

disk logs (backup)

Prune original raw memories

Purpose

This layer:

condenses noise into meaning

creates episodes for meta-reasoning

forms the “middle-term” history that agents can reflect on

reduces cognitive clutter

Equivalent to a human taking a chaotic day and summarizing it into “a story”.

3. Long-Term Memory (LTM) — “Semantic Knowledge Base”
Backend

CVA uses ChromaDB as a vector store for compressed memory embeddings.

Querying LTM

Agents can perform semantic retrieval:

results = query_long_term_memory("what happened last time CPU spiked?")


Search returns:

summaries

timestamps

relevance scores

Purpose

LTM is used for:

planning context

pattern recognition

trend detection

preventing repeated mistakes

cross-episode learning

contextualizing new tasks

This forms CVA’s conceptual memory.

4. Memory Retrieval During Cognition
Agents Retrieve Memory When:

Reflecting
_generate_reflection_narrative() builds prompts based on STM + compressed memory.

Planning
Planner uses LTM to provide historical context for tasks.

Stagnation Detection
Agents compare recent cycles to detect loop repetition.

Behavioral Adaptation
Agents adjust strategies based on past failures/successes.

Example (Reflection Prompt)
Generate reflection for ProtoAgent_Planner
Recent memories: [...]

Cognitive Purpose

Memory retrieval lets CVA:

avoid forgetting

avoid repeating errors

learn across time

reason with historical continuity

This is what gives CVA identity across cycles.

5. Persistence & Reconstructability

CVA writes all memory to disk for:

transparency

replay

debugging

offline training

Formats:

JSONL logs: every memory event from every agent

SQLite: agent state, task history, snapshots

ChromaDB: long-term embeddings

Checkpoint files: orchestrator + agents state serialization

Why This Matters

Even if CVA restarts:

its short-term + mid-term memory rehydrates

long-term memory persists

nothing is forgotten

learning continues

This gives CVA lifelong continuity, not “stateless LLM randomness”.

6. Cognitive Interpretation

Biologically analogous:

CVA Component	Cognitive Analog
STM (deque)	Working memory / sensory buffer
Compressed Memory	Episodic memory
Chroma LTM	Conceptual memory / semantic knowledge
Reflection	Metacognition
Pattern detection	Behavioral learning
Stagnation detection	Self-correction

CVA learns by compressing experience → extracting meaning → retrieving patterns → adjusting behavior.

This is not just storage — it’s cognition.

7. Safety & Stability in Memory

CVA protects memory integrity:

failed LLM generations raise errors

timeouts prevent memory stalls

compression has guardrails

vector store failures fall back to logs

pruning prevents memory overload

no silent degradation allowed

This ensures memory remains:

clean

concise

consistent

reliable

Conclusion

CVA’s memory architecture is one of its strongest components.
It supports:

learning

pattern recognition

episodic abstraction

semantic reasoning

self-reflection

adaptive planning

lifelong continuity

This system transforms CVA from a rule engine into an actual thinking environment with internal experience and evolving behavior.
