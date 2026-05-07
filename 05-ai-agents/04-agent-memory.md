# Agent Memory

## The Problem / Why This Matters

Without memory, every agent interaction starts from zero. The agent doesn't remember: what it did 5 minutes ago (unless still in context window), user preferences from previous sessions, lessons learned from past failures, or facts it discovered during earlier tasks. This creates frustrating user experiences ("I already told you my preferences!"), inefficient workflows (rediscovering the same information repeatedly), and inability to learn from mistakes (making the same errors across sessions). Human assistants are valuable partly because they accumulate knowledge over time — they learn your preferences, build relationships, develop institutional knowledge. For AI agents to be similarly valuable, they need memory systems: short-term memory (current task context), working memory (intermediate results and scratchpad), long-term memory (persistent knowledge across sessions), and episodic memory (recollection of specific past events). In 2026, memory is the differentiator between simple chatbots and truly useful AI assistants. The engineering challenge is: what to remember (everything is too much, nothing is too little), how to organize it (retrievable when relevant), how to update it (knowledge changes), and how to forget (outdated information should fade).

---

## The Analogy

Think of agent memory like human memory systems:

- **Short-term / working memory** = Your mental scratchpad. While solving a math problem, you hold intermediate results in your head ("carry the 3..."). Limited capacity (7±2 items), disappears quickly. For agents: the context window.
- **Long-term semantic memory** = Facts you've learned ("Paris is the capital of France"). Permanent, accessible by association. For agents: vector database of knowledge.
- **Episodic memory** = Specific events you remember ("Last Tuesday's meeting where John proposed the redesign"). Contextual, time-stamped, retrievable by similarity. For agents: logged interactions and outcomes.
- **Procedural memory** = Skills you've internalized ("how to ride a bike"). You don't think about it, you just do it. For agents: fine-tuned behaviors, cached tool sequences.
- **Forgetting** = Your brain naturally discards irrelevant information. Without forgetting, every detail would be equally prominent and finding relevant memories would be impossible. For agents: memory decay, importance scoring, TTL (Time To Live).

---

## Deep Dive

### Memory Architecture

```yaml
Memory_Types:
  short_term_memory:
    what: "Current conversation context (in the LLM context window)"
    stores:
      - "Current user goal"
      - "Conversation history (recent messages)"
      - "Results of recent tool calls"
      - "Current plan and progress"
    capacity: "Limited by context window (8K-200K tokens)"
    duration: "Current session only"
    implementation: "Part of the prompt (messages array)"
    challenge: "Overflow — must manage when conversation exceeds window"
    
  working_memory:
    what: "Structured state that persists across agent loop iterations"
    stores:
      - "Partial results being accumulated"
      - "Hypotheses being tested"
      - "Sub-goal completion status"
      - "Variables and intermediate computations"
    capacity: "Typically JSON object with defined schema"
    duration: "Current task execution"
    implementation: "Separate state object passed alongside messages"
    
  long_term_memory:
    what: "Persistent knowledge that survives across sessions"
    stores:
      - "User preferences (communication style, timezone, tools they prefer)"
      - "Learned facts (project context, team structure, domain knowledge)"
      - "Organizational knowledge (processes, policies, contacts)"
    capacity: "Unlimited (external storage)"
    duration: "Permanent (until explicitly updated or deleted)"
    implementation: "Vector database + metadata store"
    
  episodic_memory:
    what: "Specific past interactions and their outcomes"
    stores:
      - "Past conversations with timestamps"
      - "Actions taken and their results"
      - "Failures and what was learned"
      - "Successful patterns (what worked)"
    capacity: "Large (compressed summaries of past interactions)"
    duration: "Long-term with decay (older episodes compressed/summarized)"
    implementation: "Time-stamped entries in vector DB with importance scoring"
```

### Long-Term Memory Implementation

```python
# Agent memory system with vector store + metadata

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class MemoryEntry:
    """A single memory item."""
    id: str
    content: str                    # The memory content
    memory_type: str               # "preference", "fact", "episode", "procedure"
    importance: float              # 0-1 (how important to remember)
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    metadata: dict = field(default_factory=dict)
    # Metadata examples: {"user_id": "...", "topic": "...", "source": "..."}


class AgentMemory:
    """Production agent memory with retrieval, storage, and decay."""
    
    def __init__(self, vector_store, embedding_model, user_id: str):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.user_id = user_id
        
    async def remember(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        metadata: dict = None,
    ) -> str:
        """Store a new memory."""
        
        # Generate embedding for similarity search
        embedding = await self.embedding_model.embed(content)
        
        # Check for duplicate/similar memories
        similar = await self.vector_store.search(
            embedding=embedding,
            filter={"user_id": self.user_id, "type": memory_type},
            top_k=3,
            min_score=0.9,  # Very similar
        )
        
        if similar:
            # Update existing memory rather than creating duplicate
            existing = similar[0]
            return await self._update_memory(existing.id, content)
        
        # Create new memory
        entry = MemoryEntry(
            id=f"mem_{datetime.now().timestamp()}",
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            metadata={"user_id": self.user_id, **(metadata or {})},
        )
        
        await self.vector_store.upsert(
            id=entry.id,
            embedding=embedding,
            metadata={
                "content": content,
                "type": memory_type,
                "importance": importance,
                "user_id": self.user_id,
                "created_at": entry.created_at.isoformat(),
                **(metadata or {}),
            }
        )
        
        return entry.id
    
    async def recall(
        self,
        query: str,
        memory_types: list[str] = None,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Retrieve relevant memories for current context."""
        
        embedding = await self.embedding_model.embed(query)
        
        filters = {"user_id": self.user_id}
        if memory_types:
            filters["type"] = {"$in": memory_types}
        if min_importance > 0:
            filters["importance"] = {"$gte": min_importance}
        
        results = await self.vector_store.search(
            embedding=embedding,
            filter=filters,
            top_k=top_k,
        )
        
        # Update access counts and timestamps
        memories = []
        for result in results:
            memory = MemoryEntry(
                id=result.id,
                content=result.metadata["content"],
                memory_type=result.metadata["type"],
                importance=result.metadata["importance"],
                created_at=datetime.fromisoformat(result.metadata["created_at"]),
                last_accessed=datetime.now(),
                access_count=result.metadata.get("access_count", 0) + 1,
                metadata=result.metadata,
            )
            memories.append(memory)
            
            # Update last_accessed
            await self.vector_store.update_metadata(result.id, {
                "last_accessed": datetime.now().isoformat(),
                "access_count": memory.access_count,
            })
        
        return memories
    
    async def forget(self, memory_id: str):
        """Explicitly forget a memory (user requested or outdated)."""
        await self.vector_store.delete(memory_id)
    
    async def decay(self, max_age_days: int = 90, min_importance: float = 0.3):
        """Remove old, low-importance memories that haven't been accessed."""
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)
        
        # Find old, unimportant, rarely accessed memories
        candidates = await self.vector_store.query(
            filter={
                "user_id": self.user_id,
                "importance": {"$lt": min_importance},
                "last_accessed": {"$lt": datetime.fromtimestamp(cutoff).isoformat()},
            }
        )
        
        for memory in candidates:
            await self.vector_store.delete(memory.id)
        
        return len(candidates)


class MemoryManager:
    """Manages memory injection into agent prompts."""
    
    def __init__(self, memory: AgentMemory):
        self.memory = memory
        
    async def build_memory_context(
        self,
        current_query: str,
        max_tokens: int = 2000,
    ) -> str:
        """Build memory context to inject into agent's system prompt."""
        
        # Retrieve relevant memories
        memories = await self.memory.recall(
            query=current_query,
            top_k=10,
        )
        
        if not memories:
            return ""
        
        # Format memories for injection
        memory_text = "## Relevant Memory\n"
        memory_text += "You have the following memories about this user/context:\n\n"
        
        tokens_used = 0
        for mem in memories:
            entry = f"- [{mem.memory_type}] {mem.content}\n"
            entry_tokens = len(entry) // 4
            
            if tokens_used + entry_tokens > max_tokens:
                break
                
            memory_text += entry
            tokens_used += entry_tokens
        
        memory_text += "\nUse these memories to personalize your response.\n"
        return memory_text
```

### Memory Extraction (What to Remember)

```yaml
Memory_Extraction:
  what: "Deciding which information from an interaction to persist to long-term memory"
  
  automatic_extraction:
    approach: "After each interaction, use LLM to identify memorable information"
    prompt: |
      Review this conversation and extract information worth remembering:
      
      {conversation}
      
      Extract:
      1. User preferences (how they like things done, communication style)
      2. Facts about the user (role, company, projects, constraints)
      3. Decisions made (and reasoning behind them)
      4. Outcomes (what worked, what didn't)
      
      Return as JSON array of memory entries with content, type, and importance (0-1).
      Only extract genuinely useful information. Do NOT remember trivial small talk.
      
  importance_scoring:
    high_importance_0_8_plus:
      - "Explicit user preferences ('I always want responses in bullet points')"
      - "Critical constraints ('Never contact clients on weekends')"
      - "Project context ('We're migrating from AWS to GCP by Q3')"
      - "Corrections ('Actually, our API uses v3, not v2')"
      
    medium_importance_0_4_to_0_7:
      - "Task patterns ('User usually asks for code reviews on Tuesdays')"
      - "Domain knowledge ('Their stack is Python/FastAPI/PostgreSQL')"
      - "Team context ('Sarah handles frontend, Mike handles DevOps')"
      
    low_importance_0_1_to_0_3:
      - "One-time requests ('Help me format this CSV')"
      - "Transient information ('Meeting at 3pm today')"
      - "General observations (may not be useful again)"
      
  what_NOT_to_remember:
    - "Trivial greetings and small talk"
    - "Generic information the LLM already knows"
    - "PII that shouldn't be persisted (unless user consents)"
    - "Temporary state that won't be relevant next session"
```

### Episodic Memory for Learning

```yaml
Episodic_Memory:
  what: "Recording specific events/interactions and their outcomes for future reference"
  
  episode_structure:
    timestamp: "When did this happen?"
    context: "What was the user trying to do?"
    actions_taken: "What did the agent do?"
    outcome: "Did it succeed? What was the result?"
    lessons: "What should be done differently next time?"
    
  use_cases:
    failure_learning:
      scenario: "Agent tried approach X, it failed because of Y"
      memory: "When dealing with [situation], avoid [approach X] because [Y]. Instead, try [alternative]."
      future_use: "Next time similar situation arises, recall this lesson"
      
    success_patterns:
      scenario: "Agent used approach Z for user's data analysis, user was very satisfied"
      memory: "For [user]'s data analysis requests, use [approach Z] — they prefer [format/style]"
      future_use: "Replicate successful patterns"
      
    preference_refinement:
      scenario: "User corrected the agent's output format three times this week"
      memory: "User strongly prefers [specific format]. Importance: 0.95"
      future_use: "Apply preference proactively without being asked"
      
  retrieval_strategy:
    by_similarity: "When facing a similar task, recall relevant episodes"
    by_recency: "Recent episodes weighted higher (preferences may have changed)"
    by_importance: "Critical lessons (failures, corrections) always surface"
    combined: "score = similarity × 0.4 + recency × 0.3 + importance × 0.3"
```

### Memory at Scale

```yaml
Scaling_Considerations:
  per_user_memory:
    challenge: "Each user has their own memory store"
    solution: "Partition vector DB by user_id (namespace per user)"
    sizing: "Average user: 100-500 memories. Power user: 1000-5000 memories."
    
  organizational_memory:
    what: "Shared knowledge across all agents in the organization"
    examples:
      - "Company policies and procedures"
      - "Product documentation"
      - "Common troubleshooting patterns"
      - "Team structure and responsibilities"
    implementation: "Shared vector namespace accessible to all agents (read-only)"
    
  memory_consistency:
    problem: "User tells one agent their preference, other agents don't know"
    solution: "Centralized memory store (all agents read/write same user memory)"
    conflict: "Two agents write conflicting memories (handled via timestamp + importance)"
    
  privacy_and_compliance:
    gdpr_right_to_forget: "User requests deletion → delete all their memories"
    data_minimization: "Only remember what's necessary for service"
    consent: "User opts in to memory persistence"
    encryption: "Memories encrypted at rest"
    access_control: "Only agents serving this user can access their memories"
    
  performance:
    retrieval_latency: "Target < 50ms for memory recall"
    storage: "Vector DB with metadata filtering (Pinecone, Weaviate, Qdrant)"
    embedding: "Fast embedding model for real-time memory operations"
    caching: "LRU (Least Recently Used) cache for frequently accessed memories"
```

---

## How It Works in Practice

### Production Memory System

```yaml
Production_Architecture:
  components:
    memory_store: "Qdrant/Pinecone vector DB (per-user namespaces)"
    embedding_model: "text-embedding-3-small (fast, cheap, good enough for memory)"
    extraction_model: "Claude 4 Haiku / GPT-5-mini (cheap, runs after each session)"
    cache: "Redis (LRU cache for hot memories)"
    
  lifecycle:
    during_session:
      - "At session start: recall relevant memories for context (top 5-10)"
      - "Inject memories into system prompt"
      - "During session: working memory tracks task state"
      
    after_session:
      - "Memory extraction LLM identifies memorable information"
      - "New memories stored with embedding + metadata"
      - "Duplicate detection prevents redundant entries"
      - "Importance scoring determines retention priority"
      
    maintenance:
      - "Weekly: decay low-importance, old, unaccessed memories"
      - "Monthly: consolidate similar memories (merge related facts)"
      - "On user request: delete specific memories or all memories"
      
  monitoring:
    metrics:
      - "Memory retrieval latency (P50, P95)"
      - "Memories per user (growth rate)"
      - "Retrieval relevance (do recalled memories get used?)"
      - "Memory extraction quality (are we extracting useful info?)"
```

---

## Interview Tip

> When asked about agent memory: "I implement a three-tier memory system: (1) Short-term — the context window itself. Manages current conversation, recent tool results, and task state. I implement sliding window with summarization to handle overflow (summarize old messages, keep recent in full). (2) Long-term — vector database (Qdrant/Pinecone) storing persistent knowledge per user: preferences, facts, project context. Retrieved by embedding similarity at session start and during conversation when context shifts. Key challenge: what to remember — I use an extraction LLM after each session that identifies preferences, decisions, corrections (high importance) vs transient information (don't store). (3) Episodic — specific past interactions with outcomes. Critical for learning: 'Last time this approach failed because X — try Y instead.' Retrieved when facing similar situations. Operational considerations: memory decay (old, unaccessed, low-importance memories are pruned quarterly), duplicate detection (don't store the same fact 10 times), privacy compliance (GDPR right-to-forget = delete all user memories on request), and consistency (all agents for a user share the same memory store). The metric I track: memory utility rate — what % of recalled memories actually influence the response? If low, my retrieval or extraction is broken."

---

## Common Mistakes

1. **Remembering everything** — Storing every sentence from every conversation. Memory becomes a landfill — too noisy to find relevant information. Retrieval returns irrelevant memories that confuse the agent. Solution: selective memory — extract only genuinely useful information (preferences, corrections, decisions, lessons). Most conversation is transient and shouldn't be stored.

2. **No memory decay** — Memories accumulate forever. A preference from a year ago ("I use Python 2") conflicts with recent behavior (user switched to Python 3 months ago). Solution: implement decay — memories lose importance over time if not reinforced. Unaccessed memories are pruned. Recent memories weighted higher.

3. **Injecting too much memory into context** — Recalling 20 memories (5000 tokens) for a simple question. Wastes context window and may confuse the model. Solution: retrieve top 3-5 memories maximum, budget memory to 1000-2000 tokens, only inject when relevant to current query.

4. **No deduplication** — User mentions their timezone 50 times across sessions. You now have 50 memories saying "User is in PST." Solution: before storing, check for similar existing memories (>0.9 similarity). Update existing rather than creating duplicates.

5. **Memory without privacy controls** — Storing sensitive information (health data, financial details, personal relationships) without user consent or ability to delete. Violation of GDPR and user trust. Solution: explicit consent for memory persistence, easy deletion ("forget everything about me"), encryption at rest, access controls.

---

## Key Takeaways

- Three memory tiers: short-term (context window), long-term (vector DB), episodic (past events)
- Selective memory: only store genuinely useful information (preferences, facts, lessons, corrections)
- Memory extraction: use LLM after each session to identify memorable information
- Importance scoring: preferences and corrections (0.8+), domain facts (0.5), transient info (don't store)
- Memory decay: prune old, unaccessed, low-importance memories (prevents noise accumulation)
- Deduplication: check similarity before storing, update existing memories instead of duplicating
- Retrieval: embed query → vector search → top 3-5 relevant memories → inject into context
- Privacy: user consent, right to delete, encryption, access controls (GDPR compliance)
- Episodic learning: record failures and successes, recall when facing similar situations
- Scale: partition by user (namespace), shared org knowledge read-only, Redis cache for hot memories
