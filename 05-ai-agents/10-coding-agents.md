# Coding Agents

## The Problem / Why This Matters

Coding agents — AI systems that autonomously write, debug, test, and refactor code — are the fastest-growing category of AI agents in 2026. GitHub Copilot's agent mode, Cursor's Composer, Claude Code, Devin, and dozens of others are fundamentally changing how software is built. But most engineers use these tools at 10% of their potential because they don't understand the underlying architecture: how does a coding agent navigate a 100,000-file codebase? How does it decide which files to read? How does it know when its generated code is correct? How does it handle ambiguous requirements? Understanding the ML engineering behind coding agents lets you: (1) use them more effectively (better prompts, better workflows), (2) build custom coding agents for your specific codebase, (3) evaluate and compare different tools objectively, and (4) understand their limitations (when to trust them vs when to verify). Coding agents represent the intersection of LLM capabilities (code generation), tool use (file operations, terminal commands), planning (multi-step code changes), and evaluation (testing, type checking) — making them the most complex agent type in production today.

---

## The Analogy

Think of a coding agent like a new senior engineer joining your team:

- **Day 1**: They know programming in general but nothing about YOUR codebase. First task: understand the project structure (read README, look at directory layout, identify patterns).
- **Exploration**: Before making changes, they read relevant files — following imports, checking tests, understanding conventions. They don't just blindly edit files.
- **Planning**: For complex tasks, they make a plan first (modify these 3 files, update these tests, check these edge cases).
- **Iteration**: They write code, run tests, see failures, fix issues, run again. The cycle continues until tests pass.
- **Code review**: They check their own work (type checking, linting, tests) before submitting. They might revise if something looks wrong.
- **Asking for help**: When requirements are ambiguous or they're unsure about architectural decisions, good engineers ask. Good coding agents do too.

The difference: a human does this over hours/days. A coding agent does it in minutes — but with less judgment about when to break conventions or when requirements are subtly wrong.

---

## Deep Dive

### Coding Agent Architecture

```yaml
Coding_Agent_Architecture:
  core_components:
    codebase_understanding:
      what: "How the agent understands and navigates the codebase"
      techniques:
        file_tree_analysis: "Read directory structure, identify key files"
        semantic_search: "Embed codebase, retrieve relevant files by query"
        symbol_resolution: "Follow imports, find definitions, trace call chains"
        ast_parsing: "Parse code into Abstract Syntax Trees for structural understanding"
        lsp_integration: "Language Server Protocol for go-to-definition, find-references"
      context_management:
        challenge: "100K+ file codebase doesn't fit in context window"
        solution: "Selective retrieval — only read files relevant to current task"
        strategies:
          - "Start with README, package.json, directory structure"
          - "Follow imports from the file being modified"
          - "Search for similar patterns in codebase"
          - "Use git blame to find related recent changes"
          
    code_generation:
      what: "Producing correct, idiomatic code"
      techniques:
        in_context_learning: "Show agent existing code patterns → it follows conventions"
        structured_output: "Generate code in specific formats (function, class, test)"
        multi_file_editing: "Coordinate changes across multiple files"
        diff_generation: "Generate minimal diffs rather than full file rewrites"
      quality_factors:
        - "Follows existing code style (indentation, naming, patterns)"
        - "Uses existing utilities (doesn't reinvent what's already available)"
        - "Handles edge cases (null checks, error handling)"
        - "Includes appropriate types/type annotations"
        
    verification:
      what: "Confirming generated code is correct"
      methods:
        static_analysis: "Type checking (TypeScript tsc, mypy), linting (ESLint, ruff)"
        test_execution: "Run existing tests to catch regressions"
        test_generation: "Write new tests for new code"
        build_verification: "Ensure project still compiles"
        runtime_testing: "Execute code and verify output"
      iteration_loop:
        1: "Generate code"
        2: "Run verification (type check + tests)"
        3: "If failures → analyze errors"
        4: "Fix issues (targeted edits, not full rewrite)"
        5: "Run verification again"
        6: "Repeat until all checks pass (max 5 iterations)"
        
    planning:
      what: "Decomposing complex tasks into steps"
      approach:
        simple_tasks: "Direct execution (no explicit planning needed)"
        medium_tasks: "Brief mental plan → execute sequentially"
        complex_tasks: "Explicit multi-step plan → execute → verify each step"
      plan_structure:
        - "Identify affected files"
        - "Determine order of changes"
        - "Identify dependencies between changes"
        - "Plan verification strategy"
```

### How GitHub Copilot Agent Mode Works

```yaml
Copilot_Agent_Mode:
  what: "Multi-step autonomous coding within VS Code"
  released: "2025, iteratively improved through 2026"
  
  architecture:
    model: "GPT-4.1 / Claude 4 Sonnet (user-selectable)"
    context:
      - "Open files in editor"
      - "Workspace file tree"
      - "Terminal output"
      - "Diagnostic errors (red squiggles)"
      - "Selected code / current file"
    
    tools_available:
      file_operations:
        - "Read file (with line ranges)"
        - "Write file (create new)"
        - "Edit file (search-and-replace)"
        - "List directory"
        - "Search files by name (glob)"
        - "Search file contents (grep)"
      
      code_intelligence:
        - "Go to definition"
        - "Find references"
        - "Get diagnostics (type errors, lint errors)"
        - "Semantic search (workspace-wide)"
      
      execution:
        - "Run terminal command"
        - "Run tests"
        - "Build project"
      
    execution_flow:
      1: "User provides task in chat"
      2: "Agent analyzes workspace context"
      3: "Agent plans approach (may be implicit)"
      4: "Agent calls tools (read files, search, etc.)"
      5: "Agent generates code changes (edits)"
      6: "Agent runs verification (tests, type check)"
      7: "If errors: agent reads errors, fixes, re-runs"
      8: "Agent presents changes to user for review"
      
  capabilities_2026:
    - "Multi-file editing (coordinated changes)"
    - "Terminal command execution"
    - "Automatic test running and fixing"
    - "Error diagnosis from diagnostic output"
    - "Context from entire workspace (semantic search)"
    - "Custom instructions (.github/copilot-instructions.md)"
    - "MCP tool integration (external tools)"
```

### Cursor Agent Architecture

```yaml
Cursor_Composer:
  what: "Agent-mode code editing in Cursor IDE"
  
  differentiators:
    codebase_indexing:
      what: "Pre-indexes entire codebase for fast retrieval"
      how: "Embeds all files → vector index → semantic search"
      benefit: "Agent finds relevant code instantly (no manual file search)"
      
    apply_model:
      what: "Specialized model for applying diffs to files"
      how: "Separate small model that applies generated changes cleanly"
      benefit: "Faster edits, fewer merge conflicts, cleaner diffs"
      
    multi_file_composer:
      what: "Edit multiple files in a single operation with full context"
      how: "Agent sees all affected files, generates coordinated changes"
      benefit: "Atomic multi-file changes (imports, implementations, tests all updated)"
      
    context_retrieval:
      strategies:
        - "Semantic search (find conceptually similar code)"
        - "Recently edited files (likely relevant)"
        - "@-mentions (user explicitly includes files)"
        - "Import graph (follow dependencies)"
```

### Building Custom Coding Agents

```python
# Building a custom coding agent for a specific codebase

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodingAgentConfig:
    """Configuration for a custom coding agent."""
    
    # Codebase context
    workspace_root: Path
    file_patterns: list[str]  # ["*.py", "*.ts", "*.yaml"]
    ignore_patterns: list[str]  # ["node_modules/**", "*.pyc"]
    
    # Style and conventions
    style_guide_files: list[str]  # ["CONVENTIONS.md", ".eslintrc"]
    example_files: list[str]  # Representative files showing desired style
    
    # Verification
    test_command: str  # "pytest", "npm test"
    lint_command: str  # "ruff check .", "eslint ."
    type_check_command: str  # "mypy .", "tsc --noEmit"
    build_command: str  # "npm run build"
    
    # Limits
    max_iterations: int = 5
    max_files_to_read: int = 20
    max_files_to_edit: int = 10


class CustomCodingAgent:
    """A coding agent specialized for a specific codebase."""
    
    def __init__(self, config: CodingAgentConfig, llm_client):
        self.config = config
        self.llm = llm_client
        self.codebase_index = None
        
    async def initialize(self):
        """Index the codebase for fast retrieval."""
        self.codebase_index = await self._build_index()
        
    async def execute_task(self, task: str) -> dict:
        """Execute a coding task end-to-end."""
        
        # Phase 1: Understand
        context = await self._gather_context(task)
        
        # Phase 2: Plan
        plan = await self._create_plan(task, context)
        
        # Phase 3: Execute + Verify loop
        for iteration in range(self.config.max_iterations):
            # Generate code changes
            changes = await self._generate_changes(task, context, plan)
            
            # Apply changes
            await self._apply_changes(changes)
            
            # Verify
            verification = await self._verify()
            
            if verification.all_passed:
                return {"status": "success", "changes": changes, "iterations": iteration + 1}
            
            # Fix issues
            context = await self._add_error_context(context, verification.errors)
            plan = await self._revise_plan(plan, verification.errors)
        
        return {"status": "max_iterations_exceeded", "last_errors": verification.errors}
    
    async def _gather_context(self, task: str) -> dict:
        """Retrieve relevant files and context for the task."""
        
        # Semantic search for relevant files
        relevant_files = await self.codebase_index.search(task, top_k=10)
        
        # Read file contents
        file_contents = {}
        for file_path in relevant_files[:self.config.max_files_to_read]:
            content = Path(file_path).read_text()
            file_contents[file_path] = content
        
        # Get project structure
        structure = await self._get_directory_structure()
        
        # Get conventions
        conventions = await self._read_conventions()
        
        return {
            "relevant_files": file_contents,
            "project_structure": structure,
            "conventions": conventions,
            "recent_changes": await self._get_recent_git_changes(),
        }
    
    async def _verify(self) -> "VerificationResult":
        """Run all verification checks."""
        
        results = []
        
        # Type checking
        type_result = await self._run_command(self.config.type_check_command)
        results.append(("type_check", type_result))
        
        # Linting
        lint_result = await self._run_command(self.config.lint_command)
        results.append(("lint", lint_result))
        
        # Tests
        test_result = await self._run_command(self.config.test_command)
        results.append(("tests", test_result))
        
        # Build
        build_result = await self._run_command(self.config.build_command)
        results.append(("build", build_result))
        
        all_passed = all(r.returncode == 0 for _, r in results)
        errors = [f"{name}: {r.stderr}" for name, r in results if r.returncode != 0]
        
        return VerificationResult(all_passed=all_passed, errors=errors)
```

### Code Generation Techniques

```yaml
Code_Generation_Techniques:
  fill_in_the_middle:
    what: "Model generates code between a prefix and suffix"
    use_case: "Autocomplete (inline suggestions)"
    models: "Specialized: Codestral, StarCoder2, DeepSeek-Coder"
    
  instruction_following:
    what: "Natural language instruction → code output"
    use_case: "Agent-mode generation (describe what you want)"
    models: "General: Claude 4, GPT-5, Gemini 2.5"
    
  edit_mode:
    what: "Given existing code + instruction → modified code"
    use_case: "Refactoring, bug fixing, feature addition"
    approach: "Show original code in context, instruct to modify"
    
  test_generation:
    what: "Given implementation → generate comprehensive tests"
    strategies:
      - "Happy path tests (normal usage)"
      - "Edge case tests (boundaries, empty inputs, nulls)"
      - "Error tests (invalid inputs, exceptions)"
      - "Integration tests (component interactions)"
    quality: "Generated tests should catch real bugs, not just increase coverage"
    
  debugging:
    what: "Given code + error → diagnose and fix"
    flow:
      1: "Agent sees error message and stack trace"
      2: "Agent reads relevant source code"
      3: "Agent reasons about root cause"
      4: "Agent generates fix"
      5: "Agent verifies fix resolves error"
    tools: "Debugger integration, logging, tracing"
```

### Evaluation: SWE-bench

```yaml
SWE_bench:
  what: "Standard benchmark for coding agents"
  description: "Real GitHub issues → agent must produce a patch that resolves the issue"
  
  dataset:
    source: "Real issues from popular Python repos (Django, Flask, sympy, etc.)"
    size: "2,294 issues (SWE-bench full), 300 (SWE-bench Lite)"
    format:
      input: "Issue description + repository state at time of issue"
      expected: "Patch that makes failing tests pass"
      
  scoring:
    metric: "% of issues resolved (patch passes the test suite)"
    levels:
      random_baseline: "~0% (random patches don't fix bugs)"
      early_2024: "~5-10% (GPT-4 + basic agent)"
      mid_2025: "~30-40% (Claude 3.5 + sophisticated agents)"
      current_2026: "~50-60% (Claude 4 / GPT-5 + advanced agent tooling)"
      
  what_it_measures:
    - "Code understanding (read and comprehend large codebases)"
    - "Bug localization (find relevant files among thousands)"
    - "Fix generation (produce correct patches)"
    - "Test awareness (understand what tests verify)"
    
  limitations:
    - "Only Python repositories"
    - "Only bug fixes (not feature development)"
    - "Ground truth is specific patches (many valid alternatives exist)"
    - "Doesn't test interactive workflows (agent can't ask questions)"
```

---

## How It Works in Practice

### Using Coding Agents Effectively

```yaml
Best_Practices:
  effective_prompting:
    do:
      - "Provide context: 'In the user auth module...'"
      - "Be specific: 'Add input validation for the email field in signup form'"
      - "Reference existing patterns: 'Follow the same pattern as UserService'"
      - "Specify constraints: 'Must maintain backward compatibility'"
      - "Include acceptance criteria: 'Should handle null input, empty string, invalid format'"
    dont:
      - "Vague: 'Make the code better'"
      - "Too broad: 'Refactor the entire authentication system'"
      - "No context: 'Fix the bug' (which bug? where?)"
      
  workflow_integration:
    simple_tasks: "Direct agent mode — describe task, let agent work"
    medium_tasks: "Agent generates plan → you approve → agent executes"
    complex_tasks: "Break into smaller tasks yourself → agent handles each one"
    
  when_to_trust:
    high_trust:
      - "Well-defined tasks with clear acceptance criteria"
      - "Boilerplate code (CRUD endpoints, form components)"
      - "Test generation for existing code"
      - "Refactoring with tests already in place"
    low_trust:
      - "Security-sensitive code (auth, crypto, input validation)"
      - "Performance-critical paths (always benchmark)"
      - "Architecture decisions (agent follows patterns, doesn't question them)"
      - "Business logic subtleties (domain expertise needed)"
```

---

## Interview Tip

> When asked about coding agents: "I understand coding agents at the architecture level. The core loop is: (1) Context retrieval — semantic search over the codebase to find relevant files (can't fit 100K files in context window, so selective retrieval is critical). (2) Planning — for complex tasks, decompose into steps with dependencies. (3) Code generation — generate edits (not full rewrites) that follow existing conventions. (4) Verification — run type checking, linting, and tests. If failures, feed errors back and iterate (up to 5 attempts). Key engineering challenges: context window management (how to give the agent enough context without overwhelming it), verification loops (agent must be able to interpret error messages and fix its own mistakes), and multi-file coordination (changes to one file often require changes to imports, tests, types in other files). For custom coding agents, I'd build: codebase indexing (embed all files for semantic search), convention detection (few-shot examples from the codebase), and tool integration (terminal access for running tests/builds). The evaluation standard is SWE-bench — current best agents resolve ~55% of real GitHub issues, up from ~5% in early 2024. The gap to 100% is mainly: complex reasoning about subtle bugs, understanding implicit requirements, and multi-step debugging."

---

## Common Mistakes

1. **Over-trusting generated code** — Agent produces code that looks correct, you accept without review. Later: security vulnerability, subtle logic bug, or performance issue. Solution: always review generated code like you'd review a junior engineer's PR (Pull Request). Pay special attention to security, error handling, and edge cases.

2. **Vague prompts for complex tasks** — Telling the agent "fix the authentication" when you mean "add rate limiting to the login endpoint." Agent rewrites the entire auth module. Solution: be specific about scope, provide acceptance criteria, reference relevant files.

3. **Not providing codebase context** — Agent generates code that doesn't follow your conventions, uses wrong patterns, or imports non-existent utilities. Solution: include relevant example files, reference existing patterns ("follow the same approach as UserService"), use custom instructions files (.github/copilot-instructions.md).

4. **Skipping the verification loop** — Agent generates code → you accept without running tests. Tests are broken, types don't check, linting fails. Solution: always require the agent to run verification (tests + type check + lint) before presenting results. If your agent tool doesn't do this automatically, ask explicitly.

5. **Using agents for architecture decisions** — Agent is great at implementing patterns but poor at choosing them. You ask "how should I structure this?" and get a plausible but wrong answer. Solution: make architecture decisions yourself (or discuss with humans), then use agents for implementation. Agents follow patterns; they don't question whether the pattern is appropriate.

---

## Key Takeaways

- Coding agents: LLM + codebase navigation + code generation + verification loop
- Core challenge: selective context retrieval (100K files can't fit in context window)
- Verification loop: type check → lint → test → fix → repeat (up to 5 iterations)
- SWE-bench: current best ~55% resolution rate (up from 5% in early 2024)
- GitHub Copilot agent mode: multi-step editing with terminal access and diagnostics
- Custom agents need: codebase indexing, convention detection, tool integration
- Trust gradient: high for boilerplate/tests, low for security/architecture
- Effective use: specific prompts, provide context, reference existing patterns
- Always verify: treat agent output like a junior engineer's PR — review carefully
- Agents implement patterns, they don't choose patterns — architecture decisions are yours
