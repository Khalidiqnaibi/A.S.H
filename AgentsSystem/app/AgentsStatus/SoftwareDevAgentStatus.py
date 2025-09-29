from typing import TypedDict, List, Optional

# --- LangGraph-compatible status for a software development agent ---
class SoftwareDevAgentStatus(TypedDict, total=False):
    project_name: str                   # Name of the current project
    current_task: str                   # Current task the agent is working on
    code_snippet: str                   # Latest code generated/modified
    plan: str                           # Development plan or roadmap
    test_results: str                   # Output from running tests
    errors: List[str]                   # List of current errors/issues
    last_action: str                    # Last action taken by the agent
    context: str                        # Contextual info for the task
    review_notes: Optional[str]         # Optional feedback from human review
    status: str                         # Overall status ("in_progress", "testing", "completed")
