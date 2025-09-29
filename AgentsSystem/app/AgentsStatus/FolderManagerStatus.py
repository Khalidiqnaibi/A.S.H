from typing import TypedDict, List, Optional

class FolderManagerStatus(TypedDict, total=False):
    base_path: str
    structure_planner_response: str
    folders_creator_responsoe: str
    files_creator_response: str
