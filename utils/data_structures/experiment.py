# Experiment dataclass

"""
Extended Description:
A dataclass representing a single experiment run on a Kaggle competition, capturing metadata like script filename, rationale ("why"), goals ("what"), implementation details ("how"), and evaluation scores.
"""

from typing import Optional, Annotated

class Experiment:
    """Dataclass for experiment metadata.

    Extended Description:
    Contains metadata for a competition experiment including script file name, rationale ('why'), goals ('what'), implementation plan ('how'), cross-validation score, and leaderboard score.
    """
    def __init__(
        self,
        script_file_name: Annotated[str, "Script file name"],
        why: Annotated[str, "Rationale ('why')"],
        what: Annotated[str, "Goals ('what')"],
        how: Annotated[str, "Implementation plan ('how')"],
        cv_score: Annotated[Optional[float], "Cross-validation score"] = None,
        lb_score: Annotated[Optional[float], "Leaderboard score"] = None,
        status: Annotated[str, "Experiment status: planned, in_progress, done, abandoned"] = "planned",
    ) -> None:
        self.script_file_name = script_file_name
        self.why = why
        self.what = what
        self.how = how
        self.cv_score = cv_score
        self.lb_score = lb_score
        self.status = status 