# Competition dataclass

"""
Extended Description:
A dataclass representing metadata for a Kaggle competition, including a textual description, data modality, problem type, host industry, and any other optional details.
"""

from typing import Optional, Annotated, List

class Post:
    """Class representing a forum post."""
    def __init__(
        self,
        url: Annotated[Optional[str], "Post URL"] = None,
        topic: Annotated[Optional[str], "Post topic"] = None,
        post_text: Annotated[Optional[str], "Text of the post"] = None,
        comments: Annotated[Optional[str], "Comments on the post"] = None,
        private_lb_place: Annotated[Optional[int], "Private leaderboard place"] = None,
    ) -> None:
        self.url = url
        self.topic = topic
        self.post_text = post_text
        self.comments = comments
        self.private_lb_place = private_lb_place

class Competition:
    """Dataclass for competition metadata.

    Extended Description:
    Holds key attributes describing a Kaggle competition to help track and configure experiments.
    """
    def __init__(
        self,
        data_modality: Annotated[str, "Data modality: tabular, image, text, audio, etc."],
        problem_type: Annotated[str, "Problem type: classification, regression, etc."],
        url: Annotated[str, "Competition URL"],
        target: Annotated[str, "Target variable: what we are predicting"],
        host_industry: Annotated[Optional[str], "Industry hosting the competition, e.g. finance, healthcare"] = None,
        other_details: Annotated[Optional[str], "Any other optional details"] = None,
        eval_metric: Annotated[Optional[str], "Evaluation metric"] = None,
        data_origin: Annotated[Optional[str], "Data origin: real vs synthetic"] = None,
        overview: Annotated[Optional[str], "Overview of the competition"] = None,
        raw_description: Annotated[Optional[str], "Raw description text"] = None,
        data_description: Annotated[Optional[str], "Data description text"] = None,
        operator_comments: Annotated[Optional[str], "Operator comments"] = None,
        forum_posts: Annotated[Optional[List[Post]], "List of forum posts"] = None,
    ) -> None:
        self.data_modality = data_modality
        self.url = url
        self.target = target
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.data_origin = data_origin
        self.overview = overview
        self.raw_description = raw_description
        self.data_description = data_description
        self.operator_comments = operator_comments
        self.forum_posts = forum_posts
        self.host_industry = host_industry
        self.other_details = other_details 