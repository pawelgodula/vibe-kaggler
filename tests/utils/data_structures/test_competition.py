import pytest
from typing import List
from utils.data_structures.competition import Post, Competition

def test_post_instantiation():
    """Test that a Post object can be instantiated correctly."""
    url = "http://example.com/post1"
    topic = "Test Topic"
    post_text = "This is the post text."
    comments = "These are comments."
    private_lb_place = 10

    post = Post(
        url=url,
        topic=topic,
        post_text=post_text,
        comments=comments,
        private_lb_place=private_lb_place,
    )

    assert post.url == url
    assert post.topic == topic
    assert post.post_text == post_text
    assert post.comments == comments
    assert post.private_lb_place == private_lb_place

def test_post_instantiation_with_defaults():
    """Test that a Post object can be instantiated with default None values."""
    post = Post()
    assert post.url is None
    assert post.topic is None
    assert post.post_text is None
    assert post.comments is None
    assert post.private_lb_place is None


def test_competition_instantiation_required_only():
    """Test instantiating Competition with only required fields."""
    data_modality = "tabular"
    problem_type = "classification"
    url = "http://example.com/competition1"
    target = "is_fraud"

    comp = Competition(
        data_modality=data_modality,
        problem_type=problem_type,
        url=url,
        target=target,
    )

    assert comp.data_modality == data_modality
    assert comp.problem_type == problem_type
    assert comp.url == url
    assert comp.target == target
    assert comp.host_industry is None
    assert comp.other_details is None
    assert comp.eval_metric is None
    assert comp.data_origin is None
    assert comp.overview is None
    assert comp.raw_description is None
    assert comp.data_description is None
    assert comp.operator_comments is None
    assert comp.forum_posts is None

def test_competition_instantiation_all_fields():
    """Test instantiating Competition with all fields."""
    data_modality = "image"
    problem_type = "object detection"
    url = "http://example.com/competition2"
    target = "bounding_box"
    host_industry = "automotive"
    other_details = "Includes synthetic data"
    eval_metric = "mAP"
    data_origin = "mixed"
    overview = "Detect cars in images."
    raw_description = "Detailed description text."
    data_description = "Image dataset description."
    operator_comments = "Initial setup comments."
    forum_posts_data = [
        Post(url="http://example.com/post2", topic="Data Augmentation"),
        Post(url="http://example.com/post3", topic="Model Choice")
    ]

    comp = Competition(
        data_modality=data_modality,
        problem_type=problem_type,
        url=url,
        target=target,
        host_industry=host_industry,
        other_details=other_details,
        eval_metric=eval_metric,
        data_origin=data_origin,
        overview=overview,
        raw_description=raw_description,
        data_description=data_description,
        operator_comments=operator_comments,
        forum_posts=forum_posts_data,
    )

    assert comp.data_modality == data_modality
    assert comp.problem_type == problem_type
    assert comp.url == url
    assert comp.target == target
    assert comp.host_industry == host_industry
    assert comp.other_details == other_details
    assert comp.eval_metric == eval_metric
    assert comp.data_origin == data_origin
    assert comp.overview == overview
    assert comp.raw_description == raw_description
    assert comp.data_description == data_description
    assert comp.operator_comments == operator_comments
    assert comp.forum_posts == forum_posts_data
    assert len(comp.forum_posts) == 2
    assert comp.forum_posts[0].url == "http://example.com/post2"
    assert comp.forum_posts[1].topic == "Model Choice"

@pytest.mark.parametrize(
    "field",
    ["data_modality", "problem_type", "url", "target"]
)
def test_competition_instantiation_missing_required(field):
    """Test that Competition raises TypeError if required fields are missing."""
    required_args = {
        "data_modality": "text",
        "problem_type": "NER",
        "url": "http://example.com/comp3",
        "target": "entities"
    }
    # Remove one required field
    del required_args[field]
    with pytest.raises(TypeError):
        Competition(**required_args) 