import numpy as np
import pytest

from rice_ml.unsupervised_learning.community_detection import (
    _validate_adjacency,
    label_propagation_communities,
)


def test_validate_adjacency_requires_square_matrix():
    with pytest.raises(ValueError, match="square"):
        _validate_adjacency(np.ones((2, 3)))


def test_validate_adjacency_rejects_negative_weights():
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_adjacency(np.array([[0.0, -1.0], [1.0, 0.0]]))


def test_label_propagation_discovers_two_clusters():
    """
    Two disconnected dense clusters should be assigned to two distinct communities.
    """
    cluster_a = np.ones((3, 3)) - np.eye(3)
    cluster_b = np.ones((3, 3)) - np.eye(3)

    top = np.hstack([cluster_a, np.zeros((3, 3))])
    bottom = np.hstack([np.zeros((3, 3)), cluster_b])
    adjacency = np.vstack([top, bottom])

    labels = label_propagation_communities(adjacency, seed=0, shuffle=True)

    a_labels = set(labels[:3])
    b_labels = set(labels[3:])

    # Each cluster should have a single label, and the labels should differ
    assert len(a_labels) == len(b_labels) == 1
    assert a_labels != b_labels


def test_label_propagation_is_reproducible_with_fixed_seed():
    """
    With a fixed random seed, label propagation should be reproducible even when
    random tie-breaking is enabled.
    """
    adjacency = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=float,
    )

    labels_run_1 = label_propagation_communities(adjacency, seed=7, shuffle=True)
    labels_run_2 = label_propagation_communities(adjacency, seed=7, shuffle=True)

    assert np.array_equal(labels_run_1, labels_run_2)


def test_isolated_nodes_keep_unique_labels_after_relabeling():
    """
    Nodes with no edges should remain in distinct communities after relabeling.
    """
    adjacency = np.zeros((3, 3), dtype=float)

    labels = label_propagation_communities(adjacency, shuffle=False)

    # No edges -> no propagation; relabeling should return 0, 1, 2
    assert np.array_equal(labels, np.arange(3))

def test_output_shape_and_type():
    """
    The output should be a 1D array of integer labels with the same length
    as the number of nodes.
    """
    adjacency = np.eye(4)

    labels = label_propagation_communities(adjacency)

    assert labels.shape == (4,)
    assert issubclass(labels.dtype.type, np.integer)


def test_fully_connected_graph_forms_single_community():
    """
    In a fully connected graph, all nodes should collapse into a single community.
    """
    adjacency = np.ones((5, 5)) - np.eye(5)

    labels = label_propagation_communities(adjacency, shuffle=False)

    assert len(set(labels)) == 1
