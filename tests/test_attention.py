from dataclasses import replace

import pytest

from motion_prior_tracker import TrackObservation


def test_attention_uses_visible_object_and_requests_search_when_lost():
    from examples.embodied_attention import attention_hint
    seen = TrackObservation(0, (.75, .25, .1, .1), (150, 25), (2, 0),
                            .9, 'matched', 0, 7, (200, 100))
    hint = attention_hint(seen)
    assert hint['mode'] == 'observe'
    assert hint['image_error'] == pytest.approx([.5, -.5])
    predicted = replace(seen, mode='predicted', score=0., lost_frames=2)
    assert attention_hint(predicted)['mode'] == 'hold'
    assert attention_hint(predicted)['image_error'] is None
    lost = replace(predicted, lost_frames=4)
    assert attention_hint(lost)['mode'] == 'search'


def test_attention_does_not_treat_weak_similarity_as_a_reliable_control_cue():
    from examples.embodied_attention import attention_hint
    weak = TrackObservation(1, (.5, .5, .2, .2), (50, 50), (0, 0),
                            .2, 'matched', 0, 2, (100, 100))
    assert attention_hint(weak)['mode'] == 'hold'
