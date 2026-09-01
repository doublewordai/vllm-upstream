# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Delta-scoped reasoning-end detection for parser-engine adapters.

The scheduler calls ``is_reasoning_end_streaming`` once per structured-output
request per decode step. The ``ReasoningParser`` base implementation ignores
``delta_ids`` and rescans the whole sequence, which makes structured-output
decoding quadratic in the generated length. Parser-engine adapters must
override it with a delta-scoped check.

Contract: the method answers "did reasoning end *within this delta*". Callers
only ask while the request is still inside its reasoning span --
``StructuredOutputManager.should_advance`` short-circuits once
``reasoning_ended`` is latched -- so a transition can only occur in the delta.
This matches ``vllm.reasoning.basic_parsers``, which answers the same question
with ``end_token_id in delta_ids``.
"""

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.deepseek_v4 import DSML_THINK_END, DSML_THINK_START
from vllm.parser.engine import registered_adapters
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.registered_adapters import DeepSeekV4ParserReasoningAdapter
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

_THINK_START_ID = 50
_THINK_END_ID = 51
_TEXT_ID = 7


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            DSML_THINK_START: _THINK_START_ID,
            DSML_THINK_END: _THINK_END_ID,
        }
    )


@pytest.fixture
def parser(mock_tokenizer):
    return DeepSeekV4ParserReasoningAdapter(mock_tokenizer)


# (output_ids, delta_ids, expected). ``output_ids`` already includes the delta,
# matching how the scheduler calls this after appending the step's tokens.
TEST_CASES = [
    pytest.param([_THINK_START_ID], [_THINK_START_ID], False, id="open-think"),
    pytest.param([_THINK_START_ID, _TEXT_ID], [_TEXT_ID], False, id="mid-reasoning"),
    pytest.param(
        [_THINK_START_ID, _TEXT_ID, _THINK_END_ID],
        [_THINK_END_ID],
        True,
        id="ends-in-delta",
    ),
    pytest.param([_THINK_END_ID], [_THINK_END_ID], True, id="bare-end-in-delta"),
    pytest.param(
        [_THINK_START_ID, _TEXT_ID, _THINK_END_ID, _TEXT_ID],
        [_THINK_END_ID, _TEXT_ID],
        True,
        id="ends-mid-delta",
    ),
    pytest.param(
        [_THINK_END_ID, _THINK_START_ID],
        [_THINK_END_ID, _THINK_START_ID],
        False,
        id="reopened-in-delta",
    ),
]


@pytest.mark.parametrize("output_ids, delta_ids, expected", TEST_CASES)
def test_detects_transition_within_the_delta(parser, output_ids, delta_ids, expected):
    assert parser.is_reasoning_end_streaming(output_ids, delta_ids) is expected


@pytest.mark.parametrize("output_ids, delta_ids, expected", TEST_CASES)
def test_long_prefix_does_not_change_result(parser, output_ids, delta_ids, expected):
    """The answer must depend on the delta, not on how much came before.

    This is the regression case: the buggy implementation rescanned the whole
    sequence, so its cost grew with the prefix while the answer did not.
    """
    padded = [_THINK_START_ID] + [_TEXT_ID] * 10_000 + list(output_ids)
    assert parser.is_reasoning_end_streaming(padded, delta_ids) is expected


def test_large_delta_is_fully_covered(parser):
    """Speculative decoding can accept many tokens in a single step."""
    body = [_THINK_START_ID] + [_TEXT_ID] * 32 + [_THINK_END_ID] + [_TEXT_ID] * 8
    padded = [_THINK_START_ID] + [_TEXT_ID] * 10_000 + body
    # The marker sits deep inside the accepted window, not at either edge.
    assert parser.is_reasoning_end_streaming(padded, body) is True


def test_already_ended_is_not_re_asked(parser):
    """Reasoning that ended in an *earlier* step is outside the contract.

    ``should_advance`` latches ``reasoning_ended`` and stops calling, so a
    delta with no marker correctly reports False even though a full scan of
    the sequence would find the earlier end marker. ``basic_parsers`` behaves
    identically. This test pins the boundary so it is a deliberate choice.
    """
    ids = [_THINK_START_ID, _TEXT_ID, _THINK_END_ID, _TEXT_ID]
    assert parser.is_reasoning_end(ids) is True
    assert parser.is_reasoning_end_streaming(ids, [_TEXT_ID]) is False


def test_empty_delta_falls_back_to_full_scan(parser):
    """No tokens this step: defer to the full scan for prompt-derived state."""
    assert parser.is_reasoning_end_streaming([], []) is parser.is_reasoning_end([])
    ids = [_THINK_START_ID, _TEXT_ID, _THINK_END_ID, _TEXT_ID]
    assert parser.is_reasoning_end_streaming(ids, []) is True


def test_streaming_does_not_rescan_the_prefix(parser, monkeypatch):
    """With a delta supplied, the full-sequence scan must not be reached.

    Asserted structurally rather than by timing so the guard cannot flake.
    """
    calls = []
    original = ParserEngine.is_reasoning_end

    def _spy(self, input_ids):
        calls.append(len(input_ids))
        return original(self, input_ids)

    # Patch the base class so the engine still counts as "not overridden"
    # for the delta fast path's override guard.
    monkeypatch.setattr(ParserEngine, "is_reasoning_end", _spy)

    ids = [_THINK_START_ID] + [_TEXT_ID] * 10_000
    parser.is_reasoning_end_streaming(ids, [_TEXT_ID])
    assert calls == []

    # The empty-delta cold path may still fall back to the full scan.
    parser.is_reasoning_end_streaming(ids, [])
    assert calls == [len(ids)]


def test_engine_is_reasoning_end_override_is_respected(mock_tokenizer):
    """Engines with extra end conditions keep them under streaming.

    Six engines (qwen3, glm47_moe, kimi_k2, gemma4, inkling, mistral) override
    ``is_reasoning_end`` -- e.g. an unpaired tool-call marker ends reasoning.
    The delta fast path must not bypass those overrides, so an overriding
    engine falls back to its own full scan.
    """
    from vllm.parser.deepseek_v4 import DeepSeekV4Parser

    sentinel = []

    class _CustomEndParser(DeepSeekV4Parser):
        def is_reasoning_end(self, input_ids):
            sentinel.append(len(input_ids))
            return True

    from vllm.parser.engine.adapters import make_adapters

    reasoning_adapter, _ = make_adapters(_CustomEndParser)
    parser = reasoning_adapter(mock_tokenizer)
    ids = [_THINK_START_ID] + [_TEXT_ID] * 100
    assert parser.is_reasoning_end_streaming(ids, [_TEXT_ID]) is True
    assert sentinel == [len(ids)]


def test_every_registered_adapter_overrides_the_streaming_check():
    """Guard against a parser engine silently inheriting the O(n) base.

    The DeepSeek V4 port regressed exactly this way: the model moved onto the
    parser-engine framework and lost the delta-scoped override it had before.
    """
    adapters = [
        obj
        for name in dir(registered_adapters)
        if name.endswith("ReasoningAdapter")
        and isinstance(obj := getattr(registered_adapters, name), type)
        and issubclass(obj, ParserEngineReasoningAdapter)
    ]
    assert adapters, "no reasoning adapters discovered"
    for adapter in adapters:
        assert (
            adapter.is_reasoning_end_streaming
            is not ReasoningParser.is_reasoning_end_streaming
        ), (
            f"{adapter.__name__} inherits the full-rescan "
            "is_reasoning_end_streaming; structured-output decoding will be "
            "quadratic in the generated length"
        )
