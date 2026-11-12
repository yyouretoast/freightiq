"""Fast, offline unit tests for FreightIQ guardrails, SQL parsing, and retrieval helpers."""

import sys
from unittest.mock import MagicMock, patch

try:
    import ddgs  # noqa: F401
except ImportError:
    sys.modules.setdefault("ddgs", MagicMock())

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import config
from agent.nodes import (
    _prepare_context_messages,
    agent_node,
    get_windowed_messages,
)
from agent.tools import (
    _find_top_level_where,
    _split_sql_and_conditions,
    _strip_trailing_sql_clauses,
    carrier_sql_query,
    freight_class_calculator,
)
from rag.retriever import reciprocal_rank_fusion, sanitize_fts5_query


class TestSQLParsingAndGuardrails:
    def test_find_top_level_where_ignores_quotes_and_parens(self):
        sql = "SELECT * FROM carriers WHERE notes = 'WHERE x = 1' AND (hq_state = 'TX')"
        start, end = _find_top_level_where(sql)
        assert sql[start:end].upper() == "WHERE"
        assert start == sql.index("WHERE notes")

    def test_find_top_level_where_missing(self):
        assert _find_top_level_where("SELECT * FROM carriers LIMIT 5") == (-1, -1)

    def test_strip_trailing_sql_clauses_preserves_quoted_keywords(self):
        where_clause = "notes = 'ORDER BY priority' AND hq_state = 'OH' ORDER BY years_operating DESC LIMIT 10"
        conds, trailing = _strip_trailing_sql_clauses(where_clause)
        assert conds == "notes = 'ORDER BY priority' AND hq_state = 'OH'"
        assert trailing == "ORDER BY years_operating DESC LIMIT 10"

    def test_split_sql_and_conditions_handles_between_and_parens(self):
        where_conds = (
            "hq_state = 'OH' AND years_operating BETWEEN 5 AND 15 "
            "AND (equipment_types LIKE '%reefer%' AND safety_rating = 'satisfactory')"
        )
        parts = _split_sql_and_conditions(where_conds)
        assert parts == [
            "hq_state = 'OH'",
            "years_operating BETWEEN 5 AND 15",
            "(equipment_types LIKE '%reefer%' AND safety_rating = 'satisfactory')",
        ]

    @patch("agent.tools.query_carriers_sql")
    def test_carrier_sql_query_rejects_mutations(self, mock_sql):
        for bad_sql in [
            "DELETE FROM carriers",
            "DROP TABLE carriers",
            "UPDATE carriers SET hq_state = 'TX'",
            "/* comment */ INSERT INTO carriers VALUES (1)",
        ]:
            res = carrier_sql_query.invoke({"query": bad_sql})
            assert "Only SELECT queries are permitted" in res
        mock_sql.assert_not_called()

    @patch("agent.tools.query_carriers_sql", return_value="row1")
    def test_carrier_sql_query_strips_comments_and_wraps_limit_25(self, mock_sql):
        res = carrier_sql_query.invoke({
            "query": "-- leading comment\nSELECT carrier_name FROM carriers WHERE hq_state = 'TX';"
        })
        assert res == "row1"
        called_sql = mock_sql.call_args[0][0]
        assert called_sql == (
            "SELECT * FROM (SELECT carrier_name FROM carriers WHERE hq_state = 'TX') "
            "AS _bounded_carriers LIMIT 25"
        )

    @patch("agent.tools.query_carriers_sql")
    def test_zero_row_relaxation_preserves_safety_and_hazmat_filters(self, mock_sql):
        # First strict query returns no rows; relaxed query returns a match
        mock_sql.side_effect = [
            "No matching records found in the SQL database.",
            "Carrier A | satisfactory",
        ]
        query = (
            "SELECT * FROM carriers WHERE safety_rating = 'satisfactory' "
            "AND cargo_specializations LIKE '%hazardous materials%' "
            "AND hq_state = 'ZZ'"
        )
        res = carrier_sql_query.invoke({"query": query})
        assert "Relaxed partial search" in res
        assert "hq_state = 'ZZ'" in res
        relaxed_sql = mock_sql.call_args_list[1][0][0]
        assert "safety_rating = 'satisfactory'" in relaxed_sql
        assert "hazardous materials" in relaxed_sql
        assert "hq_state = 'ZZ'" not in relaxed_sql


class TestFreightClassCalculator:
    def test_standard_density_tier(self):
        # 12x12x12 = 1 ft^3, 20 lbs => 20 lb/ft^3 => Class 70 (15 <= density < 22.5)
        res = freight_class_calculator.invoke({
            "weight_lbs": 20.0,
            "length_in": 12.0,
            "width_in": 12.0,
            "height_in": 12.0,
            "cargo_description": "machined steel brackets",
        })
        assert "Standard NMFC Freight Class: 70" in res

    def test_exception_rule_overrides_density(self):
        res = freight_class_calculator.invoke({
            "weight_lbs": 50.0,
            "length_in": 12.0,
            "width_in": 12.0,
            "height_in": 12.0,
            "cargo_description": "fiberglass insulation rolls",
        })
        assert "fixed NMFC Class 150 overrides density calculation" in res

    def test_negation_prevents_false_exception_match(self):
        res = freight_class_calculator.invoke({
            "weight_lbs": 20.0,
            "length_in": 12.0,
            "width_in": 12.0,
            "height_in": 12.0,
            "cargo_description": "steel panels with no insulation",
        })
        assert "Standard NMFC Freight Class: 70" in res
        assert "EXCEPTION" not in res

    def test_invalid_inputs_rejected(self):
        for bad_val in (0.0, -5.0, float("nan"), float("inf")):
            res = freight_class_calculator.invoke({
                "weight_lbs": bad_val,
                "length_in": 12.0,
                "width_in": 12.0,
                "height_in": 12.0,
            })
            assert res.startswith("Error:")


class TestContextWindowAndLoopGuardrails:
    def test_windowed_messages_aligns_to_human_turn(self):
        msgs = [
            HumanMessage(content="turn 1"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "t1", "args": {}}]),
            ToolMessage(content="out1", tool_call_id="1", name="t1"),
            AIMessage(content="ans 1"),
            HumanMessage(content="turn 2"),
            AIMessage(content="", tool_calls=[{"id": "2", "name": "t2", "args": {}}]),
            ToolMessage(content="out2", tool_call_id="2", name="t2"),
            AIMessage(content="ans 2"),
            HumanMessage(content="turn 3"),
            AIMessage(content="ans 3"),
        ]
        windowed = get_windowed_messages(msgs, max_messages=4)
        assert isinstance(windowed[0], HumanMessage)
        assert len(windowed) <= 4
        assert windowed[0].content == "turn 3"

    def test_prepare_context_truncates_oversized_tool_output(self, monkeypatch):
        monkeypatch.setattr(config, "TOOL_TRUNCATION_LIMIT", 20)
        msgs = [
            HumanMessage(content="hi"),
            ToolMessage(content="x" * 100, tool_call_id="1", name="t1"),
        ]
        cleaned = _prepare_context_messages(msgs)
        assert "Output truncated to stay within model token quota" in cleaned[1].content
        assert cleaned[1].content.startswith("x" * 20)

    @patch("agent.nodes._invoke_with_retry")
    def test_exact_repeat_tool_call_triggers_unbound_guardrail(self, mock_invoke):
        mock_invoke.return_value = AIMessage(content="Synthesized answer")
        dup_call = [{"id": "c1", "name": "carrier_sql_query", "args": {"query": "SELECT 1"}}]
        state = {
            "messages": [
                HumanMessage(content="find carriers"),
                AIMessage(content="", tool_calls=dup_call),
                ToolMessage(content="res1", tool_call_id="c1", name="carrier_sql_query"),
                AIMessage(content="", tool_calls=dup_call),
                ToolMessage(content="res1", tool_call_id="c1", name="carrier_sql_query"),
            ]
        }
        out = agent_node(state)
        assert out["messages"][0].content == "Synthesized answer"
        # First positional arg is is_tool_bound=False for loop guardrail
        assert mock_invoke.call_args[0][0] is False

    @patch("agent.nodes._invoke_with_retry")
    def test_alternating_ping_pong_triggers_guardrail(self, mock_invoke):
        mock_invoke.return_value = AIMessage(content="Stopped ping-pong")
        state = {
            "messages": [
                HumanMessage(content="find carriers"),
                AIMessage(content="", tool_calls=[{"id": "1", "name": "tool_a", "args": {}}]),
                ToolMessage(content="a1", tool_call_id="1", name="tool_a"),
                AIMessage(content="", tool_calls=[{"id": "2", "name": "tool_b", "args": {}}]),
                ToolMessage(content="b1", tool_call_id="2", name="tool_b"),
                AIMessage(content="", tool_calls=[{"id": "3", "name": "tool_a", "args": {}}]),
                ToolMessage(content="a2", tool_call_id="3", name="tool_a"),
            ]
        }
        agent_node(state)
        assert mock_invoke.call_args[0][0] is False

    @patch("agent.nodes._invoke_with_retry")
    def test_loop_detection_is_scoped_to_current_turn(self, mock_invoke):
        mock_invoke.return_value = AIMessage(content="Normal response")
        call = [{"id": "1", "name": "carrier_sql_query", "args": {"query": "SELECT 1"}}]
        state = {
            "messages": [
                HumanMessage(content="turn 1"),
                AIMessage(content="", tool_calls=call),
                ToolMessage(content="r1", tool_call_id="1", name="carrier_sql_query"),
                AIMessage(content="answer 1"),
                HumanMessage(content="turn 2"),
                AIMessage(content="", tool_calls=call),
                ToolMessage(content="r2", tool_call_id="1", name="carrier_sql_query"),
            ]
        }
        agent_node(state)
        # Only 1 tool call in turn 2 -> normal tool-bound invocation (is_tool_bound=True)
        assert mock_invoke.call_args[0][0] is True


class TestRetrievalHelpers:
    def test_sanitize_fts5_query_strips_stopwords_and_keeps_hazard_digits(self):
        expr = sanitize_fts5_query("Find carriers for hazmat Class 3 and 8!")
        assert expr == '"hazmat" OR "Class" OR "3" OR "8"'
        assert sanitize_fts5_query("...") is None

    def test_reciprocal_rank_fusion_boosts_overlap_and_deduplicates(self):
        bm25 = [
            {"dot_number": "100", "carrier_name": "A"},
            {"dot_number": "200", "carrier_name": "B"},
        ]
        dense = [
            {"dot_number": "200", "carrier_name": "B"},
            {"dot_number": "300", "carrier_name": "C"},
        ]
        fused = reciprocal_rank_fusion(bm25, dense, k=60, top_n=3)
        dots = [item["dot_number"] for item in fused]
        assert dots[0] == "200"
        assert set(dots) == {"100", "200", "300"}
