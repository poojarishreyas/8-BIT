"""Unit tests for agent tool-call parsing and argument matching (no model needed)."""
from eval.agent.run import arg_ok, parse_calls


def test_parse_qwen_tool_call():
    t = 'x <tool_call>\n{"name": "get_weather", "arguments": {"city": "ಮೈಸೂರು"}}\n</tool_call> y'
    assert parse_calls(t) == [{"name": "get_weather", "arguments": {"city": "ಮೈಸೂರು"}}]


def test_parse_string_arguments_and_multiple_calls():
    t = ('<tool_call>{"name": "a", "arguments": "{\\"q\\": 1}"}</tool_call>'
         '<tool_call>{"name": "b", "arguments": {}}</tool_call>')
    assert [c["name"] for c in parse_calls(t)] == ["a", "b"]
    assert parse_calls(t)[0]["arguments"] == {"q": 1}


def test_parse_malformed():
    assert parse_calls("<tool_call>{not json}</tool_call>")[0].get("malformed")


def test_arg_matching_alternatives_and_normalisation():
    assert arg_ok(["Mysuru", "Mysore", "ಮೈಸೂರು"], " mysore. ")
    assert arg_ok("500D", "500d")
    assert not arg_ok("Mysuru", "Mandya")
    assert not arg_ok("Mysuru", None)
