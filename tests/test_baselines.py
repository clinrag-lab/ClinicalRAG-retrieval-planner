"""Fixed-order baselines and the answer slot on the fake sources (data and config as in test_planner)."""

from clinrag.baselines.fixed import AlwaysFilterFirstRouter, AlwaysSearchFirstRouter, QueryAllRouter
from clinrag.llm.answerer import NullAnswerer
from clinrag.planner.execute import execute
from clinrag.planner.router import OracleCostRouter
from clinrag.question import QuestionSpec, StructuredFilter
from tests.fakes import CREATININE
from tests.test_planner import CONFIG, make_sources, mixed_question

TIGHT = {1, 2, 3, 4}  # 40 notes: the rule picks filter_first
LOOSE = set(range(1, 61))  # 600 notes: the rule picks search_first


def test_always_search_first_overrides_the_rule():
    sources = make_sources(high=TIGHT)
    oracle = OracleCostRouter(sources, CONFIG).decide(mixed_question())
    forced = AlwaysSearchFirstRouter(sources, CONFIG).decide(mixed_question())
    assert oracle.plan.name == "relational+notes:filter_first"
    assert forced.plan.name == "relational+notes:search_first"
    assert forced.plan.note_search.n_candidates == 125
    assert forced.chosen.feasible
    assert [c.plan.name for c in forced.candidates] == [c.plan.name for c in oracle.candidates]
    result = execute(forced.plan, sources, mixed_question(), CONFIG)
    assert [s.step for s in result.steps] == ["relational", "notes:search_first"]


def test_always_filter_first_overrides_the_rule():
    sources = make_sources(high=LOOSE)
    assert OracleCostRouter(sources, CONFIG).decide(mixed_question()).plan.name == "relational+notes:search_first"
    forced = AlwaysFilterFirstRouter(sources, CONFIG).decide(mixed_question())
    assert forced.plan.name == "relational+notes:filter_first"
    result = execute(forced.plan, sources, mixed_question(), CONFIG)
    assert [s.step for s in result.steps] == ["relational", "notes:filter_first"]
    assert sources.notes.search_args[-1][1] == tuple(sorted(LOOSE))


def test_query_all_router():
    sources = make_sources(high=TIGHT)
    decision = QueryAllRouter(sources, CONFIG).decide(mixed_question())
    assert decision.plan.name == "query_all"
    assert decision.chosen.cost is not None and decision.predicted_sources == {"relational", "notes"}
    result = execute(decision.plan, sources, mixed_question(), CONFIG)
    assert [s.step for s in result.steps] == ["relational", "notes:unfiltered"]


def test_order_baselines_keep_oracle_plan_without_filtered_note_search():
    question = QuestionSpec(
        qid="a", text="highest creatinine in admission 2", qclass="A",
        relational_filter=StructuredFilter(CREATININE, hadm_id=2), required_sources=frozenset({"relational"}),
    )
    sources = make_sources(high=TIGHT)
    oracle = OracleCostRouter(sources, CONFIG).decide(question)
    for router in (AlwaysFilterFirstRouter(sources, CONFIG), AlwaysSearchFirstRouter(sources, CONFIG)):
        assert router.decide(question).plan == oracle.plan
    assert QueryAllRouter(sources, CONFIG).decide(question).plan.name == "query_all"


def test_baseline_names():
    sources = make_sources(high=TIGHT)
    assert QueryAllRouter(sources, CONFIG).name == "query_all"
    assert AlwaysFilterFirstRouter(sources, CONFIG).name == "always_filter_first"
    assert AlwaysSearchFirstRouter(sources, CONFIG).name == "always_search_first"


def test_null_answerer():
    assert NullAnswerer().answer(mixed_question(), [{"hadm_id": 1, "text": "x"}]) == (None, 0, 0.0)
