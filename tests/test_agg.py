"""Aggregation pushdown for class A: latest and max return the one row that is the gold answer.

Fixture facts (tests/fixtures/labevents.csv): admission 02 has creatinine 6.4
(07-15) then 4.9 (07-18), so its latest row is not its max row; admission 04
has 3.2 then 1.8.
"""

import random

import pytest

from clinrag.config import PlannerConfig
from clinrag.eval.metrics import retrieval_score
from clinrag.planner.cost import relational_step_cost
from clinrag.planner.execute import execute
from clinrag.planner.plan import RelationalFilter
from clinrag.planner.planner import CostPlanner
from clinrag.planner.probes import Probes
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import RelationalQuery
from clinrag.sources.reference import build_reference_sources
from clinrag.workload.labs import LABS
from clinrag.workload.paraphrase import FAMILIES, paraphrase
from clinrag.workload.schema import question_from_dict, question_to_dict, template_agg
from clinrag.workload.templates import A_KINDS, WorkloadData, class_a
from tests.test_workload import CONCEPT_TAGS, FIXTURES, write_tags

CREATININE = 50912


def h(n):
    return 20000000 + n


@pytest.fixture(scope="module")
def sources():
    return build_reference_sources(FIXTURES, FIXTURES)


@pytest.fixture(scope="module")
def class_a_questions(tmp_path_factory):
    """Every class A template on every fixture admission that has the lab."""
    notes_dir = tmp_path_factory.mktemp("notes")
    write_tags(notes_dir / "note_concepts.csv", CONCEPT_TAGS)
    data = WorkloadData.load(FIXTURES, notes_dir)
    return [q.with_qid(f"A{i:03d}") for i, q in enumerate(class_a(data, None, random.Random(0)))]


def test_filter_validates_agg():
    assert StructuredFilter(CREATININE, hadm_id=1).agg is None
    assert StructuredFilter(CREATININE, hadm_id=1, agg="latest").agg == "latest"
    with pytest.raises(ValueError):
        StructuredFilter(CREATININE, hadm_id=1, agg="mean")
    with pytest.raises(ValueError):
        StructuredFilter(CREATININE, agg="max")  # one row of which admission?


def test_templates_set_agg_from_the_kind(class_a_questions):
    seen = set()
    for q in class_a_questions:
        kind = next(k for k in ("ever_above", "latest", "max") if q.template_id.startswith(f"A_{k}_"))
        seen.add(q.template_id)
        flt = q.spec.relational_filter
        if kind == "ever_above":
            assert flt.agg is None and flt.op == ">"  # stays a predicate
        else:
            assert flt.agg == kind and flt.op is None
    assert seen == {f"A_{kind}_{lab}" for kind in A_KINDS for lab in LABS}


def test_single_row_equals_gold_for_every_class_a_template(sources, class_a_questions):
    checked = set()
    for q in class_a_questions:
        flt = q.spec.relational_filter
        if flt.agg is None:
            continue
        query = RelationalQuery(lab=flt)
        result = sources.relational.run(query)
        assert len(result.items) == 1, q.spec.text
        row = result.items[0]
        assert row["hadm_id"] == q.gold["hadm_id"] and row["itemid"] == flt.itemid
        assert row["valuenum"] == pytest.approx(q.gold["answer"]), q.spec.text
        assert sources.relational.count(query) == 1
        assert result.ids == [q.gold["hadm_id"]] == sources.relational.matching_ids(query)
        checked.add(q.template_id)
    assert checked == {f"A_{kind}_{lab}" for kind in ("latest", "max") for lab in LABS}


def test_latest_and_max_pick_different_rows(sources):
    latest = sources.relational.run(RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(2), agg="latest")))
    highest = sources.relational.run(RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(2), agg="max")))
    every = sources.relational.run(RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(2))))
    assert [i["valuenum"] for i in latest.items] == [4.9]
    assert [i["valuenum"] for i in highest.items] == [6.4]
    assert [i["valuenum"] for i in every.items] == [6.4, 4.9]  # no agg: every row, oldest first
    assert latest.items[0]["charttime"] == "2151-07-18 06:00:00"
    assert latest.items[0]["label"] and latest.items[0]["unit"] == "mg/dL"


def test_agg_with_no_matching_rows_returns_nothing(sources):
    query = RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=99, agg="latest"))
    assert sources.relational.run(query).items == []
    assert sources.relational.count(query) == 0


def test_gold_and_retrieval_score_unchanged(sources, class_a_questions):
    """The planner's chosen plan returns one row for latest/max and still scores 1.0 against the same gold."""
    config = PlannerConfig()
    planner = CostPlanner(sources, config)
    for q in class_a_questions:
        result = execute(planner.plan(q.spec).plan, sources, q.spec, config)
        assert retrieval_score(q, result.items, config.top_k).value == 1.0, q.spec.text
        if q.spec.relational_filter.agg is not None:
            assert len(result.items) == 1
            # the same question without the aggregate: gold is among the rows it returned before
            plain = StructuredFilter(q.spec.relational_filter.itemid, hadm_id=q.gold["hadm_id"])
            rows = sources.relational.run(RelationalQuery(lab=plain)).items
            values = [r["valuenum"] for r in rows]
            expected = max(values) if q.spec.relational_filter.agg == "max" else values[-1]
            assert q.gold["answer"] == pytest.approx(expected)


def test_cost_model_estimates_one_row_with_agg():
    config = PlannerConfig(avg_tokens_per_row=10)
    agg = RelationalFilter(StructuredFilter(CREATININE, hadm_id=1, agg="latest"))
    plain = RelationalFilter(StructuredFilter(CREATININE, hadm_id=1))
    assert relational_step_cost(agg, Probes(), config).est_tokens == 10  # unprobed: one row
    assert relational_step_cost(agg, Probes(relational_rows=40), config).est_tokens == 10
    assert relational_step_cost(agg, Probes(relational_rows=0), config).est_tokens == 0
    assert relational_step_cost(plain, Probes(), config).est_tokens == 0  # unprobed lower bound, as before
    assert relational_step_cost(plain, Probes(relational_rows=40), config).est_tokens == 400
    not_evidence = RelationalFilter(StructuredFilter(CREATININE, hadm_id=1, agg="max"), as_evidence=False)
    assert relational_step_cost(not_evidence, Probes(), config).est_tokens == 0


def test_agg_round_trips_and_old_files_get_it_from_the_template(class_a_questions):
    for q in class_a_questions:
        d = question_to_dict(q)
        assert question_from_dict(d) == q
        del d["relational_filter"]["agg"]  # a workload file written before agg existed
        assert question_from_dict(d) == q
    assert (template_agg("A_latest_creatinine"), template_agg("A_max_potassium")) == ("latest", "max")
    assert template_agg("A_ever_above_creatinine") is None and template_agg("C_creatinine_dialysis") is None


def test_paraphrases_keep_agg(class_a_questions):
    for q in class_a_questions:
        for family in FAMILIES["A"]:
            p = paraphrase(q, family).question
            assert p.spec.relational_filter == q.spec.relational_filter and p.gold == q.gold


def test_agg_is_only_on_class_a():
    spec = QuestionSpec(qid="c", text="c", qclass="C", relational_filter=StructuredFilter(CREATININE, ">", 2.0),
                        note_query="dialysis", required_sources=frozenset({"relational", "notes"}))
    assert spec.relational_filter.agg is None
