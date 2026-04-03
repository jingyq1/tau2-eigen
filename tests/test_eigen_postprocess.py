from types import SimpleNamespace

from tau2.eigen.postprocess import (
    _build_sample_status_markdown,
    _build_sample_status_rows,
)


def _make_simulation(task_id: str, *, overall: bool, config: bool, func: bool, llm: bool):
    return SimpleNamespace(
        task_id=task_id,
        reward_info=SimpleNamespace(
            info={
                "eigen_eval": {
                    "overall_pass": overall,
                    "config_match": config,
                    "function_call_match": func,
                    "llm_judgment": {"passed": llm},
                }
            }
        ),
    )


def test_build_sample_status_rows_and_markdown_table():
    results = SimpleNamespace(
        simulations=[
            _make_simulation("eigen_000012", overall=True, config=True, func=True, llm=True),
            _make_simulation("eigen_000013", overall=False, config=True, func=False, llm=True),
        ]
    )

    rows = _build_sample_status_rows(results)

    assert rows == [
        {
            "sample": "eigen_000012",
            "status": "PASS",
            "config": True,
            "func": True,
            "llm": True,
        },
        {
            "sample": "eigen_000013",
            "status": "FAIL",
            "config": True,
            "func": False,
            "llm": True,
        },
    ]

    markdown = _build_sample_status_markdown(rows)

    assert "| sample | status | config | func | llm |" in markdown
    assert "| eigen_000012 | PASS | True | True | True |" in markdown
    assert "| eigen_000013 | FAIL | True | False | True |" in markdown
