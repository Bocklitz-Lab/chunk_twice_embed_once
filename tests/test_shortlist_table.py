import pandas as pd

from tools import shortlist_table as ST


def test_table_marks_stays_leaves_enters_and_writes_latex(tmp_path):
    sel = pd.DataFrame([{"model": m, "revision": "r", "geom_ChemHotpotQARetrieval": .5, "rank_ChemHotpotQARetrieval": 1, "p_top_ChemHotpotQARetrieval": p, "observed_top_ChemHotpotQARetrieval": o,
                         "geom_ChemNQRetrieval": .5, "rank_ChemNQRetrieval": 2, "p_top_ChemNQRetrieval": 0.1, "observed_top_ChemNQRetrieval": False, "observed_top": o, "max_p_top": p,
                         "selected_top_fraction": o, "selected_p0.4": p >= .4 or o, "selected_p0.5": p >= .5 or o, "selected_p0.6": p >= .6 or o, "selected_primary": p >= .5 or o}
                        for m, p, o in (("a/stay", .9, True), ("b/leave", .3, False), ("c/enter", .55, False), ("d/none", .0, False))])
    sel.to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "p.csv").write_text("model_name\na/stay\nb/leave\n")
    df = ST.build(tmp_path / "s.csv", tmp_path / "p.csv")
    assert dict(zip(df.model, df.change)) == {"a/stay": "stays", "b/leave": "leaves", "c/enter": "enters", "d/none": ""}
    ST.to_latex(df, tmp_path / "t.tex")
    tex = (tmp_path / "t.tex").read_text()
    assert "a/stay" in tex and "c/enter" in tex and "d/none" not in tex and "tab:supp_shortlist_selection" in tex
