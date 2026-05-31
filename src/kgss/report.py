"""Render the EFA review report as a self-contained HTML file."""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _scree_png(real_ev, rand_ev, n_factors: int) -> str:
    """Scree plot (real vs parallel-analysis random eigenvalues) as base64 PNG."""
    k = min(20, len(real_ev))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(1, k + 1)
    ax.plot(x, real_ev[:k], "o-", label="real")
    ax.plot(x, rand_ev[:k], "s--", color="grey", label="random (PA 95%)")
    ax.axvline(n_factors + 0.5, color="red", ls=":", label=f"retained = {n_factors}")
    ax.axhline(1.0, color="black", lw=0.5)
    ax.set_xlabel("factor")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Parallel analysis scree")
    ax.legend()
    ax.set_xticks(list(x))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _loadings_html(loadings: pd.DataFrame, cutoff: float = 0.40) -> str:
    """Loadings as a coloured heatmap table; |loading|<cutoff dimmed."""
    sty = (
        loadings.style.format("{:.2f}")
        .background_gradient(cmap="RdBu", vmin=-1, vmax=1, axis=None)
        .set_properties(**{"text-align": "center", "font-size": "11px"})
    )
    return sty.to_html()


CSS = """
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222;max-width:1100px}
h1{border-bottom:3px solid #333;padding-bottom:6px}
h2{margin-top:34px;background:#f0f0f0;padding:6px 10px;border-left:4px solid #555}
table{border-collapse:collapse;margin:8px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:3px 7px}
.flag{color:#b00;font-weight:bold}
.ok{color:#070}
.note{background:#fff7e6;border:1px solid #f0c36d;padding:10px;border-radius:5px}
code{background:#eee;padding:1px 4px}
</style>
"""


def build_report(art: dict) -> str:
    """``art`` is the artifact dict assembled by the phase1 script."""
    p = art
    scree = _scree_png(
        p["pa"]["real_eigenvalues"], p["pa"]["random_eigenvalues"], p["pa"]["n_factors"]
    )

    def tbl(df, **kw):
        return df.to_html(**kw)

    kmo = p["kmo"]
    kmo_flag = "ok" if kmo["kmo_model"] > 0.6 else "flag"
    var_total = p["variance"]["prop_var"].sum()
    var_flag = "ok" if var_total >= 0.50 else "flag"

    html = [CSS, "<h1>KGSS EFA Report (2016 wave)</h1>"]
    html.append(
        "<p class='note'><b>STATUS: EFA review complete.</b> "
        "n_factors=4; 5 unattached items dropped; factor identities + direction "
        "fixed (§5); POLEFF kept as 정치효능감(신뢰/냉소). BNS direct-resampling pool "
        "built (§9).</p>"
    )

    html.append("<h2>1. Data + missing handling (Step 1)</h2>")
    html.append(
        f"<ul><li>Original records: <b>{p['clean']['n_original']:,}</b></li>"
        f"<li>After dropping &gt;30% item-missing: <b>{p['clean']['n_after_drop']:,}</b></li>"
        f"<li>Items: 42 backbone; demographics: {', '.join(p['demo_cols'])}</li></ul>"
    )
    html.append("<b>Top-10 item missingness (post-drop):</b>")
    html.append(tbl(p["clean"]["per_item_missing"].head(10).to_frame("n_missing")))
    html.append("<b>Wave coverage (respondents × mean items answered):</b>")
    html.append(tbl(p["clean"]["wave_coverage"]))

    html.append("<h2>2. KMO + Bartlett (Step 4)</h2>")
    html.append(
        f"<ul><li>KMO model: <span class='{kmo_flag}'>{kmo['kmo_model']:.3f}</span> "
        f"(threshold &gt; 0.6)</li>"
        f"<li>Bartlett χ² = {kmo['bartlett_chi_sq']:,.0f}, p = {kmo['bartlett_p']:.1e}</li></ul>"
        "<b>5 lowest per-item KMO:</b>"
    )
    html.append(tbl(kmo["kmo_per_item"].head(5).round(3).to_frame("kmo")))

    html.append("<h2>3. Factor count — parallel analysis (Step 5)</h2>")
    html.append(
        f"<p>Horn parallel analysis retained <b>{p['pa']['n_factors']}</b> factors "
        f"(Kaiser &gt;1 would give {p['pa']['kaiser_n']}). "
        f"Prior hypothesis: 6 (F1-F6).</p>"
        f"<img src='data:image/png;base64,{scree}'/>"
    )

    html.append(f"<h2>4. EFA loadings — {p['rotation']} (Step 6-7)</h2>")
    html.append(
        f"<p>Variance explained: <span class='{var_flag}'>{var_total:.1%}</span> "
        f"(threshold ≥ 50%).</p>"
    )
    html.append(tbl(p["variance"].round(3)))
    html.append("<b>Loadings heatmap (cutoff |0.40|):</b>")
    html.append(_loadings_html(p["loadings"]))

    html.append("<h2>5. Factor identities + direction</h2>")
    html.append(
        "<p>Empirical factors mapped to confirmed identities by signature-item "
        "loading. Each substantive item recodes so HIGH=진보, so factors are "
        "<b>flipped</b> to put 보수 (정치) / 신뢰 (효능감) at the POSITIVE score pole "
        "(<code>positive_label</code>). <code>political:false</code> = excluded from "
        "the 보수/진보 BNS mapping.</p>"
    )
    html.append(tbl(p["direction"], index=False))
    html.append("<b>Signature-item match scores (empirical factor × identity):</b>")
    html.append(tbl(p["match_score"]))

    html.append("<h2>6. TBD-EFA reverse-coding recommendations (Step 7)</h2>")
    html.append(tbl(p["tbd"], index=False))

    html.append("<h2>7. Cross-loadings (≥2 factors |loading|&gt;0.40)</h2>")
    if len(p["cross"]):
        rate = 100 * len(p["cross"]) / len(p["loadings"])
        flag = "ok" if rate < 20 else "flag"
        html.append(f"<p><span class='{flag}'>{len(p['cross'])} items "
                    f"({rate:.0f}%)</span> cross-load (threshold &lt; 20%).</p>")
        html.append(tbl(p["cross"], index=False))
    else:
        html.append("<p class='ok'>None.</p>")
    if p["none_items"]:
        html.append(f"<p class='flag'>Items loading nowhere (&lt;0.40): "
                    f"{', '.join(p['none_items'])}</p>")

    html.append("<h2>8. Factor-score distribution (Step 8)</h2>")
    html.append(tbl(p["score_dist"]))

    html.append("<h2>9. Group respondent pools — BNS direct resampling (Step 9)</h2>")
    sr = p["sparsity"]
    html.append(
        "<p>BNS gives each agent a <b>real</b> KGSS respondent's factor-score "
        "vector (sampled with replacement) from its demographic×orientation cell, "
        "with a fallback chain — not Gaussian noise around a cell mean.</p>"
        "<ul>"
        f"<li>Respondents grouped: {sr['n_respondents_grouped']:,}</li>"
        f"<li>Distinct full cells (age×sex×region5×orient): {sr['n_distinct_full_cells']:,} "
        f"(mean {sr['mean_n_per_full_cell']} respondents/cell)</li>"
        f"<li>Primary cells with n≥{sr['min_cell_size']}: {sr['n_primary_cells_ge_min']:,} "
        f"→ {sr['pct_in_usable_primary']}% of respondents land in a usable primary cell</li>"
        f"<li>Fallback (region5×orient): {sr['n_region_orient_cells']} cells, "
        f"mean {sr['mean_n_per_region_orient']} respondents/cell</li>"
        f"<li>Fallback (orientation-only) pool sizes: {sr['orient_pool_sizes']}</li>"
        "</ul>"
    )
    if p.get("resampling"):
        rs = p["resampling"]
        html.append(
            "<b>Resampling diversity (projected, orientation-only worst case):</b>"
            "<ul>"
            f"<li>{rs['projected_n_agents']:,} agents would use "
            f"{rs['projected_unique_respondents_used']:,} / {rs['total_pool_respondents']:,} "
            f"distinct real respondents</li>"
            f"<li>Heaviest single-respondent re-use: {rs['max_single_respondent_reuse']}×</li>"
            "</ul>"
        )

    html.append("<h2>EFA review resolutions</h2>")
    html.append(
        "<ul>"
        "<li><b>Factor count:</b> 4 (confirmed)</li>"
        "<li><b>Names:</b> 경제·재분배(정부책임) / 대북·안보 / 시장·민영화 / 정치효능감(신뢰·냉소)</li>"
        "<li><b>Dropped (load <0.40):</b> CNSRVTV4, 6, 8, 10 + GOVRES11 → paper limitations</li>"
        "<li><b>Direction:</b> positive score = 보수 (정치 factors) / 신뢰 (효능감); §5</li>"
        "<li><b>Variance 29.2%:</b> accepted</li>"
        "<li><b>POLEFF5/6:</b> kept as 정치효능감; <b>excluded</b> from 보수/진보 BNS mapping</li>"
        "</ul>"
    )
    return "\n".join(html)
