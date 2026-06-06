"""
Allied Health Nudge — Streamlit Interactive Demonstrator
=========================================================
A live, interactive web app that lets anyone see the model working in real time.

Three pages (sidebar navigation):
  1. Member Lookup — search any member, see their risk profile and nudge message
  2. Population Overview — charts and numbers across all 50,000 members
  3. ROI Simulator — sliders to estimate financial savings

Designed for layman audiences: plain English, no unexplained jargon,
consistent colours, and a "What This Means" explanation on every page.

Run:  streamlit run src/streamlit_app.py --server.port 8501
Access from T14s: http://100.68.3.15:8501
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# Page configuration — set this FIRST, before any other Streamlit commands
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Allied Health Nudge — Live Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Colour scheme — consistent across all pages and all Stage 6 deliverables
# ═══════════════════════════════════════════════════════════════════════════════

COLOURS = {
    "High": "#E8533B",      # Coral red — urgent attention needed
    "Medium": "#F2A65A",    # Amber — needs attention
    "Low": "#4A6FA5",       # Slate blue — low concern
    "Green": "#00b894",     # Positive action / savings
    "Dark": "#2d3436",      # Text
    "Grey": "#636e72",      # Secondary text
    "Bg": "#f5f6fa",        # Page background
}

TIER_ORDER = {"Low": 1, "Medium": 2, "High": 3}  # For sorting

# ═══════════════════════════════════════════════════════════════════════════════
# Load data and model — run once at startup, cached so it doesn't re-run
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_data
def load_scored_members() -> pd.DataFrame:
    """Load the 50,000 pre-scored members from the CSV output of Stage 5."""
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "outputs" / "scored_members.csv"
    if not path.exists():
        st.error(f"❌ Cannot find scored members file at: {path}")
        st.stop()
    df = pd.read_csv(path)
    # Ensure condition_cluster is plain text (not a pandas category type)
    df["condition_cluster"] = df["condition_cluster"].astype(str)
    return df


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    """Load SHAP feature importance rankings from Stage 4 output.
    Returns None if the file doesn't exist."""
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "outputs" / "feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_scoring_summary() -> dict:
    """Load the one-page summary with counts, rates, and thresholds."""
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "outputs" / "scoring_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_eval_metrics() -> dict:
    """Load model evaluation metrics: ROC-AUC, precision, recall, etc."""
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "outputs" / "eval_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# Load everything at startup
scored = load_scored_members()
feat_imp = load_feature_importance()
summary = load_scoring_summary()
metrics = load_eval_metrics()

# Key numbers used across multiple pages
N_MEMBERS = len(scored)
N_HIGH_RISK: int = int((scored["risk_tier"] == "High").sum())  # type: ignore[arg-type]
N_MEDIUM_RISK: int = int((scored["risk_tier"] == "Medium").sum())  # type: ignore[arg-type]
N_NUDGE: int = int(scored["nudge_signal"].sum())  # type: ignore[arg-type]
NUDGE_RATE = N_NUDGE / N_MEMBERS
PRECISION_TOP20 = metrics.get("precision_top20pct", 0.393)
RECALL_TOP20 = metrics.get("recall_top20pct", 0.446)
ROC_AUC = metrics.get("roc_auc", 0.742)

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — navigation between the three pages
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h2 style="margin: 0; color: #2d3436;">🏥 Allied Health Nudge</h2>
        <p style="margin: 4px 0 0 0; font-size: 12px; color: #636e72;">
            Live Model Demonstrator
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Choose a page:",
    ["🔍 Member Lookup", "📊 Population Overview", "💰 ROI Simulator"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This is a proof of concept on synthetic data. "
    "Not for clinical use. Colour key: "
    "🟥 High risk  🟧 Medium  🟦 Low  🟩 Positive."
)
st.sidebar.caption(f"Data: {N_MEMBERS:,} members scored • Model: LightGBM • AUC: {ROC_AUC:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Shared helper functions
# ═══════════════════════════════════════════════════════════════════════════════


def format_pct(value: float) -> str:
    """Format a decimal (0.682) as a readable percentage (68.2%)."""
    return f"{value * 100:.1f}%"


def tier_badge(tier: str) -> str:
    """Return a coloured HTML badge for a risk tier. Used in metric cards."""
    colour = COLOURS.get(tier, COLOURS["Grey"])
    return f'<span style="background:{colour};color:white;padding:4px 14px;border-radius:12px;font-weight:600;font-size:14px;">{tier.upper()}</span>'


def metric_card(label: str, value: str, colour: str = COLOURS["Dark"], help_text: str = "") -> None:
    """Display a single metric card with large value and small label below."""
    st.markdown(
        f"""
        <div style="background:white;border-radius:10px;padding:16px 20px;
                    text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07);
                    border-left:4px solid {colour};height:100%;">
            <div style="font-size:26px;font-weight:700;color:{COLOURS["Dark"]};margin-bottom:4px;">{value}</div>
            <div style="font-size:12px;color:{COLOURS["Grey"]};line-height:1.3;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
        help=help_text,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Member Lookup
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🔍 Member Lookup":
    st.title("🔍 Member Lookup")
    st.caption(
        "Search for any member to see their risk profile and personalised nudge "
        "recommendation. Or click the button to see a random high-risk member."
    )

    # ── Controls row ──────────────────────────────────────────────────────────
    col_search, col_random, col_spacer = st.columns([3, 2, 5])

    with col_search:
        member_id_input = st.text_input(
            "Search by Member ID",
            placeholder="e.g. acc25029-406b-4afa-a15d-1c2baa775f6b",
            label_visibility="collapsed",
        )

    with col_random:
        show_random = st.button(
            "🎲 Show Me a Random High-Risk Member",
            use_container_width=True,
            help="Picks a random member from the High risk tier — the most interesting cases to demonstrate.",
        )

    # ── Determine which member to show ────────────────────────────────────────
    member = None

    if show_random:
        # Pick a random member from the High risk tier
        high_risk_ids = scored.loc[scored["risk_tier"] == "High", "member_id"].values
        if len(high_risk_ids) > 0:
            random_id = random.choice(high_risk_ids)
            member = scored[scored["member_id"] == random_id].iloc[0]

    elif member_id_input:
        matches = scored[scored["member_id"] == member_id_input.strip()]
        if len(matches) == 0:
            st.warning(
                f"❌ No member found with ID: `{member_id_input}`. "
                "Try another ID, or click the random button above."
            )
        else:
            member = matches.iloc[0]

    # If no member selected yet, show a welcome message
    if member is None and not show_random and not member_id_input:
        st.info(
            "👆 **Search for a member** by typing their ID above, "
            "or click **'Show Me a Random High-Risk Member'** to jump straight to an interesting case."
        )

    # ── Display the selected member ───────────────────────────────────────────
    if member is not None:
        st.markdown("---")

        # Row 1: Four key metric cards
        col1, col2, col3, col4 = st.columns(4)

        risk_tier = member["risk_tier"]
        risk_score = member["risk_score"]
        risk_calibrated = member["risk_score_calibrated"]
        nudge = member["nudge_signal"]

        with col1:
            colour = COLOURS.get(risk_tier, COLOURS["Grey"])
            st.markdown(
                f"""
                <div style="background:white;border-radius:10px;padding:16px 20px;
                            text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07);
                            border-left:4px solid {colour};">
                    <div style="font-size:12px;color:{COLOURS["Grey"]};margin-bottom:6px;">Risk Tier</div>
                    <div style="font-size:22px;font-weight:700;color:{colour};">{risk_tier.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            metric_card(
                "Risk Score (raw)",
                f"{risk_score:.3f}",
                help_text="The model's raw output. Higher = more likely to have an acute event. "
                "Used for ranking members against each other.",
            )

        with col3:
            # Calibrated score — shown as a percentage for layman readability
            metric_card(
                "Estimated Risk",
                format_pct(risk_calibrated),
                colour=COLOURS["Green"] if risk_calibrated < 0.3 else COLOURS["Medium"],
                help_text="Estimated chance (as a percentage) that this member will visit "
                "the emergency department in the next 6 months. Adjusted so the "
                "number is easy to understand.",
            )

        with col4:
            if nudge == 1:
                nudge_html = f'<span style="color:{COLOURS["Green"]};font-weight:700;">✅ YES</span>'
                nudge_label = "Nudge Recommended"
            else:
                nudge_html = '<span style="color:#636e72;font-weight:600;">—</span>'
                nudge_label = "No Nudge Needed"
            st.markdown(
                f"""
                <div style="background:white;border-radius:10px;padding:16px 20px;
                            text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07);
                            border-left:4px solid {COLOURS["Green"] if nudge == 1 else COLOURS["Grey"]};">
                    <div style="font-size:12px;color:{COLOURS["Grey"]};margin-bottom:6px;">{nudge_label}</div>
                    <div style="font-size:22px;">{nudge_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Row 2: Member profile card
        st.markdown("### 👤 Member Profile")

        profile_col1, profile_col2, profile_col3 = st.columns(3)

        with profile_col1:
            st.markdown(
                f"""
                | Detail | Value |
                |---|---|
                | **Age** | {int(member["age"])} |
                | **Gender** | {member["gender"]} |
                | **Plan Type** | {member["plan_type"]} |
                | **Employer Group** | {member["employer_group_id"]} |
                """,
            )

        with profile_col2:
            st.markdown(
                f"""
                | Detail | Value |
                |---|---|
                | **Main Health Condition** | {member["condition_cluster"]} |
                | **Number of Conditions** | {int(member["comorbidity_count"])} |
                | **GP Visits (6 months)** | {int(member["gp_visits_6m"])} |
                | **Allied Health Visits (6m)** | {int(member["allied_health_claims_6m"])} |
                """,
            )

        with profile_col3:
            # Show what allied health benefits they've used
            utilisation = member["allied_health_utilisation_rate"]
            days_since = int(member["days_since_last_allied"])
            days_text = f"{days_since} days ago" if days_since < 999 else "Never used"

            st.markdown(
                f"""
                | Detail | Value |
                |---|---|
                | **Benefit Use Rate** | {format_pct(utilisation)} |
                | **Last Allied Health Visit** | {days_text} |
                | **Total Sessions Remaining** | {int(member["sessions_remaining_total"])} |
                """,
            )

        # Compute these once — used in both nudge and non-nudge paths
        sessions_left = int(member["sessions_remaining_total"])

        # Row 3: Nudge recommendation (only if nudge=YES)
        if nudge == 1:
            st.markdown("### ✉️ Personalised Nudge Message")

            modality = member["recommended_modality"]

            # Map modality to the correct sessions-remaining column
            modality_session_map = {
                "Physiotherapy": "sessions_remaining_physio",
                "Chiropractic": "sessions_remaining_chiro",
                "Dietetics": "sessions_remaining_dietetics",
                "Psychology": "sessions_remaining_psychology",
            }
            session_col = modality_session_map.get(modality, "")
            sessions_for_modality = int(member.get(session_col, 0)) if session_col else 0

            # Build the nudge message
            nudge_message = (
                f"You have {sessions_for_modality} {modality} sessions remaining "
                f"this year — book now to manage your health."
            )

            st.markdown(
                f"""
                <div style="background:#f0faf5;border:2px solid {COLOURS["Green"]};
                            border-radius:12px;padding:24px 32px;margin:12px 0;">
                    <div style="font-size:18px;font-weight:600;color:{COLOURS["Dark"]};margin-bottom:12px;">
                        📱 The model would send this message:
                    </div>
                    <div style="font-size:17px;color:{COLOURS["Dark"]};font-style:italic;line-height:1.5;">
                        "{nudge_message}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show remaining sessions across all four benefit types
            st.caption("**All benefit sessions remaining for this member:**")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Physiotherapy", int(member["sessions_remaining_physio"]))
            with s2:
                st.metric("Chiropractic", int(member["sessions_remaining_chiro"]))
            with s3:
                st.metric("Dietetics", int(member["sessions_remaining_dietetics"]))
            with s4:
                st.metric("Psychology", int(member["sessions_remaining_psychology"]))

        else:
            # Member is not being nudged — explain why in plain English
            st.markdown("### ✉️ Nudge Status")

            zero_usage = int(member["zero_allied_health_flag"])

            if sessions_left == 0:
                reason = (
                    "This member has **used all their allied health sessions** for the year. "
                    "There is nothing left to offer them. If they are High risk, "
                    "this may be a **plan design issue** — their benefit limits may be too low "
                    "for their level of need."
                )
            elif zero_usage == 0:
                reason = (
                    "This member is **already using their allied health benefits**. "
                    "They don't need a nudge — they're already engaged with their care. "
                    "The nudge programme targets members who have benefits but aren't using them."
                )
            else:
                reason = (
                    "This member's **risk level is Low** — the model estimates they are unlikely "
                    "to need emergency care in the next 6 months. Nudges are reserved for "
                    "Medium and High risk members where the potential savings justify the outreach."
                )

            st.info(reason)

        # Row 4: "What This Means" — plain English explanation, always present
        st.markdown("---")
        st.markdown("### 📖 What This Means")

        # Build a plain-English summary based on risk tier
        tier_explanations = {
            "High": (
                f"This member has a **high estimated risk** ({format_pct(risk_calibrated)}) "
                f"of visiting the emergency department in the next 6 months. "
                f"Only 10% of members (about {N_HIGH_RISK:,} out of {N_MEMBERS:,}) fall into this category. "
                f"This is the top priority group for outreach."
            ),
            "Medium": (
                f"This member has a **medium estimated risk** ({format_pct(risk_calibrated)}) "
                f"of visiting the emergency department in the next 6 months. "
                f"About 10% of members fall into this category. "
                f"They should be contacted after the High risk group."
            ),
            "Low": (
                f"This member has a **low estimated risk** ({format_pct(risk_calibrated)}) "
                f"of visiting the emergency department in the next 6 months. "
                f"About 80% of members fall into this category. "
                f"No outreach is recommended at this time."
            ),
        }

        st.markdown(tier_explanations.get(risk_tier, ""))

        if nudge == 1:
            st.markdown(
                f"\n\nThe model recommends sending a nudge because:\n"
                f"- They are at **{risk_tier} risk** of needing emergency care\n"
                f"- They have **{int(member['sessions_remaining_total'])} unused sessions** across their benefits\n"
                f"- They have **not used any allied health services** in the past 6 months\n\n"
                f"*A nudge is only sent when ALL three of these conditions are met. "
                f"The message includes exactly what they're entitled to — "
                f"no generic \"check your benefits\" emails.*"
            )

        if sessions_left == 0 and risk_tier == "High":
            st.markdown(
                "\n⚠️ **Plan design note:** This member is at High risk but has exhausted "
                "all their benefit sessions. Their plan may not provide enough allied health "
                "cover for their level of need. This is flagged for plan design review — "
                "in practice, the employer or insurer would review whether the benefit limits "
                "are appropriate."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Population Overview
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Population Overview":
    st.title("📊 Population Overview")
    st.caption(
        "The big picture — what the model found across all 50,000 members. "
        "Hover over any chart to see details. These charts are interactive."
    )

    # ── Summary cards row ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card("Members Scored", f"{N_MEMBERS:,}", COLOURS["Dark"])

    with c2:
        metric_card("At High Risk", f"{N_HIGH_RISK:,}", COLOURS["High"])

    with c3:
        metric_card("Nudge Signals", f"{N_NUDGE:,}", COLOURS["Green"])

    with c4:
        metric_card("Nudge Rate", format_pct(NUDGE_RATE), COLOURS["Medium"])

    with c5:
        # Estimated savings at default 20% conversion, $800/ED
        true_positives = int(N_NUDGE * PRECISION_TOP20)
        est_savings = true_positives * 0.20 * 800
        metric_card(
            "Est. Savings",
            f"${est_savings:,.0f}",
            COLOURS["Green"],
            help_text="Estimated at default assumptions: 20% of nudged members book a visit, "
            "each avoided ED visit saves AUD $800. Adjust on the ROI Simulator page.",
        )

    st.markdown("---")

    # ── Chart row 1: Risk tier donut + Nudge rate by condition ─────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        tier_counts = (
            scored["risk_tier"]
            .value_counts()
            .reindex(["High", "Medium", "Low"])
        )
        fig_tier = px.pie(
            values=tier_counts.values,
            names=tier_counts.index,
            hole=0.55,
            color=tier_counts.index,
            color_discrete_map=COLOURS,
            title="Risk Tier Distribution",
        )
        fig_tier.update_traces(textposition="outside", textinfo="percent+label")
        fig_tier.update_layout(
            height=380,
            margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_tier, use_container_width=True)
        st.caption(
            "Only 10% of members are flagged as High risk. "
            "The model is designed to find the needle in the haystack — "
            "the members most likely to need emergency care."
        )

    with chart_col2:
        nudge_by_cluster = (
            scored.groupby("condition_cluster")["nudge_signal"]
            .agg(["sum", "count"])
            .assign(rate=lambda x: x["sum"] / x["count"])
            .sort_values("rate", ascending=False)
            .reset_index()
        )
        fig_cluster = px.bar(
            nudge_by_cluster,
            x="condition_cluster",
            y="rate",
            color="rate",
            color_continuous_scale=[COLOURS["Low"], COLOURS["High"]],
            title="Nudge Rate by Health Condition",
            labels={"rate": "Nudge Rate", "condition_cluster": "Health Condition"},
            text=nudge_by_cluster["rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig_cluster.update_traces(textposition="outside")
        fig_cluster.update_layout(
            height=380,
            margin=dict(t=40, b=0, l=0, r=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        st.caption(
            "The nudge rate is highest for members with multiple conditions (Mixed) "
            "and metabolic conditions. These are the groups where the model sees "
            "the biggest gap between having benefits and using them."
        )

    # ── Chart row 2: SHAP feature importance (full width) ─────────────────────
    st.markdown("### 🔑 What Drives the Risk Score?")

    if feat_imp is not None and len(feat_imp) > 0:
        top_features = feat_imp.sort_values("mean_abs_shap").tail(10)
        fig_shap = px.bar(
            top_features,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top 10 Factors That Influence Risk",
            labels={
                "mean_abs_shap": "Influence on Risk Score (higher = more important)",
                "feature": "",
            },
            color="mean_abs_shap",
            color_continuous_scale=[COLOURS["Low"], COLOURS["High"]],
        )
        fig_shap.update_layout(
            height=400,
            margin=dict(t=40, b=0, l=0, r=0),
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    st.caption(
        "These are the top 10 factors the model uses to estimate risk. "
        "'Influence' comes from a technique called SHAP (short for SHapley Additive exPlanations) "
        "which measures how much each piece of information contributes to the final risk score. "
        f"Notice that **allied health use rate is the #3 factor** — "
        "members who don't use their benefits are at higher risk, "
        "which is exactly why the nudge programme exists."
    )

    st.markdown("---")
    st.markdown("### 📖 What This Means")

    st.markdown(
        f"""
        The model looked at all **{N_MEMBERS:,} members** and found:

        - **{N_HIGH_RISK:,} members** ({format_pct(N_HIGH_RISK / N_MEMBERS)}) are at High risk
          of an emergency department visit in the next 6 months
        - **{N_NUDGE:,} members** ({format_pct(NUDGE_RATE)}) should receive a nudge —
          they are at Medium or High risk AND have unused benefits they haven't touched
        - The **#3 strongest predictor** of risk is whether a member uses their
          allied health benefits (physiotherapy, dietetics, psychology, chiropractic)

        This is the core insight: **coverage reduces claims**. Members who use their
        allied health benefits are less likely to end up in emergency care. The nudge
        programme connects at-risk members with the benefits they already have but aren't using.

        *Note: this is a proof of concept built on synthetic (computer-generated) data.
        Results on real claims data would be stronger because real data has more patterns
        for the model to learn from.*
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ROI Simulator
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💰 ROI Simulator":
    st.title("💰 ROI Simulator")
    st.caption(
        "Estimate the financial savings from running the nudge programme. "
        "Move the sliders to use your own assumptions — the numbers update live."
    )

    # ── Sliders for assumptions ───────────────────────────────────────────────
    slider_col1, slider_col2 = st.columns(2)

    with slider_col1:
        conversion_rate = st.slider(
            "What percentage of nudged members will actually book a visit?",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            format="%d%%",
            help="This is the 'conversion rate' — out of every 100 members who receive a nudge, "
            "how many will pick up the phone and book an appointment? "
            "Default is 20% (1 in 5), which is a conservative estimate for health outreach programmes.",
        )

    with slider_col2:
        ed_cost = st.slider(
            "How much does one avoided emergency department visit save?",
            min_value=400,
            max_value=2000,
            value=800,
            step=100,
            format="$%d",
            help="The average cost of an emergency department visit that could have been avoided "
            "with earlier allied health care. Default is AUD $800 — a conservative figure "
            "covering the ED presentation itself, not including hospital admission costs.",
        )

    # ── Calculation (same formula as HTML report and Power BI) ─────────────────
    conversion_decimal = conversion_rate / 100
    true_positives = int(N_NUDGE * PRECISION_TOP20)
    estimated_bookings = round(true_positives * conversion_decimal)
    estimated_savings = round(estimated_bookings * ed_cost)

    st.markdown("---")

    # ── Three live-updating results cards ──────────────────────────────────────
    res1, res2, res3 = st.columns(3)

    with res1:
        st.markdown(
            f"""
            <div style="background:white;border-radius:10px;padding:20px 24px;
                        text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07);
                        border-left:4px solid {COLOURS["Low"]};">
                <div style="font-size:30px;font-weight:700;color:{COLOURS["Dark"]};">{N_NUDGE:,}</div>
                <div style="font-size:13px;color:{COLOURS["Grey"]};">Members Nudged</div>
                <div style="font-size:11px;color:{COLOURS["Grey"]};margin-top:4px;">
                    Medium or High risk with unused benefits
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res2:
        st.markdown(
            f"""
            <div style="background:white;border-radius:10px;padding:20px 24px;
                        text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07);
                        border-left:4px solid {COLOURS["Medium"]};">
                <div style="font-size:30px;font-weight:700;color:{COLOURS["Dark"]};">{estimated_bookings:,}</div>
                <div style="font-size:13px;color:{COLOURS["Grey"]};">Estimated Bookings</div>
                <div style="font-size:11px;color:{COLOURS["Grey"]};margin-top:4px;">
                    {format_pct(PRECISION_TOP20)} of nudged members × {conversion_rate}% conversion
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res3:
        st.markdown(
            f"""
            <div style="background:white;border-radius:10px;padding:20px 24px;
                        text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07);
                        border-left:4px solid {COLOURS["Green"]};">
                <div style="font-size:30px;font-weight:700;color:{COLOURS["Green"]};">${estimated_savings:,}</div>
                <div style="font-size:13px;color:{COLOURS["Grey"]};">Estimated Savings</div>
                <div style="font-size:11px;color:{COLOURS["Grey"]};margin-top:4px;">
                    {estimated_bookings:,} bookings × ${ed_cost} per avoided ED visit
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Sensitivity table ─────────────────────────────────────────────────────
    st.markdown("### 📋 Sensitivity Table")
    st.caption(
        "Savings at different combinations of conversion rate and ED cost. "
        "This lets you see the full range of possible outcomes at a glance."
    )

    conv_rates = [10, 15, 20, 25, 30]
    ed_costs = [600, 800, 1000, 1200]

    # Build the sensitivity matrix
    matrix_data = []
    for cr in conv_rates:
        row = {"Conversion Rate": f"{cr}%"}
        for ec in ed_costs:
            bookings = round(true_positives * (cr / 100))
            savings = round(bookings * ec)
            row[f"${ec:,}"] = f"${savings:,}"
        matrix_data.append(row)

    sensitivity_df = pd.DataFrame(matrix_data)
    st.dataframe(
        sensitivity_df.set_index("Conversion Rate"),
        use_container_width=True,
    )

    st.caption(
        f"Based on {N_NUDGE:,} nudge candidates × {format_pct(PRECISION_TOP20)} "
        "precision (the percentage of nudged members who are genuinely at risk). "
        "Each cell shows estimated savings = members nudged × precision × conversion rate × ED cost."
    )

    # ── Plain English explanation ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📖 What This Means")

    # Find the most conservative scenario
    min_savings = round(true_positives * 0.10 * 600)
    max_savings = round(true_positives * 0.30 * 1200)

    st.markdown(
        f"""
        Here's what these numbers tell us, in plain English:

        - **Members available to nudge:** {N_NUDGE:,} people are at elevated risk AND have
          unused allied health benefits they haven't touched this year.

        - **How many are genuinely at risk:** Not every nudged member will actually visit
          the ED. The model estimates about **{format_pct(PRECISION_TOP20)}** of them are
          true high-risk cases (about {true_positives:,} people) — this is based on how
          well the model performed on data it hadn't seen before.

        - **The range of possible savings:** Even at the most conservative estimate
          (10% of nudged members act on it, each avoided visit saves $600),
          the programme saves **${min_savings:,}**. At the more realistic 20% conversion
          and $800 per visit, it saves **${estimated_savings:,}**. At the upper end
          (30% conversion, $1,200 per visit), savings reach **${max_savings:,}**.

        - **This is a conservative estimate.** It only counts the direct cost of one
          avoided emergency department visit. In reality, preventing an ED visit often
          prevents a hospital admission, specialist referrals, and ongoing chronic
          disease management — costs that are 2 to 4 times higher.

        **The bottom line:** Nudging members to use allied health benefits they already have
        is a low-cost, high-return programme. The benefits are paid for whether members
        use them or not. The nudge just makes sure they get value from their cover —
        and the health fund avoids the much larger cost of an unmanaged condition
        escalating to emergency care.
        """
    )

    st.info(
        "💡 **Try it yourself:** Move the sliders above to match your own assumptions. "
        "The numbers update instantly. The sensitivity table shows every combination "
        "so you can find the scenario that matches your experience."
    )
