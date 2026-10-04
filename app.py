"""Semester at the study office: a Streamlit app that helps the study office decide who to talk to in week 6.

Built from the hotel app "Tonight's front desk" (aaubs/tonights-front-desk). The model sits next to the code, in model/:
booster.json (XGBoost's trees) + preprocess.json (scaling and one-hot, as plain numbers), written by step 9 of our
notebook and loaded by portable.py (unchanged from the hotel app). The student data is read straight from GitHub.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from portable import Model

# ---------------------------------------------------------------- settings
MODEL_DIR = Path(__file__).parent / "model"      # the folder next to this file
URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"

# Plain-language labels for the "why is this student on the list" column (same idea as LABEL in the hotel app).
# {v} is replaced by the student's value. For 0/1 columns we show "yes" or "no" instead of 1 or 0.
LABEL = {"age": "age {v}", "admission_grade": "admission grade {v}", "international": "international: {v}",
         "first_gen": "first in family at university: {v}", "su_scholarship": "SU grant: {v}",
         "fees_owed": "owes fees: {v}", "moved_from_home": "moved from home: {v}", "married": "married: {v}",
         "evening_programme": "evening programme: {v}", "programme": "programme: {v}", "gender": "gender: {v}",
         "logins_total": "{v} logins in weeks 1-6", "logins_last3": "{v} logins in the last 3 weeks",
         "logins_trend": "login trend {v}", "submitted_share": "handed in {v} of assignments",
         "missed_last3": "{v} assignments missed lately", "quiz_mean": "quiz average {v}",
         "weeks_since_login": "{v} weeks since last login"}
YES_NO = ["international", "first_gen", "su_scholarship", "fees_owed", "moved_from_home", "married", "evening_programme"]

st.set_page_config(page_title="Semester at the study office", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- helper functions
def nice(feature, value):
    """Turn a raw value into readable text: 1/0 -> yes/no, shares -> %, other numbers rounded."""
    if feature in YES_NO:
        return "yes" if value == 1 else "no"
    if feature == "submitted_share":
        return f"{value:.0%}"
    if isinstance(value, float):
        return f"{value:.1f}" if value != int(value) else f"{int(value)}"
    return value


def reasons(pipe, features, rows, top=2):
    """Copied from the hotel app: the model's contributions per feature (SHAP values that XGBoost computes itself).
    For each student we keep the features that push the risk up the most, and write them in plain words."""
    by_f = pipe.contributions(rows[features]).reset_index(drop=True)
    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        best = by_f.iloc[i].sort_values(ascending=False).head(top)     # the biggest pushes upwards
        out.append(" · ".join(f"{LABEL[f].format(v=nice(f, row[f]))} ↑" for f, c in best.items() if c > 0))
    return out


def rule_numbers(df, k):
    """The four boxes for the rule 'talk to the k highest risks' on a group of students who already know the outcome."""
    contacted = df["risk"].rank(ascending=False, method="first") <= k    # True for the k riskiest students
    left = df["left"] == 1
    tp, fp = int((contacted & left).sum()), int((contacted & ~left).sum())
    fn, tn = int((~contacted & left).sum()), int((~contacted & ~left).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


# ---------------------------------------------------------------- the model and the data, loaded once per server
@st.cache_resource          # runs once, not on every click (as in the hotel app)
def load():
    pipe = Model(MODEL_DIR)                           # no pickle: loads with any recent pandas/xgboost
    features = pipe.features                          # the 18 columns the model was trained on
    history = pd.read_csv(URL + "history_week6.csv")
    new = pd.read_csv(URL + "new_week6.csv")
    val = history[history["cohort"] == 2025].copy()   # last year's students: we know who left, so we can count mistakes
    val["risk"] = pipe.predict_proba(val[features])
    new["risk"] = pipe.predict_proba(new[features])
    new = new.sort_values("risk", ascending=False).reset_index(drop=True)
    new["why"] = reasons(pipe, features, new)         # our own addition: why each student is on the list
    return features, val, new


features, val, new = load()

# ---------------------------------------------------------------- sidebar: the rule
with st.sidebar:
    st.header("📞 The office's rule")
    k = st.slider("Conversations this week", 10, 200, 40, 5,
                  help="Three advisers can hold about 40 conversations. The rule: talk to the students with the highest risk.")
    st.caption("Model: XGBoost, trained on the 2023 and 2024 cohorts, checked on 2025. "
               "It ranks students by risk; an adviser decides who to contact.")

st.title("🎓 Semester at the study office")
tab_list, tab_rule, tab_group, tab_about = st.tabs(
    ["📋 This week's list", "⚖️ Mistakes of the rule", "🌍 Per group", "ℹ️ How it works"])

# ---------------------------------------------------------------- 1. this week's list (2026 students)
with tab_list:
    st.markdown(f"The **{len(new)} students of 2026** at the end of week 6, ranked by the model's risk that they leave "
                f"later this semester. The **top {k}** are marked.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Students this week", len(new))
    c2.metric("Conversations", k)
    c3.metric("Expected leavers among them", f"{new['risk'].head(k).sum():.0f}",
              help="The sum of the risks. On 2025 the model over-estimated a little, so treat this as a rough guide.")
    show = new.copy()
    show["talk"] = show.index < k                     # True for the top k (the list is already sorted by risk)
    show["risk"] = (100 * show["risk"]).round()
    st.dataframe(show[["talk", "student_id", "risk", "programme", "why"]],
                 column_config={"talk": st.column_config.CheckboxColumn("talk to"),
                                "student_id": "student",
                                "risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%d%%"),
                                "why": "why on the list (biggest pushes up)"},
                 hide_index=True, width="stretch", height=500)
    st.caption("The 'why' column shows what pushes the model's risk up for that student: patterns in past data, not "
               "causes. Use it to choose the right kind of help (money, study group, a check-in).")

# ---------------------------------------------------------------- 2. the mistakes of the rule (2025 students)
with tab_rule:
    b = rule_numbers(val, k)
    st.markdown(f"What would this rule have done **last year**? On the **{len(val)} students of 2025**, "
                f"of whom **{int(val['left'].sum())} left**, talking to the **{k}** highest risks gives:")
    st.success(f"🎯 **{b['TP']} students reached in time** · 📞 **{b['FP']} worried for nothing** · "
               f"🚪 **{b['FN']} missed** · ✅ {b['TN']} correctly left alone")
    c1, c2 = st.columns(2)
    c1.metric("Precision", f"{b['precision']:.0%}", help="Of the students we talked to, the share who really left.")
    c2.metric("Recall", f"{b['recall']:.0%}", help="Of the students who left, the share we talked to.")
    st.markdown("**The four boxes**")
    st.dataframe(pd.DataFrame([[b["TN"], b["FP"]], [b["FN"], b["TP"]]],
                              index=["stayed", "left"], columns=["not contacted", "contacted"]))
    st.caption(f"For comparison: {k} students picked at random would include about {k * val['left'].mean():.0f} leavers.")

# ---------------------------------------------------------------- 3. per group: international vs domestic
with tab_group:
    st.markdown(f"The same rule (top {k} on 2025), split into **domestic** and **international** students.")
    rank = val["risk"].rank(ascending=False, method="first") <= k
    rows = []
    for name, flag in [("domestic", 0), ("international", 1)]:
        g = val[val["international"] == flag]
        contacted, left = rank[g.index], g["left"] == 1
        tp, fp, fn = int((contacted & left).sum()), int((contacted & ~left).sum()), int((~contacted & left).sum())
        rows.append({"group": name, "students": len(g), "left": int(left.sum()),
                     "reached in time": tp, "worried for nothing": fp, "missed": fn,
                     "precision": f"{tp / max(1, tp + fp):.0%}", "recall": f"{tp / max(1, tp + fn):.0%}",
                     "share who left": f"{left.mean():.1%}", "average predicted risk": f"{g['risk'].mean():.1%}"})
    st.dataframe(pd.DataFrame(rows).set_index("group"), width="stretch")
    st.warning("Only about 14 international students left in 2025, so one or two students more or less change these "
               "numbers a lot. International students log in about a third less, yet leave no more often: a low login "
               "count is not the same warning sign for everyone. Check the mistakes per group before trusting the list.")

# ---------------------------------------------------------------- 4. how it works
with tab_about:
    st.markdown("""
### How this app works
- **The model** (XGBoost) gives each student a risk: the probability of leaving later this semester. It uses only what
  the office knows at the end of week 6: enrolment data, money and life situation, logins, assignments and quizzes.
  Columns filled in later (ECTS passed, the deregistration form, the last login of the semester) are left out.
- **Training and checking:** trained on the 2023 and 2024 cohorts, checked on 2025 (AUC 0.78).
- **The rule:** talk to the students with the highest risk, as many as the advisers have time for.
- **The decision stays with people.** The list only *offers* help. An adviser decides who to contact, students should be
  told the list exists, and they can contest it. Under the EU AI Act, systems that evaluate or steer students are high-risk.

Built from the hotel app *Tonight's front desk* (AAU Business Data Science, session 10). The students are synthetic,
resampled from the UCI "Predict students' dropout and academic success" data (CC BY 4.0).
""")
