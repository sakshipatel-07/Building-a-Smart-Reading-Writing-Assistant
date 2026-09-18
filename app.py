"""
Stretch Goal (+10 pts) -- live demo front end
================================================
Turns Stage 1's spell checker into a tiny interactive "writing assistant"
using Streamlit, so the "real world" framing (a writing assistant) is
visibly usable instead of just a script.

Run with:
    streamlit run app.py
"""
import streamlit as st
from stage1_spellcheck.spellcheck import correct_text, SpellChecker

st.set_page_config(page_title="Smart Writing Assistant -- Spell Check Demo", page_icon="📝")

st.title("📝 Smart Reading & Writing Assistant")
st.caption("Stage 1 demo: from-scratch edit-distance spell checker with context-aware (n-gram) correction")

if "checker" not in st.session_state:
    with st.spinner("Loading dictionary & language model..."):
        st.session_state.checker = SpellChecker()

default_text = ("Hi Professor, I recieve you're email about the assignment. "
                 "I think there going to extend the deadline becuase alot of "
                 "students are confused about the instructions.")

text = st.text_area("Type or paste text here:", value=default_text, height=150)

if text.strip():
    corrected, diff = correct_text(text, st.session_state.checker)

    st.subheader("Corrected text")
    st.write(corrected)

    st.subheader(f"Changes made ({len(diff)})")
    if not diff:
        st.info("No corrections needed.")
    for c in diff:
        reason = c["reason"]
        with st.expander(f"'{c['original']}' → '{c['corrected']}'  ({reason['type']})"):
            st.json(reason)
else:
    st.info("Start typing above to see live corrections.")

st.divider()
st.caption("This is Stage 1 of a 5-stage NLP pipeline "
           "(Spell Checking → Syntax → Semantics → Discourse/Pragmatics → Full Assembly). "
           "See pipeline.py for the full end-to-end system.")
