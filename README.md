---
title: AI Resume Matcher
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# AI Resume Matcher

A Streamlit app to match resumes with job descriptions using AI.

## Features
- Resume parsing (PDF)
- Keyword extraction using KeyBERT
- Semantic similarity matching using Sentence Transformers
- Entity extraction (Skills, Organizations, Degrees)

## How to run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
