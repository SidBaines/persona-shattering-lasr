"""Frustration evaluation setup.

Reproduces the frustration elicitation protocol from:
  "Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"
  (Soligo, Mikulik, Saunders, 2026)

Multi-turn conversations present tasks then reject model responses, measuring
expressed frustration on a 0-10 scale via an LLM judge.
"""
