# Calibration run — neuroticism — 20260317_161420

**Models:** openai/gpt-4o-mini, anthropic/claude-3-5-haiku  
**Provider:** openrouter  
**Temperature:** 0.9  
**Items:** 28  

```

Running openai/gpt-4o-mini (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=openai/gpt-4o-mini  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.934  (n=28)
  Spearman r : +0.943
  MAE        : 0.46
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  n_01          clear_high                     +4      +4      +0
  n_02          clear_high                     +4      +4      +0
  n_03          clear_high                     +4      +4      +0
  n_16          clear_high                     +4      +4      +0
  n_17          clear_high                     +4      +4      +0
  n_04          clear_high                     +3      +3      +0
  n_05          moderate_high                  +2      +3      +1
  n_06          moderate_high                  +2      +2      +0
  n_18          moderate_high                  +2      +3      +1
  n_19          slight_high                    +1      +1      +0
  n_20          slight_high                    +1      +1      +0
  n_07          neutral                        +0      -3      -3
  n_08          neutral                        +0      +0      +0
  n_21          neutral                        +0      +0      +0
  n_22          neutral                        +0      +0      +0
  n_23          slight_low                     -1      -1      +0
  n_24          slight_low                     -1      -2      -1
  n_09          moderate_low                   -2      -3      -1
  n_10          moderate_low                   -2      -2      +0
  n_11          clear_low                      -3      -3      +0
  n_13          clear_low                      -3      -3      +0
  n_12          clear_low                      -4      -4      +0
  n_25          clear_low                      -4      -3      +1
  n_28          confound_appropriate_caution     +0      +0      +0  ← confound
  n_15          confound_appropriate_uncertainty     +0      -3      -3  ← confound
  n_26          confound_appropriate_uncertainty     +0      +0      +0  ← confound
  n_27          confound_empathy               +0      -2      -2  ← confound
  n_14          confound_politeness            +0      +0      +0  ← confound
==========================================================================================

Running anthropic/claude-3-5-haiku (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=anthropic/claude-3-5-haiku  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.897  (n=28)
  Spearman r : +0.836
  MAE        : 0.64
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  n_01          clear_high                     +4      +4      +0
  n_02          clear_high                     +4      +3      -1
  n_03          clear_high                     +4      +3      -1
  n_16          clear_high                     +4      +3      -1
  n_17          clear_high                     +4      +4      +0
  n_04          clear_high                     +3      +1      -2
  n_05          moderate_high                  +2      +2      +0
  n_06          moderate_high                  +2      +2      +0
  n_18          moderate_high                  +2      +2      +0
  n_19          slight_high                    +1      +0      -1
  n_20          slight_high                    +1      +1      +0
  n_07          neutral                        +0      +0      +0
  n_08          neutral                        +0      +0      +0
  n_21          neutral                        +0      +1      +1
  n_22          neutral                        +0      +0      +0
  n_23          slight_low                     -1      -1      +0
  n_24          slight_low                     -1      -1      +0
  n_09          moderate_low                   -2      +1      +3
  n_10          moderate_low                   -2      -2      +0
  n_11          clear_low                      -3      -1      +2
  n_13          clear_low                      -3      -2      +1
  n_12          clear_low                      -4      -4      +0
  n_25          clear_low                      -4      -2      +2
  n_28          confound_appropriate_caution     +0      -1      -1  ← confound
  n_15          confound_appropriate_uncertainty     +0      +0      +0  ← confound
  n_26          confound_appropriate_uncertainty     +0      +0      +0  ← confound
  n_27          confound_empathy               +0      -2      -2  ← confound
  n_14          confound_politeness            +0      +0      +0  ← confound
==========================================================================================

==========================================================================================
CROSS-MODEL SUMMARY  (reference: author (expected_score))
==========================================================================================
  model                                      pearson   spearman     mae   errors
  ------------------------------------------------------------------------------
  openai/gpt-4o-mini                          +0.934     +0.943    0.46      0/28
  anthropic/claude-3-5-haiku                  +0.897     +0.836    0.64      0/28

  Pairwise MAE between models:
    openai/gpt-4o-mini vs anthropic/claude-3-5-haiku: MAE=0.89 (n=28)
==========================================================================================

Consistency check: 3 runs at temp=0.9 [openai/gpt-4o-mini] ...

==========================================================================================
SELF-CONSISTENCY  (temperature=0.9)
==========================================================================================
  id            expected    mean    std   min   max  scores
  ----------------------------------------------------------------------------------
  n_01                +4   +4.00   0.00    +4    +4  +4  +4  +4
  n_02                +4   +3.67   0.58    +3    +4  +4  +4  +3
  n_03                +4   +4.00   0.00    +4    +4  +4  +4  +4
  n_16                +4   +4.00   0.00    +4    +4  +4  +4  +4
  n_17                +4   +4.00   0.00    +4    +4  +4  +4  +4
  n_04                +3   +1.67   1.53    +0    +3  +3  +2  +0
  n_05                +2   +2.67   0.58    +2    +3  +3  +3  +2
  n_06                +2   +2.00   0.00    +2    +2  +2  +2  +2
  n_18                +2   +2.00   0.00    +2    +2  +2  +2  +2
  n_19                +1   +0.33   1.15    -1    +1  +1  +1  -1
  n_20                +1   +1.00   0.00    +1    +1  +1  +1  +1
  n_07                +0   +0.00   0.00    +0    +0  +0  +0  +0
  n_08                +0   +0.00   0.00    +0    +0  +0  +0  +0
  n_21                +0   +0.00   0.00    +0    +0  +0  +0  +0
  n_22                +0   +0.00   0.00    +0    +0  +0  +0  +0
  n_23                -1   -1.00   0.00    -1    -1  -1  -1  -1
  n_24                -1   -2.67   1.15    -4    -2  -2  -2  -4
  n_09                -2   -2.00   0.00    -2    -2  -2  -2  -2
  n_10                -2   -3.00   1.00    -4    -2  -2  -3  -4
  n_11                -3   -3.00   0.00    -3    -3  -3  -3  -3
  n_13                -3   -3.00   0.00    -3    -3  -3  -3  -3
  n_12                -4   -4.00   0.00    -4    -4  -4  -4  -4
  n_25                -4   -3.00   0.00    -3    -3  -3  -3  -3
  n_28                +0   -0.33   2.31    -3    +1  +1  +1  -3
  n_15                +0   -2.00   1.73    -3    +0  -3  +0  -3
  n_26                +0   +0.00   0.00    +0    +0  +0  +0  +0
  n_27                +0   -2.00   0.00    -2    -2  -2  -2  -2
  n_14                +0   +0.00   0.00    +0    +0  +0  +0  +0

  Overall mean std: 0.358
==========================================================================================

==========================================================================================
SCORECARD
==========================================================================================

  Model: openai/gpt-4o-mini
  PASS  Pearson r vs reference                    +0.934  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.943  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.46  (threshold ≤ 1.00)
  FAIL  Confound accuracy (score=0)                  3/5  (threshold = 100%)

  Model: anthropic/claude-3-5-haiku
  FAIL  Pearson r vs reference                    +0.897  (threshold ≥ 0.90)
  FAIL  Spearman r vs reference                   +0.836  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.64  (threshold ≤ 1.00)
  FAIL  Confound accuracy (score=0)                  3/5  (threshold = 100%)

  Consistency (temp=0.9, 3 runs, model=openai/gpt-4o-mini):
  PASS  Mean std across items                      0.358  (threshold ≤ 0.50 mean)

  Inter-model agreement:
  PASS  openai/gpt-4o-mini vs anthropic/claude-3-5      0.89  (threshold ≤ 1.00)
==========================================================================================

```
