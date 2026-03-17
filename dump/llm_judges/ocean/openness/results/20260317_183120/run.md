# Calibration run — openness — 20260317_183120

**Models:** openai/gpt-4o-mini, anthropic/claude-3-5-haiku  
**Provider:** openrouter  
**Temperature:** 0.9  
**Items:** 28  

```

Running openai/gpt-4o-mini (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=openai/gpt-4o-mini  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.899  (n=28)
  Spearman r : +0.887
  MAE        : 0.93
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  o_01          clear_high                     +4      +4      +0
  o_02          clear_high                     +4      +3      -1
  o_03          clear_high                     +4      +4      +0
  o_04          clear_high                     +3      +3      +0
  o_05          clear_high                     +3      +3      +0
  o_06          moderate_high                  +2      +3      +1
  o_07          moderate_high                  +2      +4      +2
  o_08          moderate_high                  +2      +3      +1
  o_09          slight_high                    +1      +3      +2
  o_10          slight_high                    +1      +2      +1
  o_11          neutral                        +0      +0      +0
  o_12          neutral                        +0      -2      -2
  o_13          neutral                        +0      +0      +0
  o_14          neutral                        +0      -2      -2
  o_15          slight_low                     -1      -2      -1
  o_16          slight_low                     -1      -2      -1
  o_17          moderate_low                   -2      -3      -1
  o_18          moderate_low                   -2      -3      -1
  o_19          moderate_low                   -2      -2      +0
  o_20          clear_low                      -3      -3      +0
  o_21          clear_low                      -3      -3      +0
  o_22          clear_low                      -4      -3      +1
  o_23          confound_abstract_topic_not_abstract_responder     +0      +0      +0  ← confound
  o_24          confound_correct_conventional_answer     +0      -2      -2  ← confound
  o_25          slight_high                    +1      +3      +2
  o_26          slight_low                     -1      -3      -2
  o_27          neutral                        +0      -2      -2
  o_28          moderate_high                  +2      +3      +1
==========================================================================================

Running anthropic/claude-3-5-haiku (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=anthropic/claude-3-5-haiku  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.943  (n=28)
  Spearman r : +0.939
  MAE        : 0.57
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  o_01          clear_high                     +4      +4      +0
  o_02          clear_high                     +4      +4      +0
  o_03          clear_high                     +4      +3      -1
  o_04          clear_high                     +3      +3      +0
  o_05          clear_high                     +3      +3      +0
  o_06          moderate_high                  +2      +2      +0
  o_07          moderate_high                  +2      +3      +1
  o_08          moderate_high                  +2      +3      +1
  o_09          slight_high                    +1      +2      +1
  o_10          slight_high                    +1      +2      +1
  o_11          neutral                        +0      +1      +1
  o_12          neutral                        +0      +0      +0
  o_13          neutral                        +0      +1      +1
  o_14          neutral                        +0      +0      +0
  o_15          slight_low                     -1      -2      -1
  o_16          slight_low                     -1      -2      -1
  o_17          moderate_low                   -2      -2      +0
  o_18          moderate_low                   -2      -2      +0
  o_19          moderate_low                   -2      -2      +0
  o_20          clear_low                      -3      -3      +0
  o_21          clear_low                      -3      -3      +0
  o_22          clear_low                      -4      -3      +1
  o_23          confound_abstract_topic_not_abstract_responder     +0      +2      +2  ← confound
  o_24          confound_correct_conventional_answer     +0      -1      -1  ← confound
  o_25          slight_high                    +1      +2      +1
  o_26          slight_low                     -1      -2      -1
  o_27          neutral                        +0      +0      +0
  o_28          moderate_high                  +2      +3      +1
==========================================================================================

==========================================================================================
CROSS-MODEL SUMMARY  (reference: author (expected_score))
==========================================================================================
  model                                      pearson   spearman     mae   errors
  ------------------------------------------------------------------------------
  openai/gpt-4o-mini                          +0.899     +0.887    0.93      0/28
  anthropic/claude-3-5-haiku                  +0.943     +0.939    0.57      0/28

  Pairwise MAE between models:
    openai/gpt-4o-mini vs anthropic/claude-3-5-haiku: MAE=0.71 (n=28)
==========================================================================================

Consistency check: 3 runs at temp=0.9 [openai/gpt-4o-mini] ...

==========================================================================================
SELF-CONSISTENCY  (temperature=0.9)
==========================================================================================
  id            expected    mean    std   min   max  scores
  ----------------------------------------------------------------------------------
  o_01                +4   +4.00   0.00    +4    +4  +4  +4  +4
  o_02                +4   +3.33   0.58    +3    +4  +4  +3  +3
  o_03                +4   +3.67   0.58    +3    +4  +3  +4  +4
  o_04                +3   +3.00   0.00    +3    +3  +3  +3  +3
  o_05                +3   +3.00   0.00    +3    +3  +3  +3  +3
  o_06                +2   +3.00   0.00    +3    +3  +3  +3  +3
  o_07                +2   +4.00   0.00    +4    +4  +4  +4  +4
  o_08                +2   +3.00   0.00    +3    +3  +3  +3  +3
  o_09                +1   +3.00   0.00    +3    +3  +3  +3  +3
  o_10                +1   +2.33   0.58    +2    +3  +2  +3  +2
  o_11                +0   -0.67   1.15    -2    +0  +0  -2  +0
  o_12                +0   -2.00   0.00    -2    -2  -2  -2  -2
  o_13                +0   -0.67   1.15    -2    +0  +0  +0  -2
  o_14                +0   -1.33   1.15    -2    +0  +0  -2  -2
  o_15                -1   -2.33   0.58    -3    -2  -2  -2  -3
  o_16                -1   -2.00   0.00    -2    -2  -2  -2  -2
  o_17                -2   -3.00   0.00    -3    -3  -3  -3  -3
  o_18                -2   -3.00   0.00    -3    -3  -3  -3  -3
  o_19                -2   -2.00   0.00    -2    -2  -2  -2  -2
  o_20                -3   -3.00   0.00    -3    -3  -3  -3  -3
  o_21                -3   -3.00   0.00    -3    -3  -3  -3  -3
  o_22                -4   -3.00   0.00    -3    -3  -3  -3  -3
  o_23                +0   -1.67   0.58    -2    -1  -2  -1  -2
  o_24                +0   -2.00   0.00    -2    -2  -2  -2  -2
  o_25                +1   +3.00   0.00    +3    +3  +3  +3  +3
  o_26                -1   -3.00   0.00    -3    -3  -3  -3  -3
  o_27                +0   -2.00   0.00    -2    -2  -2  -2  -2
  o_28                +2   +3.00   0.00    +3    +3  +3  +3  +3

  Overall mean std: 0.227
==========================================================================================

==========================================================================================
SCORECARD
==========================================================================================

  Model: openai/gpt-4o-mini
  FAIL  Pearson r vs reference                    +0.899  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.887  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.93  (threshold ≤ 1.00)
  FAIL  Confound accuracy (score=0)                  1/2  (threshold = 100%)

  Model: anthropic/claude-3-5-haiku
  PASS  Pearson r vs reference                    +0.943  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.939  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.57  (threshold ≤ 1.00)
  FAIL  Confound accuracy (score=0)                  0/2  (threshold = 100%)

  Consistency (temp=0.9, 3 runs, model=openai/gpt-4o-mini):
  PASS  Mean std across items                      0.227  (threshold ≤ 0.50 mean)

  Inter-model agreement:
  PASS  openai/gpt-4o-mini vs anthropic/claude-3-5      0.71  (threshold ≤ 1.00)
==========================================================================================

```
