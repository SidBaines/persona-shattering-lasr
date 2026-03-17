# Calibration run — agreeableness — 20260317_183014

**Models:** openai/gpt-4o-mini, anthropic/claude-3-5-haiku  
**Provider:** openrouter  
**Temperature:** 0.9  
**Items:** 28  

```

Running openai/gpt-4o-mini (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=openai/gpt-4o-mini  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.910  (n=28)
  Spearman r : +0.834
  MAE        : 0.71
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  a_01          clear_high                     +4      +4      +0
  a_02          clear_high                     +4      +3      -1
  a_03          clear_high                     +4      +4      +0
  a_04          clear_high                     +3      +3      +0
  a_05          clear_high                     +3      +3      +0
  a_06          moderate_high                  +2      +3      +1
  a_07          moderate_high                  +2      +2      +0
  a_08          moderate_high                  +2      +3      +1
  a_09          slight_high                    +1      +3      +2
  a_10          slight_high                    +1      -2      -3
  a_11          neutral                        +0      +0      +0
  a_12          neutral                        +0      +0      +0
  a_13          neutral                        +0      +0      +0
  a_14          neutral                        +0      +0      +0
  a_15          slight_low                     -1      -2      -1
  a_16          slight_low                     -1      -3      -2
  a_17          moderate_low                   -2      -2      +0
  a_18          moderate_low                   -2      -3      -1
  a_19          moderate_low                   -2      -2      +0
  a_20          clear_low                      -3      -3      +0
  a_21          clear_low                      -3      -2      +1
  a_22          clear_low                      -3      -3      +0
  a_23          clear_low                      -3      -4      -1
  a_24          clear_low                      -4      -3      +1
  a_25          slight_low                     -1      -2      -1
  a_26          confound_expert_directness     +0      -2      -2  ← confound
  a_27          confound_honest_feedback       +0      -2      -2  ← confound
  a_28          moderate_low                   -2      -2      +0
==========================================================================================

Running anthropic/claude-3-5-haiku (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=anthropic/claude-3-5-haiku  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.933  (n=28)
  Spearman r : +0.897
  MAE        : 0.57
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  a_01          clear_high                     +4      +3      -1
  a_02          clear_high                     +4      +3      -1
  a_03          clear_high                     +4      +3      -1
  a_04          clear_high                     +3      +3      +0
  a_05          clear_high                     +3      +3      +0
  a_06          moderate_high                  +2      +3      +1
  a_07          moderate_high                  +2      +2      +0
  a_08          moderate_high                  +2      +3      +1
  a_09          slight_high                    +1      +2      +1
  a_10          slight_high                    +1      +2      +1
  a_11          neutral                        +0      +0      +0
  a_12          neutral                        +0      +0      +0
  a_13          neutral                        +0      +0      +0
  a_14          neutral                        +0      +0      +0
  a_15          slight_low                     -1      -1      +0
  a_16          slight_low                     -1      -2      -1
  a_17          moderate_low                   -2      -2      +0
  a_18          moderate_low                   -2      -2      +0
  a_19          moderate_low                   -2      -1      +1
  a_20          clear_low                      -3      -3      +0
  a_21          clear_low                      -3      -2      +1
  a_22          clear_low                      -3      -3      +0
  a_23          clear_low                      -3      -3      +0
  a_24          clear_low                      -4      -3      +1
  a_25          slight_low                     -1      -2      -1
  a_26          confound_expert_directness     +0      -2      -2  ← confound
  a_27          confound_honest_feedback       +0      -2      -2  ← confound
  a_28          moderate_low                   -2      -2      +0
==========================================================================================

==========================================================================================
CROSS-MODEL SUMMARY  (reference: author (expected_score))
==========================================================================================
  model                                      pearson   spearman     mae   errors
  ------------------------------------------------------------------------------
  openai/gpt-4o-mini                          +0.910     +0.834    0.71      0/28
  anthropic/claude-3-5-haiku                  +0.933     +0.897    0.57      0/28

  Pairwise MAE between models:
    openai/gpt-4o-mini vs anthropic/claude-3-5-haiku: MAE=0.43 (n=28)
==========================================================================================

Consistency check: 3 runs at temp=0.9 [openai/gpt-4o-mini] ...

==========================================================================================
SELF-CONSISTENCY  (temperature=0.9)
==========================================================================================
  id            expected    mean    std   min   max  scores
  ----------------------------------------------------------------------------------
  a_01                +4   +4.00   0.00    +4    +4  +4  +4  +4
  a_02                +4   +3.00   0.00    +3    +3  +3  +3  +3
  a_03                +4   +3.33   0.58    +3    +4  +3  +3  +4
  a_04                +3   +3.33   0.58    +3    +4  +3  +3  +4
  a_05                +3   +3.00   0.00    +3    +3  +3  +3  +3
  a_06                +2   +3.00   0.00    +3    +3  +3  +3  +3
  a_07                +2   +2.00   0.00    +2    +2  +2  +2  +2
  a_08                +2   +3.00   0.00    +3    +3  +3  +3  +3
  a_09                +1   +3.00   0.00    +3    +3  +3  +3  +3
  a_10                +1   -1.33   0.58    -2    -1  -1  -2  -1
  a_11                +0   +0.00   0.00    +0    +0  +0  +0  +0
  a_12                +0   +0.00   0.00    +0    +0  +0  +0  +0
  a_13                +0   +0.00   0.00    +0    +0  +0  +0  +0
  a_14                +0   +0.00   0.00    +0    +0  +0  +0  +0
  a_15                -1   -2.00   0.00    -2    -2  -2  -2  -2
  a_16                -1   -3.00   0.00    -3    -3  -3  -3  -3
  a_17                -2   -2.00   0.00    -2    -2  -2  -2  -2
  a_18                -2   -3.00   0.00    -3    -3  -3  -3  -3
  a_19                -2   -2.00   0.00    -2    -2  -2  -2  -2
  a_20                -3   -3.33   0.58    -4    -3  -3  -3  -4
  a_21                -3   -2.00   0.00    -2    -2  -2  -2  -2
  a_22                -3   -2.67   0.58    -3    -2  -2  -3  -3
  a_23                -3   -3.33   0.58    -4    -3  -4  -3  -3
  a_24                -4   -3.33   0.58    -4    -3  -3  -4  -3
  a_25                -1   -2.33   0.58    -3    -2  -3  -2  -2
  a_26                +0   -2.33   0.58    -3    -2  -2  -3  -2
  a_27                +0   -2.00   0.00    -2    -2  -2  -2  -2
  a_28                -2   -2.00   0.00    -2    -2  -2  -2  -2

  Overall mean std: 0.186
==========================================================================================

==========================================================================================
SCORECARD
==========================================================================================

  Model: openai/gpt-4o-mini
  PASS  Pearson r vs reference                    +0.910  (threshold ≥ 0.90)
  FAIL  Spearman r vs reference                   +0.834  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.71  (threshold ≤ 1.00)
  FAIL  Confound accuracy (score=0)                  0/2  (threshold = 100%)

  Model: anthropic/claude-3-5-haiku
  PASS  Pearson r vs reference                    +0.933  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.897  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.57  (threshold ≤ 1.00)
  FAIL  Confound accuracy (score=0)                  0/2  (threshold = 100%)

  Consistency (temp=0.9, 3 runs, model=openai/gpt-4o-mini):
  PASS  Mean std across items                      0.186  (threshold ≤ 0.50 mean)

  Inter-model agreement:
  PASS  openai/gpt-4o-mini vs anthropic/claude-3-5      0.43  (threshold ≤ 1.00)
==========================================================================================

```
