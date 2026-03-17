# Calibration run — extraversion — 20260317_183057

**Models:** openai/gpt-4o-mini, anthropic/claude-3-5-haiku  
**Provider:** openrouter  
**Temperature:** 0.9  
**Items:** 28  

```

Running openai/gpt-4o-mini (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=openai/gpt-4o-mini  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.943  (n=28)
  Spearman r : +0.934
  MAE        : 0.71
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  e_01          clear_high                     +4      +4      +0
  e_02          clear_high                     +4      +4      +0
  e_03          clear_high                     +4      +4      +0
  e_04          clear_high                     +3      +4      +1
  e_05          clear_high                     +3      +4      +1
  e_06          moderate_high                  +2      +3      +1
  e_07          moderate_high                  +2      +2      +0
  e_08          moderate_high                  +2      +3      +1
  e_09          slight_high                    +1      +3      +2
  e_10          slight_high                    +1      +1      +0
  e_11          neutral                        +0      +0      +0
  e_12          neutral                        +0      +0      +0
  e_13          neutral                        +0      +0      +0
  e_14          neutral                        +0      +1      +1
  e_15          slight_low                     -1      -2      -1
  e_16          slight_low                     -1      -3      -2
  e_17          moderate_low                   -2      -3      -1
  e_18          moderate_low                   -2      -3      -1
  e_19          moderate_low                   -2      -3      -1
  e_20          clear_low                      -3      -2      +1
  e_21          clear_low                      -3      -3      +0
  e_22          clear_low                      -4      -3      +1
  e_23          confound_practical_social_reference     +0      -1      -1  ← confound
  e_24          slight_high                    +1      +2      +1
  e_25          slight_high                    +1      +2      +1
  e_26          slight_low                     -1      -2      -1
  e_27          neutral                        +0      +0      +0
  e_28          moderate_high                  +2      +3      +1
==========================================================================================

Running anthropic/claude-3-5-haiku (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=anthropic/claude-3-5-haiku  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.921  (n=28)
  Spearman r : +0.920
  MAE        : 0.68
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  e_01          clear_high                     +4      +4      +0
  e_02          clear_high                     +4      +4      +0
  e_03          clear_high                     +4      +3      -1
  e_04          clear_high                     +3      +3      +0
  e_05          clear_high                     +3      +3      +0
  e_06          moderate_high                  +2      +2      +0
  e_07          moderate_high                  +2      +2      +0
  e_08          moderate_high                  +2      +3      +1
  e_09          slight_high                    +1      +2      +1
  e_10          slight_high                    +1      +1      +0
  e_11          neutral                        +0      +1      +1
  e_12          neutral                        +0      +0      +0
  e_13          neutral                        +0      +0      +0
  e_14          neutral                        +0      +1      +1
  e_15          slight_low                     -1      -1      +0
  e_16          slight_low                     -1      -2      -1
  e_17          moderate_low                   -2      -3      -1
  e_18          moderate_low                   -2      -3      -1
  e_19          moderate_low                   -2      -2      +0
  e_20          clear_low                      -3      -2      +1
  e_21          clear_low                      -3      -2      +1
  e_22          clear_low                      -4      -2      +2
  e_23          confound_practical_social_reference     +0      +2      +2  ← confound
  e_24          slight_high                    +1      +2      +1
  e_25          slight_high                    +1      +2      +1
  e_26          slight_low                     -1      -2      -1
  e_27          neutral                        +0      -1      -1
  e_28          moderate_high                  +2      +3      +1
==========================================================================================

==========================================================================================
CROSS-MODEL SUMMARY  (reference: author (expected_score))
==========================================================================================
  model                                      pearson   spearman     mae   errors
  ------------------------------------------------------------------------------
  openai/gpt-4o-mini                          +0.943     +0.934    0.71      0/28
  anthropic/claude-3-5-haiku                  +0.921     +0.920    0.68      0/28

  Pairwise MAE between models:
    openai/gpt-4o-mini vs anthropic/claude-3-5-haiku: MAE=0.54 (n=28)
==========================================================================================

Consistency check: 3 runs at temp=0.9 [openai/gpt-4o-mini] ...

==========================================================================================
SELF-CONSISTENCY  (temperature=0.9)
==========================================================================================
  id            expected    mean    std   min   max  scores
  ----------------------------------------------------------------------------------
  e_01                +4   +4.00   0.00    +4    +4  +4  +4  +4
  e_02                +4   +4.00   0.00    +4    +4  +4  +4  +4
  e_03                +4   +3.67   0.58    +3    +4  +4  +3  +4
  e_04                +3   +4.00   0.00    +4    +4  +4  +4  +4
  e_05                +3   +4.00   0.00    +4    +4  +4  +4  +4
  e_06                +2   +3.00   0.00    +3    +3  +3  +3  +3
  e_07                +2   +3.00   0.00    +3    +3  +3  +3  +3
  e_08                +2   +3.00   0.00    +3    +3  +3  +3  +3
  e_09                +1   +3.00   0.00    +3    +3  +3  +3  +3
  e_10                +1   +1.00   0.00    +1    +1  +1  +1  +1
  e_11                +0   -0.33   0.58    -1    +0  +0  +0  -1
  e_12                +0   +0.00   0.00    +0    +0  +0  +0  +0
  e_13                +0   +0.00   0.00    +0    +0  +0  +0  +0
  e_14                +0   +0.67   0.58    +0    +1  +1  +0  +1
  e_15                -1   -2.00   0.00    -2    -2  -2  -2  -2
  e_16                -1   -2.67   0.58    -3    -2  -2  -3  -3
  e_17                -2   -3.00   0.00    -3    -3  -3  -3  -3
  e_18                -2   -3.00   0.00    -3    -3  -3  -3  -3
  e_19                -2   -2.33   0.58    -3    -2  -2  -3  -2
  e_20                -3   -2.33   0.58    -3    -2  -2  -3  -2
  e_21                -3   -3.00   0.00    -3    -3  -3  -3  -3
  e_22                -4   -3.00   0.00    -3    -3  -3  -3  -3
  e_23                +0   -1.33   0.58    -2    -1  -2  -1  -1
  e_24                +1   +2.67   0.58    +2    +3  +2  +3  +3
  e_25                +1   +2.67   0.58    +2    +3  +3  +2  +3
  e_26                -1   -2.00   0.00    -2    -2  -2  -2  -2
  e_27                +0   +0.00   0.00    +0    +0  +0  +0  +0
  e_28                +2   +3.00   0.00    +3    +3  +3  +3  +3

  Overall mean std: 0.186
==========================================================================================

==========================================================================================
SCORECARD
==========================================================================================

  Model: openai/gpt-4o-mini
  PASS  Pearson r vs reference                    +0.943  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.934  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.71  (threshold ≤ 1.00)
  FAIL  Confound accuracy (score=0)                  0/1  (threshold = 100%)

  Model: anthropic/claude-3-5-haiku
  PASS  Pearson r vs reference                    +0.921  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.919  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.68  (threshold ≤ 1.00)
  FAIL  Confound accuracy (score=0)                  0/1  (threshold = 100%)

  Consistency (temp=0.9, 3 runs, model=openai/gpt-4o-mini):
  PASS  Mean std across items                      0.186  (threshold ≤ 0.50 mean)

  Inter-model agreement:
  PASS  openai/gpt-4o-mini vs anthropic/claude-3-5      0.54  (threshold ≤ 1.00)
==========================================================================================

```
