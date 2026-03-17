# Calibration run — conscientiousness — 20260317_183038

**Models:** openai/gpt-4o-mini, anthropic/claude-3-5-haiku  
**Provider:** openrouter  
**Temperature:** 0.9  
**Items:** 28  

```

Running openai/gpt-4o-mini (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=openai/gpt-4o-mini  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.961  (n=28)
  Spearman r : +0.930
  MAE        : 0.54
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  c_01          clear_high                     +4      +4      +0
  c_02          clear_high                     +4      +4      +0
  c_03          clear_high                     +4      +4      +0
  c_04          clear_high                     +3      +4      +1
  c_05          clear_high                     +3      +4      +1
  c_06          moderate_high                  +2      +3      +1
  c_07          moderate_high                  +2      +3      +1
  c_08          moderate_high                  +2      +3      +1
  c_09          slight_high                    +1      +3      +2
  c_10          slight_high                    +1      +2      +1
  c_11          neutral                        +0      +0      +0
  c_12          neutral                        +0      +0      +0
  c_13          neutral                        +0      +0      +0
  c_14          moderate_high                  +2      +2      +0
  c_15          slight_low                     -1      -2      -1
  c_16          slight_low                     -1      -2      -1
  c_17          moderate_low                   -2      -2      +0
  c_18          moderate_low                   -2      -2      +0
  c_19          moderate_low                   -2      -2      +0
  c_20          clear_low                      -3      -2      +1
  c_21          clear_low                      -3      -3      +0
  c_22          clear_low                      -3      -3      +0
  c_23          moderate_low                   -3      -3      +0
  c_24          confound_structured_topic      +0      +0      +0  ← confound
  c_25          slight_low                     -1      -1      +0
  c_26          moderate_high                  +2      +4      +2
  c_27          slight_high                    +1      +3      +2
  c_28          moderate_low                   -2      -2      +0
==========================================================================================

Running anthropic/claude-3-5-haiku (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=anthropic/claude-3-5-haiku  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.941  (n=28)
  Spearman r : +0.875
  MAE        : 0.57
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  c_01          clear_high                     +4      +4      +0
  c_02          clear_high                     +4      +3      -1
  c_03          clear_high                     +4      +4      +0
  c_04          clear_high                     +3      +3      +0
  c_05          clear_high                     +3      +3      +0
  c_06          moderate_high                  +2      +2      +0
  c_07          moderate_high                  +2      +2      +0
  c_08          moderate_high                  +2      +2      +0
  c_09          slight_high                    +1      +2      +1
  c_10          slight_high                    +1      +2      +1
  c_11          neutral                        +0      +2      +2
  c_12          neutral                        +0      +1      +1
  c_13          neutral                        +0      +2      +2
  c_14          moderate_high                  +2      +2      +0
  c_15          slight_low                     -1      -2      -1
  c_16          slight_low                     -1      -2      -1
  c_17          moderate_low                   -2      -2      +0
  c_18          moderate_low                   -2      -2      +0
  c_19          moderate_low                   -2      -2      +0
  c_20          clear_low                      -3      -2      +1
  c_21          clear_low                      -3      -2      +1
  c_22          clear_low                      -3      -2      +1
  c_23          moderate_low                   -3      -2      +1
  c_24          confound_structured_topic      +0      +0      +0  ← confound
  c_25          slight_low                     -1      -1      +0
  c_26          moderate_high                  +2      +3      +1
  c_27          slight_high                    +1      +2      +1
  c_28          moderate_low                   -2      -2      +0
==========================================================================================

==========================================================================================
CROSS-MODEL SUMMARY  (reference: author (expected_score))
==========================================================================================
  model                                      pearson   spearman     mae   errors
  ------------------------------------------------------------------------------
  openai/gpt-4o-mini                          +0.961     +0.930    0.54      0/28
  anthropic/claude-3-5-haiku                  +0.941     +0.875    0.57      0/28

  Pairwise MAE between models:
    openai/gpt-4o-mini vs anthropic/claude-3-5-haiku: MAE=0.61 (n=28)
==========================================================================================

Consistency check: 3 runs at temp=0.9 [openai/gpt-4o-mini] ...

==========================================================================================
SELF-CONSISTENCY  (temperature=0.9)
==========================================================================================
  id            expected    mean    std   min   max  scores
  ----------------------------------------------------------------------------------
  c_01                +4   +4.00   0.00    +4    +4  +4  +4  +4
  c_02                +4   +4.00   0.00    +4    +4  +4  +4  +4
  c_03                +4   +4.00   0.00    +4    +4  +4  +4  +4
  c_04                +3   +4.00   0.00    +4    +4  +4  +4  +4
  c_05                +3   +4.00   0.00    +4    +4  +4  +4  +4
  c_06                +2   +3.00   0.00    +3    +3  +3  +3  +3
  c_07                +2   +3.00   0.00    +3    +3  +3  +3  +3
  c_08                +2   +3.00   0.00    +3    +3  +3  +3  +3
  c_09                +1   +3.00   0.00    +3    +3  +3  +3  +3
  c_10                +1   +2.33   0.58    +2    +3  +3  +2  +2
  c_11                +0   +1.67   1.53    +0    +3  +0  +3  +2
  c_12                +0   +0.00   0.00    +0    +0  +0  +0  +0
  c_13                +0   +0.33   0.58    +0    +1  +0  +1  +0
  c_14                +2   +2.00   0.00    +2    +2  +2  +2  +2
  c_15                -1   -2.00   0.00    -2    -2  -2  -2  -2
  c_16                -1   -2.00   0.00    -2    -2  -2  -2  -2
  c_17                -2   -2.00   0.00    -2    -2  -2  -2  -2
  c_18                -2   -2.00   0.00    -2    -2  -2  -2  -2
  c_19                -2   -2.00   0.00    -2    -2  -2  -2  -2
  c_20                -3   -2.67   0.58    -3    -2  -2  -3  -3
  c_21                -3   -2.00   0.00    -2    -2  -2  -2  -2
  c_22                -3   -3.00   0.00    -3    -3  -3  -3  -3
  c_23                -3   -2.33   0.58    -3    -2  -2  -2  -3
  c_24                +0   +0.00   0.00    +0    +0  +0  +0  +0
  c_25                -1   -1.00   0.00    -1    -1  -1  -1  -1
  c_26                +2   +3.67   0.58    +3    +4  +4  +3  +4
  c_27                +1   +2.67   0.58    +2    +3  +3  +2  +3
  c_28                -2   -2.33   0.58    -3    -2  -3  -2  -2

  Overall mean std: 0.199
==========================================================================================

==========================================================================================
SCORECARD
==========================================================================================

  Model: openai/gpt-4o-mini
  PASS  Pearson r vs reference                    +0.961  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.930  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.54  (threshold ≤ 1.00)
  PASS  Confound accuracy (score=0)                  1/1  (threshold = 100%)

  Model: anthropic/claude-3-5-haiku
  PASS  Pearson r vs reference                    +0.941  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.875  (threshold ≥ 0.85)
  PASS  MAE vs reference                            0.57  (threshold ≤ 1.00)
  PASS  Confound accuracy (score=0)                  1/1  (threshold = 100%)

  Consistency (temp=0.9, 3 runs, model=openai/gpt-4o-mini):
  PASS  Mean std across items                      0.199  (threshold ≤ 0.50 mean)

  Inter-model agreement:
  PASS  openai/gpt-4o-mini vs anthropic/claude-3-5      0.61  (threshold ≤ 1.00)
==========================================================================================

```
