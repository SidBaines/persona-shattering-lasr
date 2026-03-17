# Calibration run — coherence — 20260317_193106

**Models:** openai/gpt-4o-mini, anthropic/claude-3-5-haiku  
**Provider:** openrouter  
**Temperature:** 0.9  
**Items:** 28  

```

Running openai/gpt-4o-mini (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=openai/gpt-4o-mini  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.994  (n=28)
  Spearman r : +0.941
  MAE        : 2.50
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  coh_01        excellent                     +94     +90      -4
  coh_02        excellent                     +92     +90      -2
  coh_03        excellent                     +90     +90      +0
  coh_04        excellent                     +88     +90      +2
  coh_05        excellent                     +91     +92      +1
  coh_06        good                          +80     +85      +5
  coh_07        good                          +84     +85      +1
  coh_08        good                          +88     +90      +2
  coh_09        moderate_issues               +55     +65     +10
  coh_10        moderate_issues               +52     +60      +8
  coh_11        moderate_issues               +50     +50      +0
  coh_12        degraded                      +28     +32      +4
  coh_13        degraded                      +22     +18      -4
  coh_14        degraded                      +25     +25      +0
  coh_15        severely_degraded              +8      +6      -2
  coh_16        severely_degraded              +5      +5      +0
  coh_17        completely_broken              +2      +2      +0
  coh_18        completely_broken              +2      +2      +0
  coh_19        confound_brevity              +88     +95      +7  ← confound
  coh_20        confound_brevity              +90     +90      +0  ← confound
  coh_21        confound_style_casual         +83     +82      -1  ← confound
  coh_22        confound_hedged_tone          +89     +90      +1  ← confound
  coh_23        confound_list_format          +91     +90      -1  ← confound
  coh_24        confound_emotional_topic      +85     +85      +0  ← confound
  coh_25        confound_off_topic_intentional    +45     +40      -5  ← confound
  coh_26        confound_partial_breakdown    +68     +70      +2  ← confound
  coh_27        confound_repetition_semantic    +42     +38      -4  ← confound
  coh_28        confound_trailing_off         +62     +66      +4  ← confound
==========================================================================================

Running anthropic/claude-3-5-haiku (temp=0.9) ...

==========================================================================================
HUMAN CORRELATION  judge=anthropic/claude-3-5-haiku  reference=author (expected_score)
==========================================================================================
  Pearson r  : +0.985  (n=28)
  Spearman r : +0.969
  MAE        : 5.00
  Judge errors: 0/28

  id            cat                           ref   judge   delta
  --------------------------------------------------------------
  coh_01        excellent                     +94     +94      +0
  coh_02        excellent                     +92     +93      +1
  coh_03        excellent                     +90     +94      +4
  coh_04        excellent                     +88     +92      +4
  coh_05        excellent                     +91     +92      +1
  coh_06        good                          +80     +87      +7
  coh_07        good                          +84     +89      +5
  coh_08        good                          +88     +91      +3
  coh_09        moderate_issues               +55     +78     +23
  coh_10        moderate_issues               +52     +65     +13
  coh_11        moderate_issues               +50     +58      +8
  coh_12        degraded                      +28     +22      -6
  coh_13        degraded                      +22     +28      +6
  coh_14        degraded                      +25     +38     +13
  coh_15        severely_degraded              +8     +12      +4
  coh_16        severely_degraded              +5      +4      -1
  coh_17        completely_broken              +2      +4      +2
  coh_18        completely_broken              +2      +4      +2
  coh_19        confound_brevity              +88     +90      +2  ← confound
  coh_20        confound_brevity              +90     +92      +2  ← confound
  coh_21        confound_style_casual         +83     +85      +2  ← confound
  coh_22        confound_hedged_tone          +89     +94      +5  ← confound
  coh_23        confound_list_format          +91     +92      +1  ← confound
  coh_24        confound_emotional_topic      +85     +91      +6  ← confound
  coh_25        confound_off_topic_intentional    +45     +58     +13  ← confound
  coh_26        confound_partial_breakdown    +68     +68      +0  ← confound
  coh_27        confound_repetition_semantic    +42     +45      +3  ← confound
  coh_28        confound_trailing_off         +62     +65      +3  ← confound
==========================================================================================

==========================================================================================
CROSS-MODEL SUMMARY  (reference: author (expected_score))
==========================================================================================
  model                                      pearson   spearman     mae   errors
  ------------------------------------------------------------------------------
  openai/gpt-4o-mini                          +0.994     +0.941    2.50      0/28
  anthropic/claude-3-5-haiku                  +0.985     +0.969    5.00      0/28

  Pairwise MAE between models:
    openai/gpt-4o-mini vs anthropic/claude-3-5-haiku: MAE=5.00 (n=28)
==========================================================================================

Consistency check: 3 runs at temp=0.9 [openai/gpt-4o-mini] ...

==========================================================================================
SELF-CONSISTENCY  (temperature=0.9)
==========================================================================================
  id            expected    mean    std   min   max  scores
  ----------------------------------------------------------------------------------
  coh_01             +94  +90.67   1.15   +90   +92  +92  +90  +90
  coh_02             +92  +90.67   1.15   +90   +92  +90  +90  +92
  coh_03             +90  +90.67   1.15   +90   +92  +92  +90  +90
  coh_04             +88  +89.33   1.15   +88   +90  +88  +90  +90
  coh_05             +91  +90.00   0.00   +90   +90  +90  +90  +90
  coh_06             +80  +85.00   0.00   +85   +85  +85  +85  +85
  coh_07             +84  +85.00   0.00   +85   +85  +85  +85  +85
  coh_08             +88  +90.00   0.00   +90   +90  +90  +90  +90
  coh_09             +55  +69.33   5.13   +65   +75  +65  +68  +75
  coh_10             +52  +58.33   1.53   +57   +60  +58  +57  +60
  coh_11             +50  +47.33   4.04   +45   +52  +52  +45  +45
  coh_12             +28  +27.00   1.73   +25   +28  +25  +28  +28
  coh_13             +22  +22.00   0.00   +22   +22  +22  +22  +22
  coh_14             +25  +25.00   5.00   +20   +30  +30  +20  +25
  coh_15              +8   +8.67   2.31    +6   +10  +10  +6  +10
  coh_16              +5   +4.00   2.00    +2    +6  +4  +2  +6
  coh_17              +2   +2.00   0.00    +2    +2  +2  +2  +2
  coh_18              +2   +1.67   0.58    +1    +2  +2  +2  +1
  coh_19             +88  +93.33   2.89   +90   +95  +90  +95  +95
  coh_20             +90  +91.33   1.15   +90   +92  +92  +92  +90
  coh_21             +83  +75.67   4.04   +72   +80  +75  +72  +80
  coh_22             +89  +90.00   0.00   +90   +90  +90  +90  +90
  coh_23             +91  +90.00   0.00   +90   +90  +90  +90  +90
  coh_24             +85  +85.00   0.00   +85   +85  +85  +85  +85
  coh_25             +45  +32.33   2.52   +30   +35  +35  +32  +30
  coh_26             +68  +72.67   4.04   +68   +75  +68  +75  +75
  coh_27             +42  +41.00   3.61   +38   +45  +45  +38  +40
  coh_28             +62  +62.33   2.52   +60   +65  +62  +60  +65

  Overall mean std: 1.704
==========================================================================================

==========================================================================================
SCORECARD
==========================================================================================

  Model: openai/gpt-4o-mini
  PASS  Pearson r vs reference                    +0.994  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.941  (threshold ≥ 0.85)
  PASS  MAE vs reference                            2.50  (threshold ≤ 5.0)
  INFO  Confound score=expected (info only)         2/10

  Model: anthropic/claude-3-5-haiku
  PASS  Pearson r vs reference                    +0.985  (threshold ≥ 0.90)
  PASS  Spearman r vs reference                   +0.969  (threshold ≥ 0.85)
  PASS  MAE vs reference                            5.00  (threshold ≤ 5.0)
  INFO  Confound score=expected (info only)         1/10

  Consistency (temp=0.9, 3 runs, model=openai/gpt-4o-mini):
  PASS  Mean std across items                      1.704  (threshold ≤ 3.0 mean)

  Inter-model agreement:
  PASS  openai/gpt-4o-mini vs anthropic/claude-3-5      5.00  (threshold ≤ 5.0)
==========================================================================================

```
