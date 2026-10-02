# M5 — consolidated results

Runs: ['m1_grounding_benchmark', 'm2_grounding_ablation', 'm3_ablation_85k', 'm4_llm_honest_85k', 'm4_llm_labeled_85k']
Skipped (superseded/missing): ['m2_grounding_ablation']

`leakage=True` rows are probes/upper bounds, not honest grounding.

```
                                                                                                               macro_f1         balanced_acc        
                                                                                                                   mean     std         mean     std
experiment             arm                  caption_mode text_model                                    leakage                                      
m1_grounding_benchmark cell_only_supervised -            -                                             False     0.5896  0.0026       0.6081  0.0175
                       grounded_probe       -            -                                             False     0.4261  0.0136       0.4788  0.0343
                                                                                                       True      0.4720  0.0423       0.5038  0.0183
                       grounded_zeroshot    -            -                                             False     0.2927  0.0270       0.4699  0.0306
                                                                                                       True      0.3967  0.0435       0.5595  0.0086
m3_ablation_85k        cell_only_supervised -            -                                             False     0.6028  0.0183       0.6177  0.0166
                       grounded_probe       own_genes    all-MiniLM-L6-v2                              False     0.4140  0.0006       0.4756  0.0246
                                                         cambridgeltl/SapBERT-from-PubMedBERT-fulltext False     0.4038  0.0130       0.4828  0.0311
                                            type_markers all-MiniLM-L6-v2                              False     0.5601  0.0531       0.5782  0.0466
                                                         cambridgeltl/SapBERT-from-PubMedBERT-fulltext False     0.5475  0.0127       0.5850  0.0183
                       grounded_zeroshot    own_genes    all-MiniLM-L6-v2                              False     0.2879  0.0217       0.4731  0.0306
                                                         cambridgeltl/SapBERT-from-PubMedBERT-fulltext False     0.2606  0.0103       0.4460  0.0495
                                            type_markers all-MiniLM-L6-v2                              False     0.5911  0.0147       0.6349  0.0338
                                                         cambridgeltl/SapBERT-from-PubMedBERT-fulltext False     0.5776  0.0253       0.6196  0.0201
m4_llm_honest_85k      cell_only_supervised -            -                                             False     0.6050  0.0296       0.6384  0.0126
                       grounded_probe       type_llm     all-MiniLM-L6-v2                              False     0.5310  0.0456       0.5531  0.0302
                                                         cambridgeltl/SapBERT-from-PubMedBERT-fulltext False     0.5494  0.0294       0.5719  0.0216
                       grounded_zeroshot    type_llm     all-MiniLM-L6-v2                              False     0.3240  0.0490       0.4358  0.0592
                                                         cambridgeltl/SapBERT-from-PubMedBERT-fulltext False     0.4006  0.0440       0.4871  0.0146
m4_llm_labeled_85k     cell_only_supervised -            -                                             False     0.6028  0.0120       0.6162  0.0135
                       grounded_probe       type_llm     all-MiniLM-L6-v2                              True      0.5101  0.0329       0.5299  0.0469
                                                         cambridgeltl/SapBERT-from-PubMedBERT-fulltext True      0.5645  0.0362       0.5794  0.0291
                       grounded_zeroshot    type_llm     all-MiniLM-L6-v2                              True      0.5745  0.0404       0.6251  0.0225
                                                         cambridgeltl/SapBERT-from-PubMedBERT-fulltext True      0.5533  0.0445       0.6011  0.0211
```
