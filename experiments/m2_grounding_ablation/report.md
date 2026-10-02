# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `tabula_blood` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                macro_f1         balanced_acc        
                                                                                    mean     std         mean     std
arm                  caption_mode text_model                                                                         
cell_only_supervised -            -                                               0.5896  0.0026       0.6081  0.0175
grounded_probe       own_genes    all-MiniLM-L6-v2                                0.4226  0.0230       0.4599  0.0241
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.3958  0.0269       0.4518  0.0466
                     type_markers all-MiniLM-L6-v2                                0.5581  0.0336       0.5949  0.0093
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5568  0.0387       0.5859  0.0370
grounded_zeroshot    own_genes    all-MiniLM-L6-v2                                0.2144  0.0612       0.3490  0.0151
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2955  0.0361       0.4171  0.0535
                     type_markers all-MiniLM-L6-v2                                0.5166  0.0301       0.6035  0.0250
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5535  0.0063       0.6260  0.0225
```
