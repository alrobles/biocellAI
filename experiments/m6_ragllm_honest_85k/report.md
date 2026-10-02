# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5366  0.0079       0.5621  0.0089
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5548  0.0372       0.5796  0.0285
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.4158  0.0435       0.5001  0.0093
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.3834  0.0588       0.5114  0.0175
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5871  0.0059       0.6259  0.0141
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5891  0.0125       0.6308  0.0152
```
