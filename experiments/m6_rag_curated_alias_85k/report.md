# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.4444  0.0294       0.4998  0.0057
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4675  0.0383       0.5173  0.0251
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1881  0.0461       0.2868  0.0443
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2043  0.0449       0.3163  0.0429
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5119  0.0057       0.5651  0.0036
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5109  0.0065       0.5653  0.0074
```
