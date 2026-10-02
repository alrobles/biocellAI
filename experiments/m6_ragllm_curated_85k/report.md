# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5079  0.0222       0.5401  0.0295
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5534  0.0349       0.5828  0.0199
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.4434  0.0692       0.5390  0.0593
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4173  0.0417       0.5474  0.0170
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5861  0.0160       0.6250  0.0184
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.6028  0.0133       0.6338  0.0170
```
