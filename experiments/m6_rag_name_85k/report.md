# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.4745  0.0494       0.5110  0.0358
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4959  0.0371       0.5326  0.0262
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1314  0.0146       0.1567  0.0364
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.0651  0.0087       0.1447  0.0121
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5058  0.0114       0.5537  0.0089
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5047  0.0043       0.5501  0.0068
```
