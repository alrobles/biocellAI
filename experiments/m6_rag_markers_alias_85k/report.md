# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5104  0.0318       0.5563  0.0071
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5289  0.0335       0.5631  0.0201
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1464  0.0107       0.1583  0.0202
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1682  0.0326       0.2479  0.0369
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5630  0.0192       0.6135  0.0075
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5662  0.0134       0.6043  0.0125
```
