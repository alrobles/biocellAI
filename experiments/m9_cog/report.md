# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_allregions_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5179  0.0230       0.5353  0.0228
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5242  0.0060       0.5354  0.0153
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5236  0.0200       0.5376  0.0267
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.5222  0.0065       0.5371  0.0159
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5242  0.0234       0.5398  0.0284
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5219  0.0066       0.5370  0.0158
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5250  0.0242       0.5398  0.0285
```
