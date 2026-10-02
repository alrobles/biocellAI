# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(10), np.int64(11), np.int64(12)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.1355  0.0168       0.1759  0.0236
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.1335  0.0103       0.1693  0.0125
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1364  0.0086       0.1722  0.0087
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1339  0.0143       0.1623  0.0125
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1347  0.0130       0.1668  0.0139
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.1341  0.0139       0.1626  0.0126
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1344  0.0126       0.1663  0.0145
```
