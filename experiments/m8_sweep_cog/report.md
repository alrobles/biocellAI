# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5346  0.0286       0.5498  0.0288
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5180  0.0345       0.5344  0.0331
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5296  0.0238       0.5526  0.0321
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.5172  0.0364       0.5373  0.0368
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5276  0.0192       0.5545  0.0316
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5177  0.0372       0.5372  0.0369
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5277  0.0191       0.5544  0.0315
```
