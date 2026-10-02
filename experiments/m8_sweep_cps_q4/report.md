# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.2629  0.0068       0.2802  0.0200
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.2666  0.0080       0.2775  0.0183
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2618  0.0088       0.2721  0.0116
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.2667  0.0076       0.2795  0.0187
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2606  0.0058       0.2747  0.0120
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.2668  0.0058       0.2802  0.0184
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2614  0.0053       0.2751  0.0111
```
