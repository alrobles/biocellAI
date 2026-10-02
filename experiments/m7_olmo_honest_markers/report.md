# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.9886  0.0037       0.9866  0.0061
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.9477  0.0377       0.9467  0.0349
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9617  0.0233       0.9618  0.0226
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.4603  0.0342       0.4951  0.0346
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.3507  0.0195       0.4831  0.0237
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.9893  0.0024       0.9881  0.0042
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9895  0.0011       0.9897  0.0008
```
