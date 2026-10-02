# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.9886  0.0037       0.9866  0.0061
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.2505  0.0021       0.2914  0.0004
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2510  0.0022       0.2915  0.0006
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.0001  0.0001       0.0001  0.0001
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1489  0.0186       0.1943  0.0239
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.2244  0.0195       0.2908  0.0043
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2329  0.0020       0.2915  0.0002
```
