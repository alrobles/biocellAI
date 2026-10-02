# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.1257  0.0303       0.1358  0.0236
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.1275  0.0311       0.1368  0.0252
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1288  0.0294       0.1374  0.0254
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1275  0.0312       0.1366  0.0252
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1272  0.0284       0.1340  0.0252
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.1273  0.0309       0.1362  0.0253
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1273  0.0283       0.1342  0.0248
```
