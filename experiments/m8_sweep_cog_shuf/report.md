# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(10), np.int64(11), np.int64(12)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.4886  0.0336       0.4956  0.0386
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.4931  0.0266       0.5004  0.0299
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4919  0.0202       0.4968  0.0215
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.4924  0.0272       0.4995  0.0299
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4904  0.0214       0.4948  0.0224
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.4925  0.0278       0.4997  0.0303
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4902  0.0216       0.4948  0.0228
```
