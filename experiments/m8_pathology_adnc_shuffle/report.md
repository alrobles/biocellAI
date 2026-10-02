# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                macro_f1         balanced_acc        
                                                                                    mean     std         mean     std
arm                  caption_mode text_model                                                                         
cell_only_supervised -            -                                               0.2148  0.0271       0.2437  0.0203
grounded_probe       type_llm     all-MiniLM-L6-v2                                0.2078  0.0210       0.2382  0.0135
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2149  0.0173       0.2461  0.0114
grounded_zeroshot    type_llm     all-MiniLM-L6-v2                                0.2052  0.0194       0.2355  0.0166
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2130  0.0160       0.2429  0.0152
```
