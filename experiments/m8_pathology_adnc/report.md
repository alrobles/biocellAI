# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                macro_f1         balanced_acc        
                                                                                    mean     std         mean     std
arm                  caption_mode text_model                                                                         
cell_only_supervised -            -                                               0.2340  0.0374       0.2542  0.0222
grounded_probe       type_llm     all-MiniLM-L6-v2                                0.2289  0.0299       0.2504  0.0168
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2284  0.0300       0.2496  0.0146
grounded_zeroshot    type_llm     all-MiniLM-L6-v2                                0.2310  0.0320       0.2462  0.0195
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2324  0.0311       0.2464  0.0169
```
