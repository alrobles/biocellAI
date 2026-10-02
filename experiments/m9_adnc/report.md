# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_allregions_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.2316  0.0224       0.2493  0.0067
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.2304  0.0180       0.2506  0.0026
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2310  0.0186       0.2522  0.0045
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.2351  0.0124       0.2477  0.0041
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2360  0.0160       0.2486  0.0027
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.2347  0.0127       0.2487  0.0039
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2361  0.0158       0.2487  0.0030
```
