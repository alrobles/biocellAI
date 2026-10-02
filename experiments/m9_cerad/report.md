# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_allregions_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.2443  0.0268       0.2709  0.0127
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.2462  0.0243       0.2743  0.0103
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2459  0.0185       0.2760  0.0084
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.2544  0.0182       0.2735  0.0091
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2529  0.0168       0.2753  0.0084
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.2518  0.0196       0.2736  0.0087
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2533  0.0168       0.2754  0.0085
```
