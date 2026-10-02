# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.9886  0.0037       0.9866  0.0061
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.9468  0.0401       0.9478  0.0376
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9568  0.0274       0.9566  0.0276
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.4448  0.0270       0.5064  0.0268
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4483  0.0227       0.5461  0.0198
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.9893  0.0027       0.9890  0.0043
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9882  0.0023       0.9877  0.0042
```
