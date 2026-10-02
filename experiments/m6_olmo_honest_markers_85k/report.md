# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5265  0.0593       0.5433  0.0528
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5697  0.0321       0.5906  0.0142
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.3568  0.0468       0.4626  0.0254
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.3735  0.0430       0.4986  0.0443
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5970  0.0253       0.6386  0.0142
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5896  0.0128       0.6411  0.0256
```
