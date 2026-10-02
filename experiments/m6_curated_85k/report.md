# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                macro_f1         balanced_acc        
                                                                                    mean     std         mean     std
arm                  caption_mode text_model                                                                         
cell_only_supervised -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe       type_markers all-MiniLM-L6-v2                                0.4849  0.0301       0.5353  0.0241
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4605  0.0445       0.5211  0.0443
grounded_zeroshot    type_markers all-MiniLM-L6-v2                                0.4493  0.0179       0.5244  0.0151
                                  cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4592  0.0130       0.5643  0.0075
```
