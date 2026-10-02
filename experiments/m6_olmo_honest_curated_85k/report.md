# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5070  0.0398       0.5310  0.0288
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5565  0.0677       0.5751  0.0642
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.4327  0.0372       0.5539  0.0225
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.3707  0.0994       0.5257  0.0382
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.5981  0.0078       0.6214  0.0198
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5985  0.0095       0.6376  0.0156
```
