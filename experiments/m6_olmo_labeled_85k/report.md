# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5351  0.0444       0.5554  0.0208
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5426  0.0289       0.5692  0.0197
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.5219  0.0712       0.6065  0.0056
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4827  0.0840       0.5730  0.0248
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.6058  0.0090       0.6375  0.0202
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5970  0.0239       0.6269  0.0168
```
