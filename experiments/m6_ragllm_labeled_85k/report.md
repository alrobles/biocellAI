# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/tabula_blood_90000_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.5996  0.0119       0.6199  0.0188
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.5294  0.0037       0.5534  0.0254
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5651  0.0425       0.5789  0.0264
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.5800  0.0258       0.6281  0.0141
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5182  0.0568       0.5784  0.0166
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.6003  0.0128       0.6349  0.0193
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.5874  0.0062       0.6250  0.0269
```
