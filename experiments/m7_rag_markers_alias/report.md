# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.9886  0.0037       0.9866  0.0061
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.8322  0.0247       0.8523  0.0245
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.8256  0.0338       0.8517  0.0250
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1646  0.0512       0.2193  0.0474
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1738  0.0450       0.1927  0.0434
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.8867  0.0024       0.9053  0.0045
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.8873  0.0024       0.9044  0.0041
```
