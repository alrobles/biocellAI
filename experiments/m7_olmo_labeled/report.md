# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.9886  0.0037       0.9866  0.0061
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.9553  0.0298       0.9527  0.0310
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9593  0.0479       0.9594  0.0451
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.9149  0.0007       0.9477  0.0046
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9341  0.0493       0.9479  0.0401
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.9891  0.0032       0.9881  0.0064
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9874  0.0005       0.9865  0.0012
```
