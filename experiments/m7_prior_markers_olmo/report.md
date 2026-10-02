# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.9886  0.0037       0.9866  0.0061
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.9479  0.0019       0.9495  0.0002
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9595  0.0259       0.9600  0.0246
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.5117  0.0489       0.5941  0.0454
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.7044  0.0634       0.7581  0.0500
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.9878  0.0014       0.9882  0.0005
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9897  0.0007       0.9905  0.0016
```
