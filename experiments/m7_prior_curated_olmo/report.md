# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.9886  0.0037       0.9866  0.0061
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.9159  0.0211       0.9178  0.0182
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.8900  0.0254       0.8938  0.0234
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1364  0.0014       0.1660  0.0009
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2177  0.0005       0.2900  0.0006
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.9455  0.0005       0.9485  0.0016
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.9421  0.0019       0.9458  0.0029
```
