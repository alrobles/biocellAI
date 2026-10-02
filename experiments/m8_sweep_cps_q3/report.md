# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.4012  0.0344       0.4102  0.0244
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.4053  0.0256       0.4181  0.0109
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4017  0.0229       0.4120  0.0094
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.4036  0.0283       0.4186  0.0128
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4011  0.0225       0.4115  0.0102
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.4039  0.0289       0.4199  0.0125
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.4009  0.0226       0.4116  0.0104
```
