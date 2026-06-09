habsa/
│
├── data/                       # Chứa file raw và file đã tiền xử lý
│   ├── raw/                    # Laptop-14, Restaurant-15/16...
│   └── processed/              # Graphs, Parsed trees, mBERT tokens
│
├── src/
│   ├── data_loader.py          # Script load dữ liệu và tạo PyTorch DataLoaders
│   ├── preprocessing.py        # spaCy dependency parsing, tạo Knowledge Graph
│   │
│   ├── modules/                # Các thành phần cốt lõi của mạng
│   │   ├── skg_an.py           # Graph Attention Layer (PyTorch Geometric)
│   │   ├── hierarchical_emb.py # Lớp gọi mBERT và tính E_combined
│   │   ├── multi_head_attn.py  # Multi-head attention implementation
│   │   ├── as_sacn.py          # Capsule Network & Dynamic Routing
│   │   └── rnn_tcn.py          # RNN, Dilated Convolutions (TCN), Softmax
│   │
│   ├── models/
│   │   └── sch_mgn.py          # Class chính ghép nối tất cả các modules trên lại
│   │
│   ├── train.py                # Vòng lặp huấn luyện (Loss, Optimizer, Backward pass)
│   ├── evaluate.py             # Tính toán Accuracy, F1-Score, MAE
│   └── utils.py                # Các hàm hỗ trợ (Positional Encodings, tính toán metric)
│
├── requirements.txt
└── README.md

