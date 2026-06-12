# OKE-HABSA

OKE-HABSA (Ontology and Knowledge-Enhanced Hierarchical Aspect-Based Sentiment
Analysis) là pipeline phân tích cảm xúc theo khía cạnh, kết hợp:

- XLM-R hoặc BiGRU để mã hóa văn bản;
- dependency graph encoder;
- ontology miền Laptop và ánh xạ aspect vào concept;
- TransE và ontological position encoding;
- cross-attention giữa văn bản và ontology;
- bottom-up GNN lan truyền thông tin từ concept con lên concept cha;
- các ràng buộc monotonicity, dominance và consistency;
- Integrated Gradients, concept attribution và structural propagation path.

Pipeline hiện hỗ trợ bài toán ABSA **3 lớp** trên SemEval Laptop:
`negative`, `neutral`, `positive`. Chế độ 5 lớp chỉ nên sử dụng khi có dữ liệu
được gán nhãn 5 mức hợp lệ.

## Yêu cầu

- Python 3.10 trở lên;
- PyTorch 2.2 trở lên;
- Java chỉ cần thiết khi chạy HermiT reasoner bằng `--reasoner`;
- GPU được khuyến nghị khi huấn luyện XLM-R.

Cài đặt thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Dữ liệu

Dữ liệu Laptop phải nằm tại:

```text
data/raw/Laptops/
  train.json
  test.json
```

Mỗi record chứa `token`, `pos`, `head`, `deprel` và `aspects`. Mỗi aspect cần
có `term`, `from`, `to` và `polarity`.

Pipeline tạo một sample riêng cho từng aspect. Việc chia train/validation được
thực hiện theo sentence ID, nên các aspect thuộc cùng một câu không bị rò rỉ
sang hai tập khác nhau.

## Xây dựng ontology

Xuất ontology Laptop sang OWL, JSON và TSV:

```powershell
python main.py build-ontology --dataset Laptops
```

Chạy thêm HermiT consistency check:

```powershell
python main.py build-ontology --dataset Laptops --reasoner
```

Các file mặc định được ghi vào thư mục `ontology/`.

## Chạy toàn bộ pipeline

Chỉnh tham số trong `config/default_config.yaml`, sau đó chạy duy nhất:

```powershell
python main.py
```

Lệnh này tự động:

1. xây dựng và kiểm tra ontology;
2. chia train/validation theo sentence;
3. pretrain và freeze TransE;
4. huấn luyện warm-up, main và fine-tuning;
5. nạp `best_model.pt`;
6. đánh giá trên test set;
7. lưu checkpoint, metric, config, manifest và log.

Mỗi epoch được hiển thị trên một dòng và đồng thời ghi vào `run.log`:

```text
[EPOCH] 006/030 stage=main local=01/20 scale=0.050 lr=3.00e-04 train_loss=... train_f1=... val_loss=... val_f1=... val_acc=... val_mae=... val_oc=... gap=... best_f1=...
```

Kết quả tổng hợp cuối cùng nằm trong `pipeline_results.json`.

## Chỉnh tham số chạy

Toàn bộ tham số chạy chính nằm trong `config/default_config.yaml`.

### Đường dẫn và giới hạn pipeline

```yaml
seed: 42
device: auto

pipeline:
  output_dir: outputs/laptops/full/42
  reasoner: false
  max_records: null
  max_test_records: null
```

- `seed`: seed tái lập thí nghiệm.
- `device`: `auto`, `cpu`, `cuda` hoặc CUDA device cụ thể.
- `pipeline.output_dir`: thư mục lưu toàn bộ kết quả.
- `pipeline.reasoner`: bật HermiT consistency check; yêu cầu Java.
- `pipeline.max_records`: giới hạn số câu train; `null` dùng toàn bộ.
- `pipeline.max_test_records`: giới hạn số câu test; `null` dùng toàn bộ.

### Cấu hình XLM-R đầy đủ

```yaml
model:
  text_backend: transformer
  pretrained_model: xlm-roberta-base
  local_files_only: false
  hidden_dim: 192
  ontology_dim: 96
  num_heads: 4
  dropout: 0.2
  num_sentiments: 3

ontology:
  transe_epochs: 50
  transe_margin: 1.0

training:
  batch_size: 16
  plm_learning_rate: 0.00002
  learning_rate: 0.0003
  warmup_epochs: 5
  main_epochs: 20
  finetune_epochs: 5
```

Lần chạy đầu tiên có thể tải tokenizer và trọng số XLM-R. Đặt
`model.local_files_only: true` nếu model đã có trong cache và không muốn truy
cập mạng.

Nên chạy lần lượt seed `42`, `52`, `62`, đổi
`pipeline.output_dir` tương ứng và báo cáo mean/standard deviation.

## Chống overfitting

Trainer áp dụng:

- train/validation split cố định theo sentence;
- stratification gần đúng theo polarity và ontology branch;
- class-weighted cross entropy tính riêng từ train set;
- dropout, weight decay và label smoothing;
- TransE được pretrain riêng rồi freeze;
- gradient clipping;
- `ReduceLROnPlateau` theo validation macro-F1;
- early stopping theo validation macro-F1;
- dừng khi generalization gap vượt ngưỡng trong nhiều epoch;
- constraint curriculum: `0` ở warm-up, tăng tuyến tính trong main stage.

Các tham số chống overfitting nằm trong nhóm `training` của
`config/default_config.yaml`.

## Đánh giá

```powershell
python main.py evaluate `
  --checkpoint outputs/laptops/full/42/best_model.pt `
  --dataset Laptops
```

Kết quả gồm accuracy, macro-F1, MAE, per-class metrics và ontological
consistency. Test set chỉ được dùng ở bước này, không dùng để chọn
hyperparameter hoặc checkpoint.

## Dự đoán

Aspect phải xuất hiện nguyên văn trong câu:

```powershell
python main.py predict `
  --checkpoint outputs/laptops/full/42/best_model.pt `
  --text "The battery life is excellent." `
  --aspect "battery life"
```

## Giải thích dự đoán

```powershell
python main.py explain `
  --checkpoint outputs/laptops/full/42/best_model.pt `
  --text "The battery life is excellent." `
  --aspect "battery life" `
  --steps 50
```

Output gồm xác suất dự đoán, Integrated Gradients theo token, attribution được
gom theo ontology concept và đường lan truyền từ concept aspect lên root.

## Artifact đầu ra

Mỗi run tạo các file:

```text
outputs/<domain>/<experiment>/<seed>/
  resolved_config.yaml
  split_manifest.json
  ontology.owl
  ontology.json
  ontology.tsv
  ontology_hash.txt
  mapping_report.json
  transe_best.pt
  best_model.pt
  last_model.pt
  history.json
  test_metrics.json
  per_class_metrics.json
  train_summary.json
  pipeline_results.json
  tokenizer.json
  explanations/
  run.log
```

`history.json` lưu metric train/validation, từng thành phần loss, constraint
scale và generalization gap cho mỗi epoch.

## Cấu trúc mã nguồn

```text
config/          Cấu hình và cơ chế override
data_loader/     Tokenizer, preprocessing, dataset và collator
ontology/        Ontology Laptop, SWRL companion rules và entity mapping
models/          Text encoder, TransE, fusion, GNN và mạng OKE-HABSA
losses/          Hierarchical loss và ontological constraints
trainers/        TransE pretraining và lịch huấn luyện nhiều giai đoạn
explainability/  Token, concept và structural explanations
utils/           Metrics và graph utilities
main.py          CLI entrypoint
```

## Giới hạn hiện tại

- Ontology và CLI hiện chỉ hỗ trợ miền Laptop.
- AOPC/LODDS chưa được tối ưu trực tiếp trong loss; nên dùng làm metric đánh
  giá faithfulness trên validation subset cố định.
- Mô hình 5 logits không được xem là mô hình 5 lớp hợp lệ nếu chỉ huấn luyện
  bằng dữ liệu 3 nhãn.

Kế hoạch huấn luyện và tiêu chí nghiệm thu chi tiết nằm trong
`script/TRAINING_PLAN.md`.
