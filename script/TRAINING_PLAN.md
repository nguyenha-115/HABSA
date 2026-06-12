# Kế hoạch huấn luyện OKE-HABSA

Tài liệu này chuyển kiến trúc trong `OKE-HABSA_KienTruc_ChiTiet-2.md` thành một
kế hoạch huấn luyện có thể kiểm chứng. Miền triển khai đầu tiên là **Laptop**;
sau khi pipeline ổn định mới mở rộng sang Restaurant và Tweets.

## 1. Mục tiêu

### Mốc M1 - Baseline khả dụng

- Huấn luyện ABSA 3 lớp `negative/neutral/positive` trên SemEval Laptop.
- Có baseline text-only để chứng minh phần ontology thực sự tạo cải thiện.
- Pipeline chạy được từ dữ liệu thô đến checkpoint và báo cáo test.

### Mốc M2 - OKE-HABSA đầy đủ

- Kết hợp PLM, TransE, ontology position encoding, cross-attention và bottom-up
  GNN.
- Áp dụng các loss monotonicity, dominance và consistency theo curriculum.
- Đánh giá cả chất lượng phân loại, tính nhất quán ontology và độ trung thực
  của giải thích.

### Mốc M3 - Thang cảm xúc 5 mức

- Mở rộng thành `very_negative/negative/neutral/positive/very_positive`.
- Chỉ thực hiện khi đã có dữ liệu hoặc quy tắc gán nhãn được chuyên gia duyệt
  cho hai lớp cực trị. Không coi mô hình 5 logits huấn luyện bằng dữ liệu 3
  nhãn là một mô hình 5 lớp hợp lệ.

## 2. Hiện trạng và ràng buộc

### Dữ liệu có sẵn

| Dataset | Train sentences | Train aspects | Phân bố nhãn train |
|---|---:|---:|---|
| Laptops | 1,454 | 2,282 | negative 851, neutral 455, positive 976 |
| Restaurants | 1,980 | 3,608 | negative 807, neutral 637, positive 2,164 |
| Restaurants16 | 1,707 | 2,506 | negative 748, neutral 101, positive 1,657 |
| Tweets | 6,051 | 6,051 | negative 1,528, neutral 3,016, positive 1,507 |

Các câu Laptop dài tối đa 83 token, vì vậy `max_length: 96` là đủ. Có 538 câu
chứa nhiều aspect và 162 câu chứa các aspect có polarity khác nhau; việc tạo
sample và mask bắt buộc phải theo từng aspect, không được dùng một nhãn chung
cho toàn câu.

### Chênh lệch cần giải quyết trước khi train

1. Dữ liệu hiện chỉ có 3 nhãn nhưng kiến trúc định nghĩa 5 mức cảm xúc.
2. Cấu hình hiện tại dùng `text_backend: bigru`; kiến trúc mục tiêu dùng
   XLM-R/mBERT.
3. Cấu hình hiện tại dùng lịch `2/12/2` epoch, trong khi kiến trúc đề xuất
   `5/20/5`.
4. Tài liệu đề xuất `alpha_main=0.4`, `beta_sub=0.6`; cấu hình hiện tại đang
   đặt ngược lại.
5. Các file lõi trong working tree hiện có kích thước 0 byte, gồm model,
   trainer, dataset, loss, metrics, entrypoint và file OWL. Pipeline chưa đủ
   điều kiện để chạy cho đến khi các module này được khôi phục hoặc cài đặt
   lại và chạy ổn định.
6. Phần "tối ưu explanation faithfulness" chưa có loss khả vi cụ thể. Ở giai
   đoạn đầu, AOPC/LODDS chỉ nên dùng để đánh giá và chọn mô hình, không tuyên
   bố đã trực tiếp tối ưu faithfulness.

## 3. Gate 0 - Khôi phục pipeline và kiểm tra kỹ thuật

Chỉ bắt đầu thí nghiệm khi toàn bộ điều kiện sau đạt:

- `main.py`, data loader, ontology manager, model, loss, trainer và metrics
  không còn trống và import thành công.
- Ontology Laptop load được, không có cycle, mọi node trừ root có đúng một
  parent, depth nằm trong `0..3`.
- Xuất được `.owl`, `.json`, `.tsv`; reasoner không báo ontology bất nhất.
- Pipeline đầy đủ chạy từ ontology, huấn luyện, checkpoint đến đánh giá.
- Các thành phần loss hữu hạn, không xuất hiện NaN/Inf.
- Checkpoint lưu và nạp lại được để đánh giá.
- Split theo sentence, không rò rỉ các aspect cùng câu sang hai tập.

**Tiêu chí pass:** không có lỗi runtime, không có NaN/Inf và toàn bộ artifact
bắt buộc được tạo.

## 4. Gate 1 - Chuẩn hóa ontology và entity mapping

Ontology Laptop tham chiếu hiện có 28 concepts, 27 cạnh phân cấp, depth tối đa
3 và 2 concepts critical. Trước khi huấn luyện:

1. Bổ sung lexicon cho các aspect phổ biến chưa được phủ tốt, ví dụ
   `use`, `features`, `motherboard`, cùng các biến thể số ít/số nhiều.
2. Tạo một tập audit tối thiểu 300 aspect mentions, phân tầng theo tần suất và
   có nhãn concept do người kiểm tra.
3. Tune `mapping_threshold` trên validation trong dải `0.45, 0.55, 0.65,
   0.75, 0.85`.
4. So sánh ba mapping mode:
   - exact/lexicon;
   - fuzzy hoặc TF-IDF;
   - contextual embedding bằng cùng PLM với text encoder.
5. Cache kết quả mapping và Sentence Knowledge Graph theo version ontology để
   các run dùng đúng một đầu vào.

**Tiêu chí pass:** concept mapping precision >= 90%, coverage >= 90%, tỷ lệ
fallback về root <= 10%. Báo cáo riêng implicit aspect recall; không trộn
implicit inference vào số liệu mapping explicit.

## 5. Gate 2 - Chuẩn bị dữ liệu

### Split

- Giữ nguyên test set chính thức và tuyệt đối không dùng để tune.
- Chia train thành 90% train, 10% validation theo `sentence_id`.
- Stratify gần đúng theo polarity và main ontology concept.
- Lưu manifest ID của từng split để mọi seed dùng cùng dữ liệu.

### Nhãn

- M1/M2 dùng head 3 lớp, class index `0, 1, 2` và ánh xạ score
  `-1, 0, +1`. Data loader phải chọn mapping theo `num_sentiments`; mapping
  index `1, 2, 3` của head 5 lớp không dùng được cho head 3 lớp.
- Dùng class-weighted cross entropy; log cả kết quả có và không có weighting.
- Với M3, xây guideline rõ ràng cho `very_negative` và `very_positive`, đo
  agreement giữa annotator trước khi train.
- Không tự động đổi `positive -> very_positive` hoặc
  `negative -> very_negative` dựa riêng vào từ tăng cường nếu chưa được kiểm
  chứng thủ công.

### Kiểm tra dữ liệu

- Aspect span phải nằm trong giới hạn token và mask không rỗng.
- Dependency head phải hợp lệ sau truncation.
- Mỗi concept phải có đường đi tới root.
- Báo cáo số sample theo polarity, concept, depth, explicit/implicit và
  mapping confidence.

## 6. Baseline bắt buộc

Chạy các baseline trước OKE-HABSA:

| ID | Mô hình | Mục đích |
|---|---|---|
| B0 | Majority class | Mức sàn dữ liệu |
| B1 | BiGRU text-only | Baseline nhẹ, offline |
| B2 | XLM-R text-only | Baseline PLM công bằng |
| B3 | XLM-R + dependency encoder | Đo riêng đóng góp cú pháp |

Mọi cải thiện của mô hình đầy đủ phải được so với B2 trên cùng split, seed,
tokenizer, batch policy và early stopping.

## 7. Huấn luyện OKE-HABSA theo 4 giai đoạn

### Stage 1 - Pretrain ontology embedding

- Input: toàn bộ triple `(head, relation, tail)` từ ontology đã validate.
- Model: TransE, dimension 96 hoặc 128, margin 1.0.
- Negative sampling: thay head/tail 50/50, lọc false negatives nếu triple sinh
  ra đã tồn tại.
- Train tối thiểu 50 epoch; thử 50, 100 và 200 nếu ontology loss chưa hội tụ.
- Lưu `transe_best.pt`, entity vectors, relation vectors và ontology hash.
- Theo đúng kiến trúc, freeze TransE trong các stage sau. Fine-tune TransE chỉ
  là một ablation riêng.

### Stage 2 - Warm-up text và fusion

- 5 epoch, constraint scale bằng 0.
- Train text encoder, dependency encoder, hierarchy projection,
  cross-attention, GNN và classifier.
- Loss: `L_hier = 0.4 * L_main + 0.6 * L_sub`.
- Mục tiêu: học tín hiệu polarity trước khi áp ràng buộc logic.

### Stage 3 - Main training

- Tối đa 20 epoch.
- Tăng `constraint_scale` tuyến tính từ 0 lên 1; trọng số cuối:
  - `lambda_mono = 0.1`;
  - `lambda_dom = 0.2`;
  - `lambda_cons = 0.1`.
- Theo dõi từng loss riêng. Nếu một constraint lớn hơn classification loss
  kéo dài 2 epoch, dừng tăng scale và kiểm tra rule/data thay vì tiếp tục ép.
- Early stopping theo validation macro-F1 với patience 5; OC là metric kiểm
  soát, không thay thế metric phân loại.

### Stage 4 - Low-LR fine-tuning

- 5 epoch, learning rate giảm 10 lần.
- Giữ constraint scale bằng 1.
- Chọn checkpoint theo macro-F1; nếu chênh lệch macro-F1 <= 0.2 điểm phần trăm,
  ưu tiên checkpoint có OC và AOPC tốt hơn.
- AOPC/LODDS được tính trên một validation subset cố định để hạn chế chi phí.

## 8. Hyperparameter khởi điểm

| Nhóm | Giá trị |
|---|---|
| PLM | `xlm-roberta-base` |
| Max length | 96 |
| Batch size | 16-24 cho XLM-R; gradient accumulation để effective batch 32-48 |
| PLM learning rate | `2e-5` |
| Non-PLM learning rate | `3e-4` |
| Weight decay | `0.01` |
| Dropout | `0.2`, sweep thêm `0.3` |
| Gradient clip | `1.0` |
| Ontology dim | 96 hoặc 128 |
| Attention heads | 4 |
| Seeds | 42, 52, 62 |

Không sweep toàn bộ không gian ngay từ đầu. Chọn cấu hình bằng một seed, sau
đó mới chạy ba seed cho baseline mạnh nhất và mô hình đầy đủ.

## 9. Ma trận ablation

Chạy lần lượt trên cùng một seed:

| ID | Thành phần |
|---|---|
| A0 | XLM-R text-only |
| A1 | A0 + ontology TransE |
| A2 | A1 + ontology position encoding |
| A3 | A2 + cross-attention |
| A4 | A3 + bottom-up GNN |
| A5 | A4 + monotonicity |
| A6 | A5 + dominance |
| A7 | A6 + consistency, mô hình đầy đủ |
| A8 | A7 nhưng fine-tune TransE |
| A9 | A7 với BiGRU thay XLM-R |

Chỉ chạy ba seed cho B2, A4 và A7 hoặc các cấu hình thắng tương ứng.

## 10. Metrics và tiêu chí nghiệm thu

### Phân loại

- Primary: macro-F1 theo aspect.
- Secondary: accuracy, per-class precision/recall/F1, MAE trên thang sentiment.
- Báo cáo mean, standard deviation qua ba seed.

### Ontology

- Ontological Consistency (OC).
- Mapping precision, coverage, root fallback rate.
- F1 theo ontology depth và theo concept tần suất thấp.
- Implicit Aspect Recall trên tập được gán nhãn riêng.

### Explainability

- Token level: Integrated Gradients, 50 steps.
- Concept level: SHAP được gom theo ontology concept.
- Structural: đường lan truyền và trọng số child-to-parent.
- Faithfulness: AOPC và LODDS trên cùng một subset.
- Human evaluation chỉ thực hiện khi có protocol, tối thiểu hai người đánh
  giá và đo agreement.

### Điều kiện chấp nhận M2

- Macro-F1 trung bình của full model cao hơn XLM-R text-only; kết quả không chỉ
  đến từ một seed.
- OC không thấp hơn baseline và không giảm mạnh macro-F1 để đổi lấy consistency.
- Mapping đạt Gate 1.
- Ablation chỉ ra ít nhất một thành phần ontology tạo cải thiện đo được.
- Checkpoint có thể tái lập từ config, split manifest, ontology hash và seed.

## 11. Trình tự chạy dự kiến

Sau khi Gate 0 hoàn thành và data loader đã hỗ trợ mapping nhãn động theo số
lớp, chỉnh tham số trong `config/default_config.yaml` và chạy:

```powershell
python main.py
```

## 12. Artifact bắt buộc cho mỗi run

```text
outputs/<domain>/<experiment>/<seed>/
  resolved_config.yaml
  split_manifest.json
  ontology.owl
  ontology.json
  ontology_hash.txt
  mapping_report.json
  transe_best.pt
  best_model.pt
  last_model.pt
  history.json
  test_metrics.json
  per_class_metrics.json
  explanations/
  run.log
```

## 13. Lịch thực hiện đề xuất

| Giai đoạn | 
|---|
| Khôi phục và xác nhận pipeline đầy đủ | 
| Audit ontology/mapping và chuẩn hóa split | 
| Baseline B0-B3 |
| OKE-HABSA stages và ablation một seed | 
| Ba seed, đánh giá chính thức và explainability | 
| Tổng hợp báo cáo | 

Thời gian GPU thực tế phụ thuộc phần cứng và việc model XLM-R đã có trong cache.
Restaurant chỉ bắt đầu sau khi Laptop đạt Gate M2; cần ontology và mapping audit
riêng, không dùng nguyên ontology Laptop.
