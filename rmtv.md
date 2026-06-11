# OKE-HABSA

Triển khai mô hình **Phân tích cảm xúc dựa trên khía cạnh phân cấp được tăng cường thực thể & tri thức**

repository này bao gồm:

* Các ontology lập trình cho các lĩnh vực máy tính xách tay (laptop), nhà hàng (restaurant) và mạng xã hội (social) với khả năng xuất ra các định dạng OWL, JSON và TSV.


* Ánh xạ thực thể từ vựng và các quy tắc khía cạnh ẩn (implicit-aspect rules).


* Bộ mã hóa BiGRU offline hoặc tùy chọn transformer Hugging Face local.


* Nhúng phân cấp TransE (TransE hierarchy embeddings) và mã hóa vị trí ontology hình sin (sinusoidal ontology position encoding).


* Cơ chế chú ý chéo hai chiều (bidirectional cross-attention) giữa văn bản và các khái niệm ontology.


* Lan truyền GNN ontology từ dưới lên (bottom-up) với trọng số khía cạnh quan trọng (critical-aspect weighting).


* Các hàm mất mát (loss): phân cấp (hierarchical), đơn điệu (monotonicity), áp đảo (dominance), nhất quán (consistency) và KGE.


* Quy trình huấn luyện bốn giai đoạn và các giải thích ở cấp độ: token, khái niệm (concept), cấu trúc (structural), phản thực tế (counterfactual) và toàn cục (global).



## Cài đặt

```powershell
python -m pip install -r requirements.txt
```

Backend mặc định `bigru` không yêu cầu tải xuống mô hình. Để sử dụng XLM-R, hãy đặt `model.text_backend=transformer` và đảm bảo mô hình đã định cấu hình có sẵn trong bộ nhớ đệm (cache) Hugging Face cục bộ, hoặc đặt `model.local_files_only=false`.

## Commands

Tạo các tài nguyên ontology:

```powershell
python main.py build-ontology --dataset Laptops
```

Huấn luyện bốn giai đoạn đầy đủ:

```powershell
python main.py train --dataset Laptops --output-dir outputs/laptops
```

Chạy một lượt huấn luyện tích hợp ngắn:

```powershell
python main.py --set data.validation_ratio=0.02 --set training.batch_size=16 train --dataset Laptops --epochs 1 --transe-epochs 1
```

Evaluate:
```powershell
python main.py evaluate --checkpoint outputs/laptops/best_model.pt
```

Dự đoán và giải thích:

```powershell
python main.py predict --checkpoint outputs/laptops/best_model.pt `
  --text "The battery life is excellent." --aspect "battery life"

python main.py explain --checkpoint outputs/laptops/best_model.pt `
  --text "The touchpad is not responsive." --aspect "touchpad" --steps 20

```

Cấu hình nằm trong tệp `config/default_config.yaml`. Bất kỳ cài đặt nào cũng có thể được ghi đè bằng các đối số lặp lại `--set section.key=value`.

## Nhãn cảm xúc

OKE-HABSA dự đoán năm cấp độ: `very_negative`, `negative`, `neutral`, `positive` và `very_positive`, tương ứng với khoảng từ `-2..+2`. Các tập dữ liệu SemEval đi kèm chứa ba nhãn, được ánh xạ thành `-1`, `0` và `+1`.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```