# Các tiêu chuẩn và khoảng cách

> Chọn sai và mọi thứ sẽ bị phá vỡ.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## Mục tiêu học tập

- Thực hiện L1, L2, cosine, Mahalanobis, Jaccard, và chỉnh sửa các hàm khoảng cách từ đầu
- Chọn số liệu khoảng cách thích hợp cho một nhiệm vụ ML nhất định và giải thích lý do tại sao các thay thế thất bại
- Kết nối các tiêu chuẩn L1 và L2 với LASSO và Ridge và các khu vực hạn chế hình học của chúng
- Hiển thị cách cùng một tập dữ liệu tạo ra các hàng xóm gần nhất khác nhau dưới các số liệu khác nhau

## Vấn đề

Bạn có hai vector. Có thể chúng là word embeddings. Có thể là user profiles. Có thể là array pixel. Bạn cần phải biết: chúng gần nhau đến mức nào?

Câu trả lời hoàn toàn phụ thuộc vào hàm khoảng cách bạn chọn. Hai điểm dữ liệu có thể là hàng xóm gần nhất dưới một thước đo và xa nhau dưới một số khác. Bộ phân loại KNN của bạn, động cơ khuyến nghị của bạn, cơ sở dữ liệu vector của bạn, thuật toán nhóm của bạn, hàm mất mát của bạn - tất cả đều phụ thuộc vào sự lựa chọn này. Hãy sai lầm và mô hình của bạn tối ưu hóa cho điều sai lầm.

Không có khoảng cách tốt nhất phổ quát. L2 hoạt động cho dữ liệu không gian. Sự tương đồng cosine thống trị NLP. Jaccard xử lý tập hợp. Edit distance handles strings. Mahalanobis tính toán cho mối tương quan. Wasserstein di chuyển khối lượng xác suất. Mỗi một mã hóa một giả định khác về điều "tương tự" có nghĩa là gì.

Bài học này xây dựng mọi hàm khoảng cách lớn từ đầu, cho bạn thấy mỗi công cụ là công cụ phù hợp, và chứng minh cách cùng một dữ liệu tạo ra những người hàng xóm gần nhất hoàn toàn khác nhau tùy thuộc vào số liệu bạn sử dụng.

## Khái niệm

### Các chuẩn: đo lường độ lớn của vector

Norm đo "sai" của một vector. Mỗi hàm khoảng cách giữa hai vector có thể được viết như chuẩn của sự khác biệt của chúng: d(a, b) = a - b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b

### L1 Norm (trường Manhattan)

Tỷ lệ L1 tổng hợp các giá trị tuyệt đối của tất cả các thành phần.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Nó được gọi là khoảng cách Manhattan vì nó đo được bạn đi bao xa trên lưới thành phố mà bạn chỉ có thể di chuyển dọc theo trục. Không có đường chéo.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

Khi nào sử dụng L1:
- Dữ liệu hiếm có chiều cao (chương vị văn bản, mã hóa đơn)
- Khi bạn muốn độ bền đến mức ngoại lệ (một sự khác biệt lớn không thống trị)
- Các vấn đề lựa chọn tính năng (L1 quy định thúc đẩy sự khan hiếm)

Kết nối với L1 điều chỉnh: thêm vào hàm mất của bạn (Lasso) phạt tổng số các giá trị trọng lượng tuyệt đối. Điều này đẩy trọng lượng nhỏ đến chính xác bằng không, thực hiện lựa chọn tính năng tự động.

Kết nối với các chức năng mất mát: Phỏng lẻo tuyệt đối trung bình (MAE) là khoảng cách trung bình L1 giữa dự đoán và mục tiêu. Nó phạt tất cả các lỗi theo đường thẳng, làm cho nó mạnh mẽ đến các mức ngoại lệ so với MSE.

### L2 Norm (trường dài Euclidean)

L2 là đường thẳng đường dài.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Đây là khoảng cách bạn học trong lớp hình học. Pythagoras trong n chiều.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

Khi nào sử dụng L2:
- Dữ liệu liên tục ở chiều thấp đến trung bình
- Khi các thang điểm tính năng có thể so sánh
- Khoảng cách vật lý (kết lượng dữ liệu không gian, đọc cảm biến)
- Sự tương đồng hình ảnh ở cấp độ pixel

Kết nối với L2 điều chỉnh: thêm Unww ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải ải                                                                    

Kết nối với các hàm mất mát: Phỏng lẻo bình phương trung bình (MSE) là trung bình của khoảng cách bình phương L2. Phản độ hình phạt lỗi lớn nặng hơn những lỗi nhỏ.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp Norm: gia đình chung

L1 và L2 là các trường hợp đặc biệt của chuẩn Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Các giá trị khác nhau của p tạo ra các "cánh đơn vị" có hình dạng khác nhau (số tất cả các điểm ở khoảng cách 1 từ nguồn):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-không giới hạn Norm (trường độ Chebyshev)

Khi p tiến gần vô hạn, chuẩn Lp hội tụ với thành phần tuyệt đối tối đa.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Khoảng cách giữa hai điểm được xác định bởi chiều kích duy nhất mà chúng khác nhau nhất.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Khi nào sử dụng L- Infinity:
- Khi sự lệch điệu tồi tệ nhất trong bất kỳ chiều kích đơn lẻ nào quan trọng
- Các bảng chơi (một vua trong cờ vua di chuyển trong L- vô hạn: một bước theo bất kỳ hướng nào chi phí 1)
- Độ dung nạp sản xuất (mỗi kích thước phải nằm trong quy định)

### Sự tương đồng của cosine và khoảng cách cosine

Sự tương đồng cosine đo góc giữa hai vector, bỏ qua quy mô của chúng.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Nó dao động từ -1 (nghĩa ngược) đến +1 (nghĩa tương tự).

Khoảng cách cosine biến nó thành khoảng cách: cosine_distance = 1 - cosine_similarity.

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Tại sao cosine thống trị NLP và nhúng: trong văn bản, chiều dài tài liệu không nên ảnh hưởng đến sự tương đồng. Một tài liệu về mèo dài gấp đôi tài liệu khác về mèo vẫn nên "tương tự". Sự tương tự của Cosine bỏ qua quy mô (giờ dài) và chỉ quan tâm đến hướng. Hai tài liệu với phân bố từ ngữ tương tự nhưng chiều dài khác nhau chỉ ra cùng một hướng và có sự tương đồng cosine 1.0.

Khi sử dụng sự tương tự cosine:
- Sự tương đồng văn bản (trường dẫn TF-IDF, nhúng từ, nhúng câu)
- Bất kỳ lĩnh vực nào mà độ lớn là tiếng ồn và hướng là tín hiệu
- Hệ thống khuyến nghị (những vector ưu tiên của người dùng)
- Nhập tìm kiếm (tổ dữ liệu vector hầu như luôn sử dụng cosine hoặc điểm sản phẩm)

### Sự tương đồng của sản phẩm điểm so với sự tương đồng của cosine

Kết quả điểm của hai vector là:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Sự tương đồng của cosine là sản phẩm điểm được bình thường hóa bởi cả hai độ lớn. Khi cả hai vector đã được bình thường hóa đơn vị (tầm lớn = 1), sản phẩm điểm và sự tương đồng cosine là giống nhau.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

Khi chúng khác nhau: sản phẩm điểm bao gồm thông tin về quy mô. Một vector có quy mô lớn hơn có điểm điểm sản phẩm cao hơn. Điều này quan trọng trong một số hệ thống tìm kiếm nơi bạn muốn các mục "thân trọng" xếp hạng cao hơn.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

Trong thực tế:
- Sử dụng sự tương đồng cosine khi bạn muốn sự tương đồng hướng tinh khiết
- Sử dụng sản phẩm điểm khi quy mô mang thông tin có ý nghĩa
- Nhiều cơ sở dữ liệu vector (Pinecone, Weaviate, Qdrant) cho phép bạn chọn giữa chúng
- Nếu các nhúng của bạn là L2- bình thường, sự lựa chọn không quan trọng

### Khoảng cách của Mahalanobis

Khoảng cách Euclidean đối xử với tất cả các chiều kích bằng nhau. Nhưng nếu các tính năng của bạn tương quan hoặc có các thang khác nhau, L2 sẽ đưa ra kết quả sai lầm.

Khoảng cách Mahalanobis chiếm cấu trúc sự khác nhau của dữ liệu.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

nơi S là matrix covariance của dữ liệu.

Nhìn trực quan: khoảng cách của Mahalanobis trước tiên giải thích và bình thường hóa dữ liệu (tẩy trắng), sau đó tính khoảng cách L2 trong không gian biến đổi đó. Nếu S là ma trận danh tính (không liên quan, tính năng biến thể đơn vị), khoảng cách của Mahalanobis giảm xuống khoảng cách Euclidean.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Khi nào để sử dụng khoảng cách Mahalanobis:
- Khám phá ngoại lệ (điểm với khoảng cách lớn của Mahalanobis từ trung bình là ngoại lệ)
- Việc phân loại khi các tính năng có quy mô và tương quan khác nhau
- Khi bạn có đủ dữ liệu để ước tính một matrix covariance đáng tin cậy
- Kiểm soát chất lượng trong sản xuất (phòng dõi quá trình đa biến)

### Jaccard tương tự (đối với các bộ)

Các biện pháp tương đồng Jaccard chồng chéo giữa hai bộ.

```
J(A, B) = |A intersect B| / |A union B|
```

Nó dao động từ 0 (không chồng chéo) đến 1 (các bộ giống nhau).

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Khi nào sử dụng Jaccard:
- So sánh các tập hợp thẻ, danh mục hoặc tính năng
- Sự tương đồng tài liệu dựa trên sự hiện diện của từ ngữ (không phải tần suất)
- Khám phá gần trùng lặp (MinHash gần Jaccard)
- So sánh các vector tính năng nhị phân (dữ liệu hiện diện/không có)
- Các mô hình phân đoạn đánh giá (Tạm dịch: giao cắt giữa Liên minh = Jaccard)

### Edit Distance (Levenshtein Distance)

Edit distance đếm số lượng tối thiểu các hoạt động đơn ký tự cần thiết để chuyển đổi một chuỗi thành một chuỗi khác.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Được tính bằng cách sử dụng lập trình động. Sẵn đi một số liệu mà nhập (i, j) là khoảng cách chỉnh sửa giữa các ký tự i đầu tiên của chuỗi A và các ký tự j đầu tiên của chuỗi B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Khi nào để sử dụng edit distance:
- Kiểm tra và sửa chữa đánh pháp
- Định dạng chuỗi DNA (với các hoạt động trọng lượng)
- Đáp ứng dây vẫy
- Kích thước của dữ liệu văn bản lộn xộn

### KL Divergence (không phải là khoảng cách, nhưng được sử dụng như một)

KL phân biệt đo lường cách phân phối xác suất khác nhau như thế nào. được bao gồm trong Bài học 09, nhưng nó thuộc về cuộc thảo luận này bởi vì mọi người sử dụng nó như một "công xa" mặc dù nó không phải là một.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Tính chất quan trọng: Sự phân biệt KL KHÔNG đối xứng.

```
D_KL(P || Q) != D_KL(Q || P)
```

Điều này có nghĩa là nó không đáp ứng được yêu cầu cơ bản của một số lượng đường dài. Nó cũng không đáp ứng được sự bất bình đẳng tam giác.

KL phía trước (D_KL(P ỏi Q)) là "số tìm kiếm": Q cố gắng bao gồm tất cả các chế độ của P.
KL ngược (D_KL(Q ỏi P)) là " tìm kiếm chế độ": Q tập trung vào một chế độ duy nhất của P.

Khi bạn thấy sự khác biệt KL:
- VAEs (từ KL trong ELBO đẩy phân bố tiềm ẩn về phía trước)
- Sự phân phối kiến thức (những học sinh cố gắng để phù hợp với phân phối của giáo viên)
- RLHF (tình phạt KL giữ cho mô hình được điều chỉnh tốt gần với mô hình cơ bản)
- Các phương pháp gradient chính sách (khác định các cập nhật chính sách)

### Khoảng cách Wasserstein (Trong cách của người di chuyển Trái đất)

Khoảng cách của Wasserstein đo "phát" tối thiểu cần thiết để biến phân phối xác suất thành phân phối khác. Hãy nghĩ về nó như: nếu một phân phối là một đống bụi và một phần khác là một lỗ, bạn phải di chuyển bao nhiêu bụi và bao xa?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

Đối với phân phối 1D, nó đơn giản hóa thành phần tích hợp của sự khác biệt tuyệt đối của các hàm phân phối tích lũy:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Tại sao Wasserstein quan trọng:
- Nó là một métric thực (tương đối, đáp ứng bất bình đẳng tam giác)
- Nó cung cấp gradient ngay cả khi phân phối không chồng chéo (khiến độ KL đi đến vô hạn)
- Cất lượng này làm cho nó trở thành trung tâm của các GAN Wasserstein (WGAN), giải quyết sự bất ổn đào tạo của các GAN ban đầu

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Khi nào sử dụng Wasserstein:
- Việc đào tạo GAN (WGAN, WGAN-GP)
- So sánh phân bố không có sự chồng chéo
- Vấn đề vận tải tối ưu
- Khám ảnh (sự so sánh histogram màu)

### Tại sao việc làm khác nhau cần cách xa khác nhau

| Task | Best distance | Why |
|------|--------------|-----|
| Text similarity | Cosine | Magnitude is noise, direction is meaning |
| Image pixel comparison | L2 | Spatial relationships matter, features are comparable scale |
| Sparse high-dim features | L1 | Robust, does not amplify rare large differences |
| Set overlap (tags, categories) | Jaccard | Data is naturally set-valued, not vectorial |
| String matching | Edit distance | Operations map to human editing intuition |
| Outlier detection | Mahalanobis | Accounts for feature correlations and scales |
| Comparing distributions | KL divergence | Measures information lost by using Q instead of P |
| GAN training | Wasserstein | Provides gradients even when distributions do not overlap |
| Embeddings (vector DB) | Cosine or dot product | Embeddings are trained to encode meaning in direction |
| Recommendation | Dot product | Magnitude can encode popularity or confidence |
| DNA sequences | Weighted edit distance | Substitution costs vary by nucleotide pair |
| Manufacturing QC | L-infinity | Worst-case deviation in any dimension matters |

### Kết nối với các chức năng mất mát

Các hàm mất mát là các hàm khoảng cách được áp dụng cho dự đoán so với mục tiêu.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Kết nối với quy định

Việc quy định thêm một hình phạt chuẩn trên trọng lượng vào hàm mất.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Tại sao L1 tạo ra sự thô lỗ nhưng L2 không tạo ra: hình dung khu vực hạn chế trong không gian trọng lượng 2D. L1 là kim cương, L2 là một vòng tròn. Các đường viền của hàm mất mát (các vòng tròn) có nhiều khả năng chạm vào kim cương ở góc, nơi một trọng lượng là không. Chúng chạm vào vòng tròn ở một điểm mịn, nơi cả hai trọng lượng là không bằng không.

### Tìm người hàng xóm gần nhất

Mỗi hàm khoảng cách có nghĩa là một vấn đề tìm kiếm hàng xóm gần nhất: với một điểm truy vấn, tìm các điểm gần nhất trong một tập dữ liệu.

Tìm kiếm hàng xóm gần nhất chính xác là O(n * d) cho mỗi truy vấn trong một tập dữ liệu gồm n điểm với kích thước d. Đối với tập dữ liệu lớn, điều này quá chậm.

Các thuật toán hàng xóm gần nhất (ANN) giao dịch một lượng nhỏ độ chính xác để tăng tốc độ lớn:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) là thuật toán thống trị trong cơ sở dữ liệu vector hiện đại. Nó xây dựng một biểu đồ đa tầng nơi mỗi nút kết nối với hàng xóm gần nhất của nó. Tìm kiếm bắt đầu ở lớp trên (sparse, nhảy dài) và xuống tầng dưới (thiên, nhảy ngắn).

```figure
norm-unit-balls
```

## Hãy xây dựng nó

### Bước 1: Tất cả các chức năng chuẩn và khoảng cách

Nhìn xem`code/distances.py`Mỗi hàm được xây dựng từ đầu chỉ bằng toán học Python cơ bản.

### Bước 2: cùng dữ liệu, khoảng cách khác nhau, hàng xóm khác nhau

Demo trong `distances.py`tạo ra một bộ dữ liệu, chọn một điểm truy vấn, và cho thấy cách hàng xóm gần nhất thay đổi tùy thuộc vào thước đo khoảng cách. Điểm "gần nhất" dưới L1 có thể không gần nhất dưới L2 hoặc cosine.

### Bước 3: Nhập tìm kiếm tương đồng

Mã bao gồm một tìm kiếm tương đồng giả mạo nhúng tìm kiếm "bài liệu" tương tự nhất với một truy vấn sử dụng tương đồng cosine so với khoảng cách L2, cho thấy xếp hạng có thể khác nhau.

## Sử dụng nó

Sử dụng thực tế phổ biến nhất: tìm các mục tương tự trong cơ sở dữ liệu vector.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Khi anh gọi`model.encode(text)`và sau đó tìm kiếm một cơ sở dữ liệu vector, đây là những gì xảy ra dưới nắp. mô hình nhúng bản đồ văn bản cho vector. cơ sở dữ liệu vector tính toán sự tương đồng cosine (hoặc sản phẩm chấm) giữa vector truy vấn của bạn và mỗi vector lưu trữ, sử dụng thuật toán ANN để tránh kiểm tra tất cả chúng.

## Các bài tập

1. Xét khoảng cách L1, L2 và L-lực lượng vô hạn giữa (1, 2, 3) và (4, 0, 6). Kiểm tra rằng L-inf <= L2 <= L1 luôn giữ cho bất kỳ cặp điểm nào.

2. Tạo hai vector nơi sự tương đồng cosine là cao (> 0,9) nhưng khoảng cách L2 là lớn (> 10). Giải thích hình học những gì đang xảy ra.

3. Thực hiện một hàm lấy một tập dữ liệu và một điểm truy vấn và trả lại hàng xóm gần nhất dưới khoảng cách L1, L2, cosine và Mahalanobis. Tìm một tập dữ liệu mà cả bốn không đồng ý về điểm gần nhất.

4. Xét khoảng cách Wasserstein giữa [0,5, 0,5, 0,0] và [0, 0, 0,5, 0,5] bằng tay bằng phương pháp CDF. Sau đó tính nó giữa [0,25, 0,25, 0,25, 0,25] và [0, 0, 0, 0,5, 0.5].

5. Thực hiện MinHash để tương tự Jaccard gần như. Tạo 100 bộ ngẫu nhiên, tính toán chính xác Jaccard cho tất cả các cặp, và so sánh với sự gần gũi MinHash bằng cách sử dụng hàm hash 50, 100, và 200.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Norm | "Size of a vector" | A function that maps a vector to a non-negative scalar, satisfying triangle inequality, absolute homogeneity, and zero only for the zero vector |
| L1 norm | "Manhattan distance" | Sum of absolute component values. Produces sparsity in optimization. Robust to outliers |
| L2 norm | "Euclidean distance" | Square root of sum of squared components. The straight-line distance in Euclidean space |
| Lp norm | "Generalized norm" | The p-th root of the sum of p-th powers of absolute components. L1 and L2 are special cases |
| L-infinity norm | "Max norm" or "Chebyshev distance" | The maximum absolute component value. The limit of Lp as p approaches infinity |
| Cosine similarity | "Angle between vectors" | Dot product normalized by both magnitudes. Ranges from -1 to +1. Ignores vector length |
| Cosine distance | "1 minus cosine similarity" | Converts cosine similarity to a distance. Ranges from 0 to 2 |
| Dot product | "Unnormalized cosine" | Sum of component-wise products. Equals cosine similarity times both magnitudes |
| Mahalanobis distance | "Correlation-aware distance" | L2 distance in a space that has been whitened (decorrelated and normalized) using the data covariance matrix |
| Jaccard similarity | "Set overlap" | Size of intersection divided by size of union. For sets, not vectors |
| Edit distance | "Levenshtein distance" | Minimum insertions, deletions, and substitutions to transform one string into another |
| KL divergence | "Distance between distributions" | Not a true distance (not symmetric). Measures extra bits from using Q to encode P |
| Wasserstein distance | "Earth mover's distance" | Minimum work to transport mass from one distribution to another. A true metric |
| Approximate nearest neighbor | "ANN search" | Algorithms (HNSW, LSH, IVF) that find approximately closest points much faster than exact search |
| HNSW | "The vector DB algorithm" | Hierarchical Navigable Small World graph. Multi-layer graph for fast approximate nearest neighbor search |
| L1 regularization | "Lasso" | Adding the L1 norm of weights to the loss. Drives weights to zero (sparsity) |
| L2 regularization | "Ridge" or "weight decay" | Adding the squared L2 norm of weights to the loss. Shrinks weights toward zero without sparsity |
| Elastic Net | "L1 + L2" | Combines L1 and L2 regularization. Handles correlated feature groups better than either alone |

## Đọc thêm

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)- Thư viện Meta cho tìm kiếm ANN tỷ tỷ
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875)- tờ báo giới thiệu khoảng cách của Earth Mover với GAN
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876)- thuật toán ANN cơ bản
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)- Word2Vec, nơi sự tương đồng cosine trở thành mặc định cho các nhúng
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html)- hướng dẫn thực tế về métrics khoảng cách và thuật toán hàng xóm trong scikit-learn
