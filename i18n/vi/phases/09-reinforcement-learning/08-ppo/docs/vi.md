# Tích cực chính sách gần (PPO)

> A2C ném đi mỗi rollout sau một bản cập nhật. PPO gói gradient chính sách trong một tỷ lệ quan trọng bị cắt giảm để bạn có thể làm 10+ thời đại trên cùng một dữ liệu mà không bị chính sách nổ. Schulman et al. (2017).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## Vấn đề

A2C (Dạy học 07) là chính sách: gradient `E_{π_θ}[A · ∇ log π_θ]`yêu cầu dữ liệu lấy mẫu từ *current* `π_θ`Hãy cập nhật một lần, và`π_θ`thay đổi; dữ liệu mà bạn đã sử dụng bây giờ là không chính sách. sử dụng lại nó và gradient của bạn là thiên vị.

Các rollout là đắt tiền. trên Atari, một rollout trên 8 envs × 128 bước = 1024 chuyển đổi và một chục giây thời gian môi trường. Thả đi sau một bước gradient là lãng phí.

Tích cực chính sách khu vực tin cậy (TRPO, Schulman 2015) là sự khắc phục đầu tiên: hạn chế mỗi bản cập nhật để sự khác biệt KL giữa chính sách cũ và mới vẫn ở dưới `δ`Về lý thuyết sạch, nhưng đòi hỏi phải giải quyết các hàm kết hợp mỗi lần cập nhật.

PPO (Schulman et al. 2017) thay thế hạn chế khu vực tin tưởng cứng bằng một mục tiêu đơn giản. Một dòng mã thêm. Mười thời đại mỗi triển khai. Không có gradient kết hợp. Đảm bảo lý thuyết đủ tốt. Chín năm sau nó vẫn là thuật toán chính sách-gradient mặc định cho mọi thứ từ MuJoCo đến RLHF.

## Khái niệm

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**The importance ratio.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Đây là tỷ lệ xác suất của chính sách mới so với chính sách thu thập dữ liệu. `r_t = 1`nghĩa là không có thay đổi.`r_t = 2`nghĩa là chính sách mới có khả năng gấp đôi`a_t`như cũ.

**The clipped surrogate.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Hai điều khoản:

- Nếu lợi thế `A_t > 0`và tỷ lệ cố gắng tăng lên qua `1 + ε`, clip làm phẳng gradient  không đẩy một hành động tốt hơn `+ε`trên khả năng cũ.
- Nếu lợi thế `A_t < 0`và tỷ lệ cố gắng tăng lên qua `1 - ε`(có nghĩa là chúng ta sẽ làm cho một hành động xấu có khả năng hơn so với việc cắt giảm), clip                                                                                                                                                                                                                                                   `-ε`- Tôi không biết.

- `min`xử lý hướng khác: nếu tỷ lệ đã di chuyển theo hướng * hữu ích *, bạn vẫn nhận được độ nghiêng (không cắt trên mặt mà sẽ làm tổn thương bạn).

Thông thường`ε = 0.2`- Định hướng mục tiêu như một hàm của`r_t`: một chức năng tuyến tính theo mảnh với mái nhà phẳng trên "mặt tốt" và sàn phẳng trên "mặt xấu".

**The full PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Tương tự cấu trúc diễn viên-chính trị như A2C. Ba hệ số, thường `c_v = 0.5`- `c_e = 0.01`- `ε = 0.2`- Tôi không biết.

**The training loop.**

1. Thu thập`N × T`chuyển đổi qua `N`các môi trường song song cho `T`từng bước.
2. Xét lợi thế (GAE), đóng băng chúng như là các định vị.
3. Đóng `π_{θ_old}`như một bức ảnh chụp của hiện tại `π_θ`- Tôi không biết.
4. Vì `K`các thời đại, cho mỗi mini batch của `(s, a, A, V_target, log π_old(a|s))`- Có thể là:
   - Lưu ý`r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`- Tôi không biết.
   - Đơn `L^{CLIP}`+ mất giá trị + entropy.
   - Bước tiến.
5. Thả ra việc triển khai, quay lại bước 1.

`K = 10`và các bộ mini 64 là một bộ siêu tham số tiêu chuẩn. PPO là mạnh mẽ: các con số chính xác hiếm khi quan trọng trong ± 50%.

**KL-penalty variant.**Bài báo ban đầu đề xuất một lựa chọn thay thế bằng cách sử dụng một hình phạt KL thích nghi: `L = L^{PG} - β · KL(π_θ || π_old)`với `β`phiên bản cắt giảm trở nên thống trị; biến thể KL tồn tại trong RLHF (nơi KL đối với chính sách tham chiếu là một hạn chế riêng biệt bạn luôn muốn bất cứ lúc nào).

```figure
ppo-clip
```

## Hãy xây dựng nó

### Bước 1: bắt `log π_old(a | s)`tại thời điểm triển khai

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Hình ảnh được chụp một lần, vào thời điểm triển khai. Nó không thay đổi trong thời gian cập nhật.

### Bước 2: tính toán lợi ích của GAE (Dạy học 07)

Tương tự như A2C. Thường hóa trên toàn bộ lô.

### Bước 3: Clip cập nhật thay thế

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Mô hình "cắt → gradient không" là cốt lõi của PPO. Nếu chính sách mới đã trôi xa quá xa theo hướng có lợi, việc cập nhật sẽ dừng lại.

### Bước 4: giá trị và entropy

Thêm MSE tiêu chuẩn vào mục tiêu phê bình và một phần thưởng entropy cho người chơi, giống như A2C.

### Bước 5: Chẩn đoán

Ba điều cần xem mỗi lần cập nhật:

- **Mean KL** `E[log π_old - log π_θ]`- Tôi nên ở lại.`[0, 0.02]`Nếu nó qua đi`0.1`, giảm `K_EPOCHS`hoặc `LR`- Tôi không biết.
- **Clip fraction** phần mẫu có tỷ lệ nằm bên ngoài `[1-ε, 1+ε]`- Phải .`~0.1-0.3`Nếu`~0`, clip không bao giờ kích hoạt → tăng `LR`hoặc `K_EPOCHS`Nếu`~0.5+`, bạn đang quá phù hợp với việc triển khai → hạ thấp chúng.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`- Đường đo chất lượng quan trọng. nên tăng lên 1 khi người phê bình học.

## Những bẫy

- **Clip coefficient mistuned.** `ε = 0.2`là tiêu chuẩn thực tế.`0.1`làm cho các bản cập nhật quá nhút nhát; `0.3+`- Không ổn định.
- **Too many epochs.** `K > 20`thường xuyên gây bất ổn vì chính sách di chuyển xa hơn `π_old`Thời đại giới hạn, đặc biệt là cho các mạng lớn.
- **No reward normalization.**Tỷ lệ phần thưởng lớn ăn vào phạm vi clip.
- **Forgetting advantage normalization.**Thường hợp bình thường hóa trung bình không/mức đơn vị là tiêu chuẩn.
- **Learning rate not decayed.**PPO được hưởng lợi từ LR tuyến tính suy giảm đến không. LR liên tục thường tồi tệ hơn.
- **Importance ratio math errors.**Luôn luôn`exp(log_new - log_old)`cho sự ổn định số, không `new / old`- Tôi không biết.
- **Wrong gradient sign.**Tăng cường người thay thế = * giảm thiểu* `-L^{CLIP}`Một dấu hiệu đảo ngược là lỗi PPO phổ biến nhất.

## Sử dụng nó

PPO là thuật toán RL mặc định của năm 2026 trên một số miền đáng ngạc nhiên:

| Use case | PPO variant |
|----------|-------------|
| MuJoCo / robotics control | PPO with Gaussian policy, GAE(0.95) |
| Atari / discrete games | PPO with categorical policy, rolling 128-step rollouts |
| RLHF for LLMs | PPO with KL penalty to reference model, reward from RM at end of response |
| Large-scale game agents | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO (Lesson 12) — PPO variant without critic |
| Preference-only data | DPO — closed-form collapsing of PPO+KL, no online sampling |

PPO *hình dạng mất mát*  cắt thay thế + giá trị + entropy  là nền tảng cho DPO, GRPO và gần như mọi đường ống RLHF.

## Chuyển nó

Cứ như `outputs/skill-ppo-trainer.md`- Có thể là:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Các bài tập

1. **Easy.**Tiếp tục PPO trên 4×4 GridWorld với `ε=0.2, K=4`So sánh hiệu quả mẫu với A2C (một thời đại mỗi lần triển khai) tại các bước môi trường phù hợp.
2. **Medium.**Tháo `K ∈ {1, 4, 10, 30}`- Trình quay lại vs bước env và theo dõi trung bình KL mỗi bản cập nhật.`K`KL sẽ nổ tung trong nhiệm vụ này?
3. **Hard.**Thay thế cho người thay thế bị cắt bằng một hình phạt KL thích ứng (`β`tăng gấp đôi nếu `KL > 2·target`, giảm một nửa nếu `KL < target/2`). So sánh lợi nhuận cuối cùng, ổn định và không clip.

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; deviation from the policy that collected the data. |
| Clipped surrogate | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; flat gradient past the clip on beneficial side. |
| Trust region | "TRPO / PPO intent" | Limit each update's KL to guarantee monotone improvement. |
| KL penalty | "Soft trust region" | Alternative PPO: `L - β · KL(π_θ \|\| π_old)`. Adaptive `β`. |
| Clip fraction | "How often clipping triggers" | Diagnostic — should be 0.1-0.3; outside means mistuned. |
| Multi-epoch training | "Data reuse" | K epochs on each rollout; variance cost traded for sample efficiency. |
| On-policy-ish | "Mostly on-policy" | PPO is nominally on-policy but K>1 epochs uses slightly-off-policy data safely. |
| PPO-KL | "The other PPO" | KL-penalty variant; used in RLHF where KL-to-reference is already a constraint. |

## Đọc thêm

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)- Báo.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)TRPO, người tiền nhiệm của PPO.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) mọi siêu tham số PPO bị loại bỏ.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) InstructGPT; công thức PPO-in-RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) Khám phá hiện đại sạch sẽ với PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) chỉ dẫn PPO đơn tập tin được sử dụng bởi nhiều bài báo.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) công thức sản xuất cho PPO trên các mô hình ngôn ngữ; đọc cùng với Bài học 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) bài báo "37 tối ưu hóa cấp độ mã"; những thủ thuật PPO nào chịu tải và là câu chuyện dân gian.
