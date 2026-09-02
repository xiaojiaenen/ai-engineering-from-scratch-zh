# A2A  Phương pháp giao dịch giữa các đại lý

> MCP là đại lý-đồ dùng. A2A (Agent2Agent) là một giao thức mở để cho phép các đại lý không minh bạch được xây dựng trên các khung khác nhau hợp tác. Được phát hành bởi Google vào tháng 4 năm 2025, được quyên góp cho Quỹ Linux vào tháng 6 năm 2025, đạt v1.0 vào tháng 4 năm 2026 với 150 + người ủng hộ bao gồm AWS, Cisco, Microsoft, Salesforce, SAP và ServiceNow. Nó hấp thụ ACP của IBM và thêm mở rộng thanh toán AP2. Bài học này sẽ kể về thẻ đại lý, vòng đời nhiệm vụ và hai liên kết vận tải.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Mục tiêu học tập

- Sự khác biệt giữa các trường hợp sử dụng từ đại lý đến công cụ (MCP) và các trường hợp sử dụng từ đại lý đến đại lý (A2A).
- Giới thiệu thẻ đại lý tại `/.well-known/agent.json`với các kỹ năng và metadata điểm cuối.
- Đi theo vòng đời nhiệm vụ (đưa → làm việc → nhập-cần → hoàn thành / thất bại / hủy bỏ / từ chối).
- Sử dụng Thông điệp với các bộ phận (tin, tệp, dữ liệu) và đồ tạo như các đầu ra.

## Vấn đề

Một đại lý dịch vụ khách hàng cần ủy quyền viết báo cáo cho một đại lý viết chuyên ngành.

- REST API tùy chỉnh, hoạt động nhưng mỗi cặp đều là một lần.
- Hình thức chia sẻ mã, yêu cầu hai đại lý chạy cùng một khung.
- MCP không phù hợp: MCP là để gọi công cụ, không phải cho hai đại lý hợp tác trong khi vẫn giữ cho lý luận nội bộ không rõ ràng của mỗi đại lý.

A2A lấp đầy khoảng trống. Nó mô hình hóa sự tương tác khi một đại lý gửi một Task đến một đại lý khác, với chu kỳ sống, tin nhắn và đồ tạo. trạng thái nội bộ của đại lý được gọi vẫn không minh bạch  người gọi chỉ thấy chuyển đổi trạng thái nhiệm vụ và đầu ra cuối cùng.

A2A là giao thức "cho phép các đại lý trên các khung nói chuyện với nhau".

## Khái niệm

### Cảnh sát Thư

Mỗi đại lý tuân thủ A2A đều xuất bản một thẻ tại `/.well-known/agent.json`- Có thể là:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Khám phá dựa trên URL: lấy thẻ, tìm hiểu URL của điểm cuối A2A, liệt kê kỹ năng.

### Thẻ đại lý được ký (AP2)

Phiên bản mở rộng AP2 (Tháng 9 năm 2025) thêm chữ ký mật mã vào thẻ Agent. Một nhà xuất bản ký thẻ của riêng mình với một JWT; người tiêu dùng xác minh.

### Chuyển đời của nhiệm vụ

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Khách hàng bắt đầu với `tasks/send`Các đại lý được gọi chuyển qua các tiểu bang; khách hàng đăng ký cập nhật tiểu bang thông qua SSE hoặc thăm dò.

### Thông điệp và phần

Một tin nhắn chứa một hoặc nhiều phần:

- `text` nội dung đơn giản.
- `file` base64 blob với mimeType.
- `data` nhập tải trọng hữu ích JSON (tài nhập cấu trúc cho đại lý được gọi).

Ví dụ:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Các đồ tạo tác

Các sản phẩm là các sản phẩm, không phải chuỗi nguyên liệu. Một sản phẩm là một sản phẩm có tên, được gõ:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Các đồ tạo vật có thể được truyền qua các mảnh, người gọi sẽ tích lũy.

### Hai liên kết vận chuyển

1. **JSON-RPC over HTTP.** `/a2a`Endpoint, POST cho yêu cầu, SSE tùy chọn cho streaming.
2. **gRPC.**Đối với môi trường doanh nghiệp nơi gRPC là bản địa.

Cả hai liên kết đều có cùng một hình dạng thông điệp logic.

### Bảo tồn độ trống

Một nguyên tắc thiết kế chính: trạng thái nội bộ của đại lý được gọi là không minh bạch. Người gọi thấy trạng thái nhiệm vụ và các vật liệu. Dòng tư tưởng của đại lý được gọi, các cuộc gọi công cụ của nó, đại diện phụ của nó  tất cả đều vô hình. Điều này khác với MCP, nơi các cuộc gọi công cụ là minh bạch.

Lý luận: A2A cho phép các đối thủ cạnh tranh hợp tác mà không tiết lộ nội bộ. A2A có thể là "hãy gọi cho đại lý dịch vụ khách hàng này" mà không cần người gọi học cách đại lý đó thực hiện dịch vụ.

### Thời gian

- **2025-04-09.**Google công bố A2A.
- **2025-06-23.**Được tặng cho Linux Foundation.
- **2025-08.**Thuốc ACP của IBM.
- **2025-09.**Tàu mở rộng AP2 (Giá nhân viên).
- **2026-04.**v1.0 được phát hành với 150+ tổ chức hỗ trợ.

### Mối quan hệ với MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

Sử dụng MCP khi bạn muốn gọi một công cụ cụ thể. Sử dụng A2A khi bạn muốn ủy thác toàn bộ nhiệm vụ cho một đại lý khác. Nhiều hệ thống sản xuất sử dụng cả hai: một đại lý sử dụng MCP cho lớp công cụ của mình và A2A cho lớp hợp tác của mình.

```figure
a2a-task-lifecycle
```

## Sử dụng nó

`code/main.py`thực hiện một vòng A2A tối thiểu: một đại lý nghiên cứu xuất bản thẻ của mình, một đại lý viết nhận được một `tasks/send`với các bộ phận bao gồm một PDF và một hướng dẫn văn bản, chuyển đổi thông qua làm việc → input_required → working → hoàn thành, và trả lại một vật liệu văn bản.

Những gì cần xem:

- Hình dạng JSON của thẻ đại lý.
- Đề xuất ID nhiệm vụ và chuyển đổi trạng thái.
- Thông điệp với các bộ phận hỗn hợp.
- Các chi nhánh cần phải nhập vào giữa nhiệm vụ.
- Các vật liệu sẽ được trả về khi hoàn thành.

## Chuyển nó

Bài học này sẽ mang lại kết quả `outputs/skill-a2a-agent-spec.md`Với một đại lý mới mà nên được gọi bởi các đại lý khác, kỹ năng tạo ra thẻ đại lý JSON, kế hoạch kỹ năng và bản phác thảo điểm cuối.

## Các bài tập

1. Đi chạy`code/main.py`. Theo dõi toàn bộ vòng đời nhiệm vụ, bao gồm cả thời gian tạm dừng cần thiết khi người gọi yêu cầu giải thích.

2. Thêm một thẻ đại lý được ký, ký với HMAC trên thẻ JSON có thể xác minh và xác nhận nó thất bại trên một thẻ đột biến.

3. Thực hiện truyền tải nhiệm vụ: đại lý viết phát ra ba khối tạo vật tăng lên trên SSE và người gọi tích lũy chúng.

4. Thiết kế một đại lý A2A bao gồm một máy chủ MCP. Bản đồ mỗi công cụ MCP để một kỹ năng A2A. Lưu ý các sự thỏa hiệp  bất độ sáng bị mất?

5. Đọc thông báo A2A v1.0 và xác định một tính năng chưa được thực hiện bởi bất kỳ khung nào vào tháng 4 năm 2026. (Thông dụ: nó liên quan đến ủy quyền nhiệm vụ đa hop).

## Các điều khoản chính

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`.well-known/agent.json`" | Published metadata describing an agent's skills and endpoint |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (text, file, data) |
| Part | "Typed chunk" | `text` / `file` / `data` element of a message |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Signed Agent Cards extension for trust and payments |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| Input-required | "Task pause" | Lifecycle state when the agent needs more info |

## Đọc thêm

- [a2a-protocol.org](https://a2a-protocol.org/latest/) Cấu chỉ A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) Các thực hiện và SDK tham chiếu
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) Tháng 6 năm 2025 chuyển giao quản trị
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) Bản đồ đường và động lực của đối tác
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) v1.0 thông báo phát hành và hướng dẫn ngược
