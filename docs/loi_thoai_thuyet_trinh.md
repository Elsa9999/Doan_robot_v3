# KỊCH BẢN LỜI NÓI (SPEAKER NOTES) - BẢO VỆ ĐỒ ÁN
*Tài liệu này chứa từng câu chữ chi tiết bạn sẽ cầm đọc/học thuộc khi đứng thuyết trình.*

---

**[SLIDE 1 - TIÊU ĐỀ] (Mở đầu điềm tĩnh, rõ ràng)**
"Kính thưa quý thầy cô, em tên là [Tên của bạn]. Hôm nay, em rất vinh dự được trình bày đề tài đồ án môn học của mình: 'Hệ thống Điều khiển & Mô phỏng Robot UR5e — Ứng dụng Học Tăng Cường cho bài toán Pick & Place'. Mong quý thầy cô lắng nghe và góp ý."

**[SLIDE 2 - ĐẶT VẤN ĐỀ]**
"Để bắt đầu, chúng ta hãy nhìn vào thực tế sản xuất công nghiệp. Thao tác Pick & Place — tức là gắp và đặt vật thể — là một trong những tác vụ phổ biến nhất. Tuy nhiên, nó luôn đặt ra thách thức lớn: vật thể thường nằm ở các vị trí ngẫu nhiên, và việc lên lịch trình quỹ đạo cho cánh tay 6 bậc tự do né tránh va chạm đòi hỏi những thuật toán rất phức tạp."

**[SLIDE 3 - MỤC TIÊU ĐỒ ÁN]**
"Chính vì lý do đó, mục tiêu của đồ án em là xây dựng một trạm làm việc mô phỏng cho robot UR5e. Điểm nhấn của hệ thống là việc tích hợp cả 2 phương pháp điều khiển: Điều khiển lập trình cứng thông thường (Auto FSM) và phương pháp tự học thông minh thông qua thuật toán Học Tăng Cường - Reinforcement Learning. Mục tiêu lớn nhất là chứng minh hệ thống AI có khả năng hoạt động tốt ngang ngửa hoặc vượt trội so với lập trình cứng."

**[SLIDE 4 - PHẠM VI & MÔI TRƯỜNG]**
"Về phạm vi, em sử dụng bộ mô phỏng vật lý PyBullet chạy ở tốc độ 240Hz để đảm bảo tính chân thực. Đối tượng tác động là robot kéo và các vật thể hình trụ 100g, sử dụng một đầu hút chân không được mô phỏng động lực học để gắp vật."

**[SLIDE 5 - CÔNG NGHỆ ÁP DỤNG]**
"Hệ thống của em được phát triển hoàn toàn bằng Python, với trí tuệ nhân tạo được xây dựng trên framework Stable-Baselines3 sử dụng PyTorch. Để người dùng cuối dễ thao tác, em cũng đã đóng gói một giao diện HMI chuyên nghiệp bằng thư viện PyQt5."

**[SLIDE 6 - KIẾN TRÚC TỔNG THỂ]**
"Nhìn vào sơ đồ kiến trúc này, hệ thống được chia thành 4 tầng rõ rệt hoạt động độc lập: Tầng Giao diện HMI trên cùng, Tầng Thuật toán xử lý Động học, Tầng Thuật toán AI, và Tầng dưới cùng là phần lõi mô phỏng trực tiếp PyBullet. Các dữ liệu được xử lý phi song song tránh nghẽn mạch."

**[SLIDE 7 - ĐỘNG HỌC THUẬN FK] (Chỉ tay vào bảng và hình)**
"Về mặt lõi toán học, em áp dụng phương pháp Động học Thuận theo quy ước Standard Denavit-Hartenberg (DH Tiêu chuẩn). Như quý thầy cô thấy trên bảng thông số, các độ dài tay đòn và độ lệch cổ tay được tham chiếu đúng 100% so với thông số thiết kế cơ khí ban đầu của dòng robot lõi e-Series từ Universal Robots."

**[SLIDE 8 - ĐỘNG HỌC NGHỊCH IK]**
"Đối với bài toán Động học Nghịch, em đã lập trình một bộ giải Hybrid kết hợp cả 2 phương pháp: Đầu tiên giải quyết bằng Phương pháp Giải tích hình học để lấy tốc độ dưới 1 mili-giây. Trong trường hợp vật rơi vào các góc 'mù' (singularity), hệ thống sẽ tự động chuyển sang phương pháp Số học dùng thuật toán Tối ưu L-BFGS-B để ép lỗi sai số về 0 một cách an toàn."

**[SLIDE 9 - QUY HOẠCH QUỸ ĐẠO]**
"Để robot không bị giật cứng khi di chuyển, em triển khai nội suy quỹ đạo Joint Space theo hình thang (Trapezoidal profile) để tăng/giảm tốc mượt mà. Đặc biệt, nội suy Cartesian Linearity được dùng ở giai đoạn cắm đầu gắp xuống vật để đảm bảo tay bám chuyển động theo một đường tịnh tiến thẳng tuyệt đối."

**[SLIDE 10 - CHẾ ĐỘ AUTO FSM]**
"Đây là chế độ điều khiển theo lập trình truyền thống. Hệ thống vận hành như một cỗ máy trạng thái (State Machine) gồm 10 bước tuần tự. Nó hoạt động với độ thành công khá cao, nhưng nhược điểm là cực kỳ cứng nhắc: nếu vật rơi ra khỏi vùng cấu hình cho trước, nó sẽ không biết cách tự xoay sở."

**[SLIDE 11 - GIỚI THIỆU HỌC TĂNG CƯỜNG]**
"Để giải quyết sự cứng nhắc đó, em chuyển sang sử dụng Học tăng cường (Reinforcement Learning). Không cần lập trình tọa độ cụ thể, robot sẽ tự biến mình thành một điệp viên: Nhìn vào môi trường (State), Giao quyết định (Action), và nhận Điểm thưởng/phạt (Reward) để cải thiện hiệu suất sau hàng triệu lần sai nghiệm."

**[SLIDE 12 - THUẬT TOÁN SAC]**
"Thuật toán cốt lõi được chọn là Soft Actor-Critic (SAC). Sự vượt trội của SAC nằm ở chỗ nó không chỉ tối đa hóa Phần thưởng, mà còn tối đa hóa 'Entropy' – tức là luôn khuyến khích agent tìm tòi nhiều quỹ đạo bay mới. Kiến trúc của nó bao gồm mạng Nơ-ron Actor quyết định bước đi và 2 mạng Critic đánh giá độ hiệu quả."

**[SLIDE 13 - OBSERVATION & ACTION]**
"Đầu vào của AI là 17 chiều dữ liệu, bao gồm vị trí tương đối giữa tay kẹp, vật và thùng rác, kèm theo Quaternion góc xoay của bề mặt. Tương ứng, Robot sẽ xuất ra hành động 7 chiều (7D Action), bao gồm dịch chuyển XYZ, xoay cổ tay Roll-Pitch-Yaw và lệnh Bật/Tắt giác hút."

**[SLIDE 14 - THIẾT KẾ HÀM REWARD] (Nhấn mạnh sự sáng tạo ở đây)**
"Chỗ này là thách thức lớn nhất của đồ án. Nếu chỉ dùng hàm phạt (penalty), AI sẽ chây lười và tự động hack điểm bằng cách đứng im lượn lờ. Để khắc phục, em đã giải quyết bằng một giải pháp thông minh hơn: Chuyển đổi ràng buộc vào Môi trường Vật lý, kẹp cứng giới hạn góc Euler gắp vật. Giúp Hàm Reward vô cùng đơn giản, AI chỉ tập trung hạ thấp trọng tâm mà vẫn đẩm bảo chuẩn tư thế công nghiệp."

**[SLIDE 15 - CURRICULUM LEARNING]**
"Thay vì bắt AI học toàn bộ thao tác cùng lúc và tốn tuần để huấn luyện, em dùng phương pháp 'Tiến hoá': Phase 1 chỉ cho học cách gắp (Grasp), sau khi thuần thục, em truyền Weights Model đó sang Phase 2 để dạy mang tới thùng rác. Sự kết hợp Transfer Learning này giúp AI hội tụ thành công tuyệt đối chỉ mất chưa tới 2.5 tiếng đồng hồ với 5.5 triệu steps."

**[SLIDE 16 - KẾT QUẢ TRAINING]**
"Như bảng trên slide, ở giai đoạn Pick & Place phức tạp nhất, chúng em thiết lập chạy 20 môi trường hoàn toàn song song đa luồng. Kết quả đem lại FPS mô phỏng đạt ~600, và tỷ lệ thả vật thành công đạt chuẩn 100% không để rơi rớt một lần nào."

**[SLIDE 17 - SO SÁNH AUTO vs AI]**
"So sánh hai phương pháp: Cả hai đều đạt 100% tỉ lệ thành công. Tuy nhiên, thay vì tốn cả tháng trời lập trình logic, thì Mạng Nơ ron AI đã giải quyết linh hoạt trọn vẹn quỹ đạo chỉ thông qua việc định nghĩa Điểm cấu trúc. Tư thế thả vật hoàn toàn chuẩn chỉ như máy móc."

**[SLIDE 18 - DEMO TRỰC TIẾP] (Nếu thầy cô cho chạy Demo)**
"Sau đây, em xin phép chạy trực tiếp phần mềm HMI. Đây là giao diện của ứng dụng. (Vừa thao tác vừa nói) Đầu tiên là chuyển sang Tab Auto để robot đi theo đường mũi thẳng... Sau đó bật sang AI Mode, robot lập tức thể hiện khả năng di chuyển mượt mà, áp gắp vật một cách tối ưu bằng bộ não Nơ-ron."

**[SLIDE 19 - HẠN CHẾ & HƯỚNG PT]**
"Tuy nhiên, đồ án vẫn còn những mặt hạn chế. Hiện tại AI chỉ thích nghi tốt nhất trong Vùng Môi trường Đào tạo. Tương lai, mô hình sẽ cần Randomize lớn hơn, tích hợp Computer Vision như camera Intel RealSense để giải tỏa sự lệ thuộc vào tọa độ ảo, từ đó tiến tới áp dụng thẳng thuật toán này lên cánh tay UR5 thật ở xưởng công nghiệp."

**[SLIDE 20 - LỜI CẢM ƠN] (Mỉm cười, cúi đầu lẹ)**
"Đồ án môn học này sẽ không thể thành công nếu thiếu đi sự hướng dẫn tận tình của Giáo viên. Đồng thời, em xin gửi lời cảm ơn thầy cô đã dành thời gian lắng nghe phần trình bày. Em xin tiếp nhận mọi câu hỏi và nhận xét từ quý thầy cô ạ!"
