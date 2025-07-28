import paddle
import numpy as np

# paddle.linalg.solve(Tensor([10, 10],"float32"), Tensor([228170138, 10],"float32"), left=False
A = paddle.randn([10, 10], dtype='float32')
# B = paddle.randn([228170138, 10], dtype='float32')  # 你之前用的是这个 size，明显错了！
B = paddle.randn([214748365, 10], dtype='float32')  # 你之前用的是这个 size，明显错了！
# 2147483647
# 正确的示例：B.shape[0] == A.shape[0]
# B = paddle.randn([10, 1], dtype='float32')

X = paddle.linalg.solve(A, B, left=False)  # 求解 AX = B
print("X:", X.shape)