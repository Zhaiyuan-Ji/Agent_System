# run_Qwen3-VL-8B-Instruct_sglang.py
import subprocess
import sys
import os

# 设置 GPU 可见性（必须在启动子进程前设置）
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", "/mnt/users/ylu/jzy/model/Qwen3-4B-Instruct-2507/",
        "--port", "54329",
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--tp", "1",                     # tensor parallel size
        "--mem-fraction-static", "0.82",       # 显存占用比例（根据你的 GPU 调整）
        "--served-model-name", "gpt-5.1",
        "--log-level", "info",
        "--trust-remote-code",                # Qwen-VL 必须加！
        "--tool-call-parser", "qwen",       # 可选：仅当确认模型支持时启用
    ]

    print("Running command:", " ".join(cmd))
    # 使用 subprocess.run 会阻塞当前进程（类似 vLLM 脚本）
    subprocess.run(cmd)