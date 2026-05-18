# run_Qwen3-VL-8B-Instruct_sglang.py
import subprocess
import sys
import os

# 设置 GPU 可见性（必须在启动子进程前设置）
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", "/mnt/users/ylu/jzy/model/Qwen3-Embedding-4B/",
        "--port", "54331",
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--tp-size", "1",                     # tensor parallel size
        "--mem-fraction-static", "0.60",       # 显存占用比例（根据你的 GPU 调整）
        "--served-model-name", "text-embedding-3-small",
        "--log-level", "info",
        "--trust-remote-code",                # Qwen-VL 必须加！
        "--is-embedding"
    ]

    print("Running command:", " ".join(cmd))
    # 使用 subprocess.run 会阻塞当前进程（类似 vLLM 脚本）
    subprocess.run(cmd)