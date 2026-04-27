"""单任务调试前端（Streamlit）。

子模块拆分：
- `loaders.py`: run/task/gold/prediction 的纯 IO 加载层（无 streamlit 依赖，可单测）
- `compare.py`: gold vs prediction 比对（纯函数）
- `streamlit_app.py`: streamlit 主入口（仅在显式 `dabench view` 时被启动）

只有 `streamlit_app.py` 依赖 streamlit；loaders / compare 不依赖，因此即便用户没装
viz 分组也能 import 它们用于其他工具脚本。
"""
