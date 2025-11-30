import time
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn
)

# 1. 配置全白色的进度条
progress = Progress(
    # [动态文本列] 显示 Epoch 信息，使用 {task.description} 占位
    TextColumn("[bold white]{task.description}", justify="right"),

    # [进度条列] 设定轨道为暗白(灰)，进度为纯白
    BarColumn(
        bar_width=40,
        style="dim white",       # 轨道颜色
        complete_style="white",  # 完成部分颜色
        finished_style="white"   # 完成后的颜色
    ),

    # [百分比列] 强制白色
    TaskProgressColumn(style="white"),

    # [时间列] 自定义格式：只显示秒数，例如 "12.5s"
    # 这里直接调用 task.elapsed 获取耗时
    TextColumn("[white]{task.elapsed:.1f}s"),
)

# 模拟训练参数
total_epochs = 3
steps_per_epoch = 50

with progress:
    # 创建一个任务，初始描述为 Epoch 1
    task_id = progress.add_task(f"Epoch 1/{total_epochs}", total=steps_per_epoch)

    for epoch in range(1, total_epochs + 1):
        # 每一轮 Epoch 开始前，更新描述文本
        progress.update(task_id, description=f"[bold white]Epoch {epoch}/{total_epochs}")
        
        # 重置进度条 (如果是新的一轮)
        progress.reset(task_id)
        
        # 模拟 Steps 训练
        for step in range(steps_per_epoch):
            time.sleep(0.05) # 模拟计算耗时
            progress.update(task_id, advance=1)
            
        # (可选) 如果希望每跑完一个Epoch保留一行记录，可以使用 print
        # progress.console.print(f"[white]Epoch {epoch} finished.")
