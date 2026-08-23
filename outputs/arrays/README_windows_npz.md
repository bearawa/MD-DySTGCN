# 窗口化样本 train/val/test.npz

因 GitHub 单次 push 体积限制，分卷压缩包发布在 Release：

**https://github.com/bearawa/cursor/releases/tag/dataset-windows-v1**

```bash
cd outputs/arrays
# 从上述 Release 下载全部 windows_npz.z* 与 windows_npz.zip 到本目录后：
zip -F windows_npz.zip --out windows_npz_full.zip
unzip windows_npz_full.zip
# 得到 train.npz / val.npz / test.npz
```

也可从仓库根目录的 `原始数据/` 重新跑预处理生成：

```bash
python -m preprocess.run_preprocess
```
