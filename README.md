# surround_depth

修改为ddad_tiny.json的绝对路径（根目录开始，train,val,test三个路径均需要修改）

## Train DDAD_tiny:

处理好上述问题，cd到surround_depth文件夹下，运行指令

`python scripts/train_new.py configs/train_ddad_tiny.yaml`

`CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 scripts/train_new.py configs/train_ddad_tiny.yaml`
## Ealuate pretrained model provided
`python scripts/eval_new.py --checkpoint /home/thuar/Desktop/surround_depth/PackNet01_MR_selfsup_D.ckpt --config configs/eval_ddad_tiny.yaml`

## 当前目标
- 在大数据集上运行程序
- 编写check point保存程序
- 检查所有计算的正确性

## 05.02更新
- 多相机数据的读取（包括相机外参extrinsics）
- Validmap的计算
- 新增Consistency Loss (optional)
- 在原先temporal loss的基础上添加spatial loss以及temporal-spatial loss

## 5.03更新
- 更正了valid map的计算，程序可正常执行至spatial_loss的计算

## 5.04更新
- 程序可正常执行至temporal_spatial_loss的计算
- 优化了loss计算中的循环，合并了spatial_loss和temporal_spatial_loss的计算函数
- 在程序中验证了下面两条：
- ref image分别为t-1图像和t+1图像，PoseNet返回的是两个“移动”，t->t-1和t->t+1
- 重建图片使用t-1和t+1重建t帧
- 使所有变量运算都在gpu上进行
- Network can be trained completely (with consistency loss)

## 5.05更新
- 可以使用官网预训练的模型进行evaluation
- 使用open3D库对输出进行点云拼接ing...
- 可以在完整的数据集上可以进行训练

## 5.06

