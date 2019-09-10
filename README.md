# DeepNSM
RL环境描述：
State：1. 网络状态：节点-链接使用情况 2. 切片请求 3. 预算出的路径
Action：从预算出的路径中选择一条
Reward: 建立成功+1，建立失败-1
问题描述：
简易版本：每一个切片都是一个service chain: RRH-BBU-Server, 每一个切片请求按照poisson过程到达，切片holding time服从指数分布，切片的源节点随机生成。每一个切片的源节点随机，目的节点固定（数据中心），RRH永远放置在源节点，Server永远在数据中心，BBU则需要策略决定。
模型算法描述：
