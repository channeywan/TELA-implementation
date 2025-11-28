import json
import pandas as pd
from tqdm import tqdm
import re
from openai import AsyncOpenAI
import os
import asyncio
import logging
from config.settings import DataConfig, ModelConfig, DirConfig, WarehouseConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
# SYSTEM_PROMPT = """你是一个专业的云业务场景分类器。你的任务是根据业务描述,联想分析后从"可用标签列表"中选择一个最相关的标签。
# ## 可用标签列表:
# ["核心数据库", "数据仓库", "日志监控", "web应用", "办公系统", "门户网站", "游戏服务器", "游戏日志", "容器Master", "容器Node","系统盘","备份归档","开发测试","未知待定"]
# ## 格式要求
# 你的回答必须有且仅有一个有效的JSON对象,不要包含任何JSON之外的文本、解释和Markdown标记。
# ## 结果(JSON格式):
# {
#     "label": "标签名称"
# }
# """
SYSTEM_PROMPT = """你是一个专业的云业务场景分类器。你的任务是根据业务描述(通常是虚拟机名称),联想分析后从"可用标签列表"中选择一个最相关的标签。
## 可用标签列表 (按业务功能划分):

1.  **game-service** 
    * 描述:承载核心游戏逻辑、战斗、场景、GM后台或游戏日志的服务器。
2.  **office-system**
    * 描述:企业内部员工使用的系统,如OA (办公自动化), CRM (客户关系管理), ERP (企业资源规划)。
3.  **gov-public-service**
    * 描述:为政府、学校、医院、事业单位(如水利、民政、派出所)提供的门户或业务系统。
4.  **corp-website-portal** 
    * 描述:企业的官方网站、信息展示型门户、品牌落地页(非新闻或电商)。
5.  **ecommerce-retail** 
    * 描述:在线零售电商平台、B2C/C2C商店、分销系统、营销推广或返利服务。
6.  **local-service-delivery** 
    * 描述:外卖、本地生活服务(如家政、票务)或O2O(线上到线下)平台。
7.  **media-video-streaming** 
    * 描述:提供视频点播(VOD)、直播、音视频处理、转码或CG渲染(如“效果图”、“动画”)的服务。
8.  **media-news-portal** 
    * 描述:新闻资讯类网站,或内容聚合门户。
9.  **finance-payment** 
    * 描述:处理金融业务、支付网关、账单、清算、记账相关的服务。
10. **data-collection-delivery** 
    * 描述:用于数据采集的爬虫(Spider),或用于数据分发的CDN类边缘节点服务。
11. **ai-machine-learning** 
    * 描述:用于AI模型训练、推理或数据科学(如基金分析)的计算服务,通常GPU密集型。
12. **dev-test-env** 
    * 描述:明确用于开发、测试、UAT(用户验收测试)或Staging(预发)的非生产环境。
13. **education-learning** 
    * 描述:在线教育平台、电子学习(E-learning)、校园管理或在线课程相关的应用(如“智行学院”)。
14. **community-social-forum** 
    * 描述:通讯、社区论坛(BBS)、社交网络、个人博客平台。
15. **compute-simulation** 
    * 描述:科学计算、HPC(高性能计算)、数据仿真或非AI/非批处理的通用计算任务。
16. **personal-use** 
    * 描述:用于个人项目、自托管、家庭服务器等非商业、非机构用途。
17. **iot-saas-platform** 
    * 描述:物联网 (IoT) 数据平台、SaaS (软件即服务) 平台,或商业监控平台。
18. **logistics-mobility** 
    * 描述:物流 (WMS), 仓储, 供应链, 或出行 (网约车, 共享单车) 服务 (如 `pro-wms-logistics-basic`)。
19. **travel-hospitality** 
    * 描述:差旅、酒店预订、机票、旅游相关的在线服务。
20. **infra-node** 
    * 描述:基础设施计算/存储节点,如K8s/TKE worker节点或通用节点池 (e.g., `st2-pknode`, `tke_cls-..._worker`)。
21. **infra-coordination** 
    * 描述:基础设施协调服务,如ZooKeeper, Etcd (e.g., `st2-zknode`, `etcd`)。
22. **infra-database** 
    * 描述:专用的数据库服务,如MySQL, MongoDB, Cassandra, SQL Server等。
23. **infra-message-queue** 
    * 描述:专用的消息中间件基础设施,如Kafka, RabbitMQ。
24. **infra-cloud-function** 
    * 描述:专用的Serverless(无服务器)云函数执行环境 (e.g., `scf_function...`)。
25. **infra-jumpbox** 
    * 描述:专用的堡垒机或跳板机,用于安全登录和运维管理 (e.g., `QY-跳板机`)。
26. **infra-cache** 
    * 描述:专用的内存缓存服务,如Redis, Memcached。
27. **infra-logging-monitoring** 
    * 描述:专用的基础设施日志收集、监控和告警系统,如ELK, Zabbix, Prometheus。
28. **generic-autoscaling** 
    * 描述:由ASG(自动伸缩组)动态创建,业务归属不明 (e.g., `as-cpu-...`, `asg-...`)。
29. **generic-unknown** 
    * 描述:无法分类的、命名不规范的、或云平台默认名称的VM(如 "activity-cvm...", "未命名", "centos", "我的云服务器")。

## 格式要求
你的回答必须有且仅有一个有效的JSON对象,不要包含任何JSON之外的文本、解释和Markdown标记。
## 结果(JSON格式):
{
   "label": "标签名称"
}

"""


def merge_string(row):
    string_list = {s.strip() for s in row.values.astype(str) if s.strip()}
    to_remove = set()
    for s_i in string_list:
        for s_j in string_list:
            if s_i != s_j and s_i in s_j:
                to_remove.add(s_i)
                break
    return '_'.join(sorted([s for s in string_list if s not in to_remove]))


def process_description():
    items_with_description = []
    for cluster_index in DataConfig.REQUEST_BUSINESS_TYPE_CLUSTER_INDEX_LIST:
        description_dir = os.path.join(
            DirConfig.CLUSTER_INFO_ROOT,
            f"cluster{cluster_index}"
        )
        df = pd.read_csv(description_dir, sep=',')
        df['description'] = df[["vm_name", "project_name", "buss_name", "disk_name"]].fillna(
            '').replace("未命名", '').apply(merge_string, axis=1)
        df['cluster_index'] = cluster_index
        items_with_description.append(df)
    return pd.concat(items_with_description, axis=0)


async def process_single_description(client: AsyncOpenAI, description: str, semaphore: asyncio.Semaphore, max_retries=1):
    classification = ""
    error = None
    cleaned_output = ""
    cache_hit_tokens = 0
    cache_miss_token = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": description}]
    async with semaphore:
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = await client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                cache_hit_tokens = response.usage.prompt_cache_hit_tokens
                cache_miss_token = response.usage.prompt_cache_miss_tokens
                output = response.choices[0].message.content
                cleaned_output = re.sub(r'(\r|\n|\r\n)', '', output.strip())
                try:
                    json_obj = json.loads(output)
                    classification = json_obj.get("label", "")
                    break
                except json.JSONDecodeError as json_error:
                    if retry_count >= max_retries:
                        error = f"No valid JSON found: {str(json_error)}"
                        break
                    else:
                        retry_count += 1
                        continue
            except Exception as e:
                if hasattr(e, 'message') and hasattr(e, 'status_code'):
                    error = f"API Error: {e.message},Error code: {e.status_code}"
                elif isinstance(e, json.JSONDecodeError):
                    error = f"JSON Decode Error: {str(e)}"
                else:
                    error = f"Error: {str(e)}"
                break
    return description, classification, cleaned_output, error, cache_hit_tokens, cache_miss_token


async def request_business_type():
    # items_with_description = process_description()
    items_with_description = pd.read_csv(os.path.join(DirConfig.TRASH_DIR,
                                                      "all_tencent_disk_description.csv"))
    client = AsyncOpenAI(api_key=os.getenv(
        "DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    semaphore = asyncio.Semaphore(20)
    items_df = items_with_description.copy()
    items_description = items_df["description"]
    task = [process_single_description(
        client, description, semaphore) for description in items_description]
    results_map = {}
    for future in tqdm(asyncio.as_completed(task), total=len(task)):
        description, classification, cleaned_output, error, cache_hit_tokens, cache_miss_token = await future
        results_map[description] = {
            "classification": classification,
            "cleaned_output": cleaned_output,
            "error": error,
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_token": cache_miss_token
        }
    print("inference finished")
    output_filename = os.path.join(
        DirConfig.BUSINESS_TYPE_DIR, "tencent_disk_business_type_results.txt")
    with open(output_filename, "w") as f:
        for description in items_description:
            classification = results_map[description]["classification"]
            cleaned_output = results_map[description]["cleaned_output"]
            error = results_map[description]["error"]
            cache_hit_tokens = results_map[description]["cache_hit_tokens"]
            cache_miss_token = results_map[description]["cache_miss_token"]
            f.write("description: " + description.strip() + "\n")
            f.write("output: " + cleaned_output + "\n")
            f.write("classification: " + classification + "\n")
            if error:
                f.write("error: " + error + "\n")
            f.write("cache_hit_tokens: " + str(cache_hit_tokens) + "\n")
            f.write("cache_miss_token: " + str(cache_miss_token) + "\n")
            f.write("\n")
    print("business_type results saved")
    items_df["business_type"] = items_df["description"].map(
        lambda desc: results_map.get(desc, {}).get("classification", ""))
    items_df.to_csv(os.path.join(DirConfig.BUSINESS_TYPE_DIR,
                                 "tencent_disk_business_type.csv"), index=False)
    # for cluster_index in DataConfig.REQUEST_BUSINESS_TYPE_CLUSTER_INDEX_LIST:
    #     df = items_df[items_df["cluster_index"] == cluster_index]
    #     df.to_csv(
    #         os.path.join(DirConfig.CLUSTER_INFO_BUSINESS_TYPE_ROOT, f"cluster{cluster_index}.csv"), index=False)


def run_request_business_type():
    asyncio.run(request_business_type())
