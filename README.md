# 磁盘description
| **index** |   **name**    | **type** |             **comment**             |
| :-------: | :-----------: | :------: | :---------------------------------: |
|     0     |     appid     |   int    |                                     |
|     1     |   disk_uid    |   str    |                盘ID                 |
|     2     | disk_instance |   str    |                                     |
|     3     |    vm_uid     |   str    |              虚拟机ID               |
|     4     |  create_time  |   date   |              创建时间               |
|     5     |  finsh_time   |   date   |              销毁时间               |
|     6     |    status     |   str    |                状态                 |
|     7     |   is_local    |   int    |        是否为本地盘（弃用）         |
|     8     |   disk_attr   |   str    | 磁盘属性（cbs、cbsPremium、cbsssd） |
|     9     |   disk_type   |   str    |       磁盘类型（data、root）        |
|    10     |     isvip     |   int    |              是否是vip              |
|    11     |   pay_mode    |   str    |     支付方式（postpay、prepay）     |
|    12     |   pay_type    |   int    |                弃用                 |
|    13     |    vm_name    |   str    |             虚拟机别名              |
|    14     |      cpu      |   int    |              CPU核心数              |
|    15     |    memory     |   int    |           虚拟机内存大小            |
|    16     |   app_name    |   str    |              业务名称               |
|    17     |   disk_name   |   str    |              磁盘别名               |
|    18     | project_name  |   str    |              项目名称               |
|    19     |  disk_usage   |   int    |         磁盘用量（单位 MB）         |
|    20     |   disk_size   |   int    |        磁盘容量（单位 GB）？        |

# warehouse
| **index** |   **name**    | **type** |        
| :-------: | :-----------: | :------: | 
|     0     |   disk_ID     |   str    |
|     1     | disk_capacity |   int    |
|     2     | disk_if_local |   int    |
|     3     |   disk_attr   |   str    | 
|     4     |   disk_type   |   str    |  
|     5     |  disk_if_VIP  |   int    |    
|     6     |   disk_pay    |   str    |   
|     7     |    vm_cpu     |   int    |   
|     8     |    vm_mem     |   int    |    
|     9     | average_disk_RBW |   float    |      
|    10     | average_disk_WBW |   float    |  
|    11     |  peak_disk_RBW |   float    |    
|    12     |  peak_disk_WBW |   float    |
|    13     | disk_timestamp_num |   int    |  
|    14     |   burst_label |   int    | 

