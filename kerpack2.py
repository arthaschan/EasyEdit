# ====================== 全局拦截Qwen3导入：必须放在第一行，无任何代码在前面 ======================
import sys
from types import ModuleType

# 1. 定义空模块，用于承接所有qwen3相关的导入请求
qwen3_module = ModuleType('qwen3')
qwen3_modeling = ModuleType('qwen3.modeling_qwen3')
qwen3_tokenization = ModuleType('qwen3.tokenization_qwen3')
qwen3_tokenization_fast = ModuleType('qwen3.tokenization_qwen3_fast')
qwen3_config = ModuleType('qwen3.configuration_qwen3')

# 2. 把空模块注册到Python的系统模块字典中，全局生效
sys.modules['transformers.models.qwen3'] = qwen3_module
sys.modules['transformers.models.qwen3.modeling_qwen3'] = qwen3_modeling
sys.modules['transformers.models.qwen3.tokenization_qwen3'] = qwen3_tokenization
sys.modules['transformers.models.qwen3.tokenization_qwen3_fast'] = qwen3_tokenization_fast
sys.modules['transformers.models.qwen3.configuration_qwen3'] = qwen3_config
# 兜底：注册所有可能的qwen3子模块，防止漏截
sys.modules['qwen3'] = qwen3_module


import huggingface_hub
import transformers
import peft
import datasets
# 验证版本
print(f'huggingface-hub版本：{huggingface_hub.__version__}（要求0.23.2<=版本<1.0）')
print(f'版本是否符合要求：0.23.2 <= float(huggingface_hub.__version__[:3]) < 1.0')
# 验证核心库正常导入
print(f'transformers/peft/datasets导入成功：', all([x in locals() for x in ['transformers','peft','datasets']]))
